import csv

import numpy as np

_ELEVATION_CSV_CACHE = {}


def _resample_height_map(height_map, target_shape):
    src_x, src_y = height_map.shape
    dst_x, dst_y = int(target_shape[0]), int(target_shape[1])
    if dst_x <= 0 or dst_y <= 0:
        raise ValueError(f"target_shape must be positive, got {target_shape}")
    if (src_x, src_y) == (dst_x, dst_y):
        return height_map.copy()

    src_x_coords = np.linspace(0.0, src_x - 1, src_x, dtype=np.float32)
    src_y_coords = np.linspace(0.0, src_y - 1, src_y, dtype=np.float32)
    dst_x_coords = np.linspace(0.0, src_x - 1, dst_x, dtype=np.float32)
    dst_y_coords = np.linspace(0.0, src_y - 1, dst_y, dtype=np.float32)

    tmp = np.empty((dst_x, src_y), dtype=np.float32)
    for y_idx in range(src_y):
        tmp[:, y_idx] = np.interp(dst_x_coords, src_x_coords, height_map[:, y_idx])

    resampled = np.empty((dst_x, dst_y), dtype=np.float32)
    for x_idx in range(dst_x):
        resampled[x_idx, :] = np.interp(dst_y_coords, src_y_coords, tmp[x_idx, :])

    return resampled


def _scale_height_map(height_map, target_height):
    target_height = float(target_height)
    if target_height <= 0.0:
        raise ValueError(f"target_height must be positive, got {target_height}")

    src_min = float(np.min(height_map))
    src_max = float(np.max(height_map))
    if src_max - src_min <= 1e-8:
        return np.zeros_like(height_map, dtype=np.float32)

    normalized = (height_map - src_min) / (src_max - src_min)
    return (normalized * target_height).astype(np.float32)


def _box_blur_height_map(height_map, passes=1):
    passes = max(int(passes), 0)
    if passes == 0:
        return height_map.copy()

    blurred = height_map.astype(np.float32, copy=True)
    for _ in range(passes):
        padded = np.pad(blurred, ((1, 1), (1, 1)), mode="edge")
        blurred = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0
    return blurred.astype(np.float32)


def _simplify_height_map(height_map, stride=1, smooth_passes=0):
    stride = max(int(stride), 1)
    simplified = height_map.copy()

    if stride > 1:
        coarse_shape = (
            max(int(np.ceil(height_map.shape[0] / stride)), 2),
            max(int(np.ceil(height_map.shape[1] / stride)), 2),
        )
        simplified = _resample_height_map(height_map, coarse_shape)
        simplified = _resample_height_map(simplified, height_map.shape)

    if smooth_passes > 0:
        simplified = _box_blur_height_map(simplified, passes=smooth_passes)

    return simplified.astype(np.float32)


class TerrainHemisphere:
    def __init__(self, x, y, radius, height, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)
        self.height = float(height)
        self.z = float(z)

    def height_at(self, px, py):
        radial_term = 1.0 - ((px - self.x) / max(self.radius, 1e-8)) ** 2 - (
            (py - self.y) / max(self.radius, 1e-8)
        ) ** 2
        if radial_term <= 0.0:
            return 0.0
        return self.height * np.sqrt(radial_term)


def load_elevation_csv(csv_path):
    cached = _ELEVATION_CSV_CACHE.get(csv_path)
    if cached is not None:
        return cached.copy()

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = list(csv.reader(f))

    if not reader:
        raise ValueError(f"Empty terrain csv: {csv_path}")

    header = [cell.strip() for cell in reader[0]]
    has_named_header = {"Latitude", "Longitude", "Elevation"}.issubset(header)
    data_rows = reader[1:] if has_named_header else reader
    if not data_rows:
        raise ValueError(f"No terrain rows found in csv: {csv_path}")

    lon = []
    lat = []
    ele = []
    for row in data_rows:
        if len(row) < 3:
            continue
        if has_named_header:
            row_dict = dict(zip(header, row))
            lat.append(float(row_dict["Latitude"]))
            lon.append(float(row_dict["Longitude"]))
            ele.append(float(row_dict["Elevation"]))
        else:
            lon.append(float(row[0]))
            lat.append(float(row[1]))
            ele.append(float(row[2]))

    if not ele:
        raise ValueError(f"Failed to parse terrain csv: {csv_path}")

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    ele = np.asarray(ele, dtype=np.float32)

    first_lat = lat[0]
    first_lon = lon[0]
    width = int(np.sum(lat == first_lat))
    height = int(np.sum(lon == first_lon))
    total_points = len(ele)

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid terrain shape inferred from csv: {csv_path}")
    if total_points < width * height:
        raise ValueError(
            f"Terrain csv has insufficient points: expected {width * height}, got {total_points}"
        )

    ele = ele[: width * height]
    elevation = np.zeros((width, height), dtype=np.float32)
    for y_idx in range(height):
        for x_idx in range(width):
            elevation[x_idx, y_idx] = ele[y_idx * width + x_idx]

    _ELEVATION_CSV_CACHE[csv_path] = elevation
    return elevation.copy()


class TerrainHeightMap:
    def __init__(
        self,
        x_extent,
        y_extent,
        enabled=False,
        base_height=0.0,
        num_hemispheres=0,
        hemisphere_radius_range=(4.0, 10.0),
        hemisphere_height_range=None,
        edge_margin=0.0,
        resolution=(128, 128),
    ):
        self.x_extent = float(x_extent)
        self.y_extent = float(y_extent)
        self.enabled = bool(enabled)
        self.base_height = float(base_height)
        self.num_hemispheres = int(num_hemispheres)
        self.hemisphere_radius_range = hemisphere_radius_range
        self.hemisphere_height_range = (
            hemisphere_radius_range
            if hemisphere_height_range is None
            else hemisphere_height_range
        )
        self.edge_margin = float(edge_margin)
        self.resolution = resolution

        self.features = []
        self.grid_x = None
        self.grid_y = None
        self.height_map = None

    def clear(self):
        self.features = []
        self.grid_x = None
        self.grid_y = None
        self.height_map = None

    def _is_feature_overlapping(self, x, y, radius):
        for feature in self.features:
            min_dist = radius + feature.radius
            if (x - feature.x) ** 2 + (y - feature.y) ** 2 < min_dist ** 2:
                return True
        return False

    def generate(self):
        self.clear()
        if not self.enabled:
            return

        max_attempts = max(100, self.num_hemispheres * 50)
        attempts = 0
        while len(self.features) < self.num_hemispheres and attempts < max_attempts:
            attempts += 1
            radius = np.random.uniform(*self.hemisphere_radius_range)
            height = np.random.uniform(*self.hemisphere_height_range)
            margin = max(radius, self.edge_margin)
            if margin * 2.0 >= self.x_extent or margin * 2.0 >= self.y_extent:
                break

            x = np.random.uniform(margin, self.x_extent - margin)
            y = np.random.uniform(margin, self.y_extent - margin)
            if self._is_feature_overlapping(x, y, radius):
                continue
            self.features.append(TerrainHemisphere(x, y, radius, height))

        self._rebuild_height_grid()

    def surface_height_at(self, px, py):
        if not self.enabled:
            return 0.0
        height = self.base_height
        for feature in self.features:
            height = max(height, feature.height_at(px, py))
        return float(height)

    def collides(self, px, py, pz, clearance=0.0):
        if not self.enabled:
            return False
        return pz <= self.surface_height_at(px, py) + clearance

    def sample_height_bounds(self, px, py, clearance=0.0):
        lower = self.surface_height_at(px, py) + clearance
        return lower, None

    def _rebuild_height_grid(self):
        if not self.enabled or self.resolution is None:
            self.grid_x = None
            self.grid_y = None
            self.height_map = None
            return

        nx, ny = self.resolution
        xs = np.linspace(0.0, self.x_extent, int(nx), dtype=np.float32)
        ys = np.linspace(0.0, self.y_extent, int(ny), dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        height_map = np.full_like(grid_x, self.base_height, dtype=np.float32)

        for feature in self.features:
            radial_term = 1.0 - ((grid_x - feature.x) / max(feature.radius, 1e-8)) ** 2 - (
                (grid_y - feature.y) / max(feature.radius, 1e-8)
            ) ** 2
            bump = feature.height * np.sqrt(np.clip(radial_term, 0.0, None))
            height_map = np.maximum(height_map, bump.astype(np.float32))

        self.grid_x = grid_x
        self.grid_y = grid_y
        self.height_map = height_map


class CSVTerrainMap:
    def __init__(
        self,
        csv_path,
        enabled=False,
        resolution=1.0,
        origin=(0.0, 0.0),
        min_clearance=0.0,
        max_clearance=None,
        edge_margin=0.0,
        target_extent=None,
        target_height=None,
        z_scale=1.0,
        simplify_stride=1,
        smooth_passes=0,
    ):
        self.csv_path = csv_path
        self.enabled = bool(enabled and csv_path)
        self.resolution = float(resolution)
        self.origin = np.asarray(origin, dtype=np.float32)
        self.min_clearance = float(min_clearance)
        self.max_clearance = None if max_clearance is None else float(max_clearance)
        self.edge_margin = float(edge_margin)
        self.target_extent = (
            None
            if target_extent is None
            else (float(target_extent[0]), float(target_extent[1]))
        )
        self.target_height = None if target_height is None else float(target_height)
        self.z_scale = float(z_scale)
        self.simplify_stride = max(int(simplify_stride), 1)
        self.smooth_passes = max(int(smooth_passes), 0)

        self.base_elevation = None
        self.height_map = None
        self.grid_x = None
        self.grid_y = None
        self.grid_shape = None
        self.x_extent = 0.0
        self.y_extent = 0.0

    def clear(self):
        self.base_elevation = None
        self.height_map = None
        self.grid_x = None
        self.grid_y = None
        self.grid_shape = None
        self.x_extent = 0.0
        self.y_extent = 0.0

    def generate(self):
        self.clear()
        if not self.enabled:
            return

        elevation = load_elevation_csv(self.csv_path)
        if self.simplify_stride > 1 or self.smooth_passes > 0:
            elevation = _simplify_height_map(
                elevation,
                stride=self.simplify_stride,
                smooth_passes=self.smooth_passes,
            )
        if self.target_extent is not None:
            target_shape = (
                int(round(self.target_extent[0] / max(self.resolution, 1e-8))) + 1,
                int(round(self.target_extent[1] / max(self.resolution, 1e-8))) + 1,
            )
            elevation = _resample_height_map(elevation, target_shape)
        if self.target_height is not None:
            elevation = _scale_height_map(elevation, self.target_height)
        if abs(self.z_scale - 1.0) > 1e-12:
            elevation = (elevation.astype(np.float32) * self.z_scale).astype(np.float32)
        self.base_elevation = elevation
        self.height_map = elevation.copy()
        self.grid_shape = self.height_map.shape
        self.x_extent = max((self.grid_shape[0] - 1) * self.resolution, 0.0)
        self.y_extent = max((self.grid_shape[1] - 1) * self.resolution, 0.0)
        self._rebuild_grid_coordinates()

    def _rebuild_grid_coordinates(self):
        if self.height_map is None:
            return
        xs = self.origin[0] + np.arange(self.grid_shape[0], dtype=np.float32) * self.resolution
        ys = self.origin[1] + np.arange(self.grid_shape[1], dtype=np.float32) * self.resolution
        self.grid_x, self.grid_y = np.meshgrid(xs, ys, indexing="ij")

    def world_to_grid(self, px, py):
        if self.height_map is None:
            return None

        gx = (float(px) - float(self.origin[0])) / max(self.resolution, 1e-8)
        gy = (float(py) - float(self.origin[1])) / max(self.resolution, 1e-8)

        x_idx = int(np.floor(gx))
        y_idx = int(np.floor(gy))
        x_idx = int(np.clip(x_idx, 0, self.grid_shape[0] - 1))
        y_idx = int(np.clip(y_idx, 0, self.grid_shape[1] - 1))
        return x_idx, y_idx

    def surface_height_at(self, px, py):
        if not self.enabled or self.height_map is None:
            return 0.0
        x_idx, y_idx = self.world_to_grid(px, py)
        return float(self.height_map[x_idx, y_idx])

    def collides(self, px, py, pz, clearance=0.0):
        if not self.enabled or self.height_map is None:
            return False
        surface_height = self.surface_height_at(px, py)
        effective_min = surface_height + self.min_clearance + clearance
        if pz <= effective_min:
            return True
        return False

    def sample_height_bounds(self, px, py, clearance=0.0):
        surface_height = self.surface_height_at(px, py)
        lower = surface_height + self.min_clearance + clearance
        return lower, None
