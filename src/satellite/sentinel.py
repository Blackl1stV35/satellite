"""Sentinel-2 L2A access through the Earth Search STAC: search, cloud-aware selection, reads."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import transform, transform_bounds, transform_geom
from rasterio.windows import Window, from_bounds

from .aoi import AOI

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
# SCL classes: 3 cloud shadow, 8 cloud medium, 9 cloud high, 10 thin cirrus.
CLOUD_CLASSES = (3, 8, 9, 10)
# Invalid for analysis: no data (0), saturated or defective (1) and the cloud classes.
INVALID_CLASSES = (0, 1, *CLOUD_CLASSES)
WATER_CLASS = 6
SCALE = 0.0001
DEFAULT_OFFSET = -0.1
# When set (see net.open_range), all COG reads and STAC searches go through the 2 s per host throttle.
OPENER = None


def scale_reflectance(dn: np.ndarray, offset: float = DEFAULT_OFFSET) -> np.ndarray:
    """Digital number to surface reflectance: DN * 0.0001 + offset. DN 0 is no data (NaN)."""
    dn = np.asarray(dn)
    out = dn.astype("float32") * np.float32(SCALE) + np.float32(offset)
    out[dn == 0] = np.nan
    return out


def invalid_mask(scl: np.ndarray) -> np.ndarray:
    """True where a pixel is no data, saturated, cloud, shadow or cirrus."""
    return np.isin(scl, INVALID_CLASSES)


def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (nir - red) / (nir + red)


def mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Modified normalised difference water index (green, SWIR1). Water is above 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (green - swir) / (green + swir)


def asset_offset(item) -> float:
    """Reflectance offset to add after DN * 0.0001.

    Checked against water pixels: Earth Search items flagged boa_offset_applied=True hold DN
    with the +1000 offset already removed, although their raster:bands metadata still lists
    -0.1. Items with the flag False keep the ESA +1000 for processing baseline 04.00 or later.
    """
    props = item.properties
    if props.get("earthsearch:boa_offset_applied") is True:
        return 0.0
    baseline = str(props.get("s2:processing_baseline", "04.00"))
    try:
        return DEFAULT_OFFSET if float(baseline) >= 4.0 else 0.0
    except ValueError:
        return DEFAULT_OFFSET


@dataclass
class Grid:
    transform: rasterio.Affine
    shape: tuple[int, int]
    crs: object


def aoi_mask(aoi: AOI, grid: Grid) -> np.ndarray:
    """Boolean mask of the AOI on a grid (grid CRS is projected, metres)."""
    h, w = grid.shape
    if aoi.polygon:
        geom = {"type": "Polygon", "coordinates": [[list(p) for p in aoi.polygon]]}
        g = transform_geom("EPSG:4326", grid.crs, geom)
        return rasterize(
            [(g, 1)], out_shape=(h, w), transform=grid.transform, fill=0, dtype="uint8"
        ).astype(bool)
    cx, cy = transform("EPSG:4326", grid.crs, [aoi.lon], [aoi.lat])
    rows, cols = np.indices((h, w))
    xs, ys = rasterio.transform.xy(grid.transform, rows, cols, offset="center")
    xs = np.asarray(xs).reshape(h, w)
    ys = np.asarray(ys).reshape(h, w)
    return (xs - cx[0]) ** 2 + (ys - cy[0]) ** 2 <= float(aoi.radius_m) ** 2


def read_window(
    href: str,
    bbox_lonlat: tuple[float, float, float, float],
    res_m: float,
    resampling: Resampling = Resampling.average,
    out_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, Grid]:
    """Read the bbox from a COG, resampled to about res_m metres per pixel."""
    with rasterio.open(href, opener=OPENER) if OPENER else rasterio.open(href) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox_lonlat)
        win = from_bounds(left, bottom, right, top, transform=src.transform)
        win = win.round_offsets().round_lengths()
        win = win.intersection(Window(0, 0, src.width, src.height))
        if out_shape is None:
            px = abs(src.transform.a)
            out_shape = (
                max(1, round(win.height * px / res_m)),
                max(1, round(win.width * px / res_m)),
            )
        arr = src.read(1, window=win, out_shape=out_shape, resampling=resampling)
        tr = src.window_transform(win) * rasterio.Affine.scale(
            win.width / out_shape[1], win.height / out_shape[0]
        )
        return arr, Grid(tr, out_shape, src.crs)


def search_month(client, bbox, year: int, month: int, limit: int = 60):
    start = f"{year:04d}-{month:02d}-01"
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    end = f"{ny:04d}-{nm:02d}-01"
    dt = f"{start}T00:00:00Z/{end}T00:00:00Z"
    if OPENER is not None:  # throttled path: one POST, no pystac-client paging
        from pystac import Item

        from . import net

        body = {"collections": [COLLECTION], "bbox": list(bbox), "datetime": dt, "limit": limit}
        feats = net.post_json(f"{STAC_URL}/search", body).json().get("features", [])
        items = [Item.from_dict(f) for f in feats]
    else:
        items = list(client.search(collections=[COLLECTION], bbox=list(bbox), datetime=dt, max_items=limit).items())
    items = [i for i in items if i.datetime and i.datetime.strftime("%Y-%m") == start[:7]]
    return sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 100.0))


def covers_bbox(item, bbox, margin: float = 0.0) -> bool:
    ib = item.bbox
    return (
        ib[0] <= bbox[0] - margin
        and ib[1] <= bbox[1] - margin
        and ib[2] >= bbox[2] + margin
        and ib[3] >= bbox[3] + margin
    )


def overlaps_bbox(item, bbox) -> bool:
    ib = item.bbox
    return not (ib[2] < bbox[0] or ib[0] > bbox[2] or ib[3] < bbox[1] or ib[1] > bbox[3])


def scl_clear_fraction(scl: np.ndarray, mask: np.ndarray) -> float:
    """Share of AOI pixels that are usable (not nodata, cloud, shadow or cirrus)."""
    n = int(mask.sum())
    if n == 0:
        return 0.0
    return float((~invalid_mask(scl) & mask).sum()) / n


def aoi_grid_for(aoi: AOI, item):
    """Read SCL for an item and return (scl, grid, mask)."""
    arr, grid = read_window(
        item.assets["scl"].href, aoi.bbox_lonlat(), aoi.resolution_m, Resampling.nearest
    )
    return arr, grid, aoi_mask(aoi, grid)


def select_best_scene(aoi: AOI, items, max_candidates: int = 4, min_clear: float = 0.5):
    """Pick the item with the largest clear share over the AOI (by SCL).

    Returns (item or None, clear fraction, candidates tried).
    """
    best = None
    bbox = aoi.bbox_lonlat()
    tried = 0
    seen = set()
    for it in items:
        if not overlaps_bbox(it, bbox):
            continue
        key = (it.datetime.date(), it.properties.get("s2:mgrs_tile"))
        if key in seen:
            continue
        seen.add(key)
        if tried >= max_candidates:
            break
        tried += 1
        try:
            scl, _, mask = aoi_grid_for(aoi, it)
        except Exception:
            continue
        cf = scl_clear_fraction(scl, mask)
        if best is None or cf > best[0]:
            best = (cf, it)
        if cf >= 0.98:
            break
    if best is None or best[0] < min_clear:
        return None, (best[0] if best else 0.0), tried
    return best[1], best[0], tried


# With the ESA +1000 offset in DN, no valid pixel can sit below DN 1000. Without it, dark pixels
# (water, dense canopy, shadow) sit well under 900, so the 1st percentile separates the cases.
OFFSET_CHECK_DN = 900


def verify_offset(claimed: float, red_dn: np.ndarray, valid: np.ndarray) -> float:
    """Keep a claimed -0.1 offset only if the data show the +1000 is really in the DN.

    Earth Search flags were found unreliable for some baseline 05 scenes (for example February
    2025 and October 2022), which gave NDVI above 1 when -0.1 was applied.
    """
    if claimed == 0.0:
        return 0.0
    vals = red_dn[valid & (red_dn > 0)]
    if vals.size == 0:
        return claimed
    return claimed if float(np.percentile(vals, 1)) >= OFFSET_CHECK_DN else 0.0


def read_bands(aoi: AOI, item):
    """Read red, green, nir, swir16 and scl on one grid. Returns bands, grid, mask, offset."""
    scl, grid, mask = aoi_grid_for(aoi, item)
    raw = {}
    for name in ("red", "green", "nir", "swir16"):
        raw[name], _ = read_window(
            item.assets[name].href,
            aoi.bbox_lonlat(),
            aoi.resolution_m,
            Resampling.average,
            out_shape=grid.shape,
        )
    off = verify_offset(asset_offset(item), raw["red"], mask & ~invalid_mask(scl))
    out = {"scl": scl}
    for name, dn in raw.items():
        out[name] = scale_reflectance(dn, off)
    return out, grid, mask, off


def compute_metrics(bands: dict, mask: np.ndarray, pixel_area_km2: float) -> dict:
    """Area-level numbers from one scene on its grid."""
    scl = bands["scl"]
    ok = mask & ~invalid_mask(scl)
    n_aoi = int(mask.sum())
    n_ok = int(ok.sum())
    res = {"n_aoi": n_aoi, "n_clear": n_ok, "clear_frac": n_ok / n_aoi if n_aoi else math.nan}
    if n_ok == 0:
        return res | {"ndvi_mean": math.nan, "water_frac": math.nan, "water_km2": math.nan, "ndbi_mean": math.nan}
    nd = ndvi(bands["red"], bands["nir"])
    mw = mndwi(bands["green"], bands["swir16"])
    nd[~np.isfinite(nd)] = np.nan
    mw[~np.isfinite(mw)] = np.nan
    water = (mw > 0.0) & ok
    land = ok & ~water & (scl != WATER_CLASS)
    res["ndvi_mean"] = float(np.nanmean(nd[land])) if land.any() else math.nan
    res["water_frac"] = float(water.sum()) / n_ok
    res["water_km2"] = res["water_frac"] * n_aoi * pixel_area_km2
    # Normalised difference built-up index (SWIR1, NIR): higher for bare ground and built surfaces.
    sw, nr = bands["swir16"], bands["nir"]
    with np.errstate(divide="ignore", invalid="ignore"):
        nb = (sw - nr) / (sw + nr)
    nb[~np.isfinite(nb)] = np.nan
    res["ndbi_mean"] = float(np.nanmean(nb[land])) if land.any() else math.nan
    return res
