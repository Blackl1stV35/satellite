# satellite

Descriptive research on free satellite proxies versus Thai listed company quarterly results. Not investment advice.
Outputs live under `E:\satellite-data` only. Definitions and the test battery are fixed before testing
(`E:\satellite-data\extension\definitions.md`).

## Commands (run from this folder, package installed with `pip install -e .`)

- `python -m satellite.cli monthly --start 2017-01 --end 2018-12 --workers 1 --throttle` Sentinel-2 monthly NDVI and water (pilot cache).
- `python -m satellite.cli radar --start 2017-01 --end 2026-09` Sentinel-1 RTC monthly backscatter and water (Planetary Computer).
- `python -m satellite.cli fundamentals` quarterly revenue and owners' profit from SEC portal filings (zip workbooks).
- `python -m satellite.cli port --start 2017-01 --end 2026-09` Laem Chabang vessel counts (use `--series-only` to rebuild).
- `python -m satellite.cli extension` run every proxy-target-lag test and write tables and charts.
- `python -m pytest` and `ruff check .` (31 tests).

## Per-run notes

- Pilot (2026-11-10): Sentinel-2 monthly NDVI and reservoir water, three companies, 2019 onward, no usable signal.
- Extension (2026-11-11): added `net.py` (one request per 2 s per host, range-read file for rasterio, RAM gate),
  `radar.py`, `ships.py`, `fundamentals.py`, `battery.py`, `extension.py`. Sentinel-1 comes from the Planetary Computer
  `sentinel-1-rtc` collection through an anonymously issued token (Earth Search `sentinel-1-grd` is requester-pays S3).
  Sentinel-2 history extended back to 2017-01 (first usable scenes are in the series files).
- Results come from primary SEC filings. The pilot's aggregator file had wrong KTIS quarters (KTIS year ends 30 September,
  not December); the extension uses the filings only and logs differences in `fundamentals/fundamentals_log.csv`.
- Free RAM on the machine stayed under the 1.5 GB gate during the run, so only single-window reads (a few MB each) were
  made, in one process, with a low RAM floor (`SAT_RAM_FLOOR_GB`); no full-scene raster work was done.

## Status

Open-source toolkit. The complete, supported part is the Sentinel-2 monthly site series (NDVI and water share) for the areas in data/areas.json. Radar, vessel counts and filing links are experimental and unsupported. Not investment advice.
