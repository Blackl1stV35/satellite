"""Command line entry: python -m satellite.cli monthly | report."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import DATA_ROOT
from .aoi import load_aois
from .series import build_monthly

PROJECT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["monthly", "report", "radar", "fundamentals", "extension", "port"])
    ap.add_argument("--areas", default=str(PROJECT / "data" / "areas.json"))
    ap.add_argument("--start", default="2019-01")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--deadline", type=int, default=0, help="hard exit after N seconds (resumable)")
    ap.add_argument("--throttle", action="store_true", help="2 s per host between requests (range reads)")
    ap.add_argument("--series-only", action="store_true", help="port: skip downloads, rebuild series only")
    ap.add_argument("--only", default="", help="substring filter on area name")
    args = ap.parse_args()
    if args.deadline:
        import os
        import threading

        timer = threading.Timer(args.deadline, lambda: os._exit(3))
        timer.daemon = True
        timer.start()
    aois = load_aois(args.areas)
    if args.only:
        aois = [a for a in aois if args.only.lower() in a.name.lower()]
    if args.command == "fundamentals":
        from .fundamentals import write_outputs

        ext = f"{DATA_ROOT}/extension"
        df, log = write_outputs(f"{ext}/filings/listing_yearly.json", f"{ext}/filings", f"{ext}/fundamentals",
                                f"{DATA_ROOT}/pilot/fundamentals/finnomena_quarterly.csv")
        print(df.groupby("ticker").size(), len(log), "log lines")
    elif args.command == "extension":
        from .extension import main as ext_main

        res = ext_main(args.areas)
        print(res[["proxy", "target", "lag_q", "n", "rho", "perm_p", "holm_p", "placebo_p", "loyo_stable", "survives"]]
              .round(3).to_string())
    elif args.command == "port":
        from . import ships
        from .extension import EXT, chart_port

        cache = f"{DATA_ROOT}/extension/cache"
        if not args.series_only:
            ships.run_fetch(args.start, args.end, cache)
        df = ships.build_series(cache)
        (EXT / "series").mkdir(parents=True, exist_ok=True)
        df.to_csv(EXT / "series" / "P1_port_vessels_monthly.csv", index=False)
        chart_port(df, EXT / "charts" / "step3_port_vessels.png")
        print(df.describe())
    elif args.command == "radar":
        from . import radar

        radar.run_all(radar.default_groups(aois), args.start, args.end, f"{DATA_ROOT}/extension/cache")
    elif args.command == "monthly":
        if args.throttle:
            from . import net
            from . import sentinel as sn

            sn.OPENER = net.open_range
        df = build_monthly(aois, args.start, args.end, f"{DATA_ROOT}/cache", args.workers)
        print(df.groupby(["area", "status"]).size())
    else:
        from .report import run_pilot

        run_pilot(aois)


if __name__ == "__main__":
    main()
