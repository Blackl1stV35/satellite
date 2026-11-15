"""Quarterly revenue and net profit from filings on the Thai SEC disclosure portal (primary documents).

Listing pages and downloads are read-only GETs through satellite.net (one request per 2 s per host).
Statements are zip or pdf files; text is extracted with pdfplumber and the income statement is parsed.
"""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
from pathlib import Path

from . import net

BASE = "https://market.sec.or.th/public/idisc/Download?FILEID="


def load_listing(path: str | Path) -> dict[str, list[dict]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pick_files(row: dict) -> list[str]:
    """Files of one filing: prefer the pdf, always keep the zip as a fallback."""
    pdfs = [f for f in row["files"] if f.lower().endswith(".pdf")]
    zips = [f for f in row["files"] if f.lower().endswith(".zip")]
    return pdfs + zips


def download(fileid: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = fileid.rsplit("/", 1)[-1]
    out = dest_dir / name
    if out.exists() and out.stat().st_size > 0:
        return out
    r = net.get(BASE + fileid, timeout=120)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out


def filing_timestamp(fileid: str) -> str:
    """Filing date from names like 28FIN130620251711110934E.zip (DDMMYYYYHHMM...); else the folder month."""
    name = fileid.rsplit("/", 1)[-1]
    m = re.search(r"FIN(\d{2})(\d{2})(\d{4})(\d{2})(\d{2})", name)
    if m:
        d, mo, y, hh, mm = m.groups()
        return f"{y}-{mo}-{d}"
    m = re.search(r"news/(\d{4})(\d{2})/", fileid)
    return f"{m.group(1)}-{m.group(2)}" if m else ""


def unzip_all(path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    res = []
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as z:
        for n in z.namelist():
            if n.endswith("/"):
                continue
            safe = Path(n).name
            (out_dir / safe).write_bytes(z.read(n))
            res.append(out_dir / safe)
    return res


# ----- income statement parsing from FINANCIAL_STATEMENTS.XLS / XLSX inside each zip -----

DURATION = [("THREE-MONTH", 3), ("THREE MONTH", 3), ("SIX-MONTH", 6), ("SIX MONTH", 6), ("NINE-MONTH", 9),
            ("NINE MONTH", 9), ("TWELVE-MONTH", 12), ("YEAR ENDED", 12), ("YEAR  ENDED", 12)]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def sheets_from_zip(zpath: Path) -> list[tuple[str, list[list]]]:
    """All sheets of the financial statements workbook inside a filing zip as (name, rows)."""
    with zipfile.ZipFile(io.BytesIO(Path(zpath).read_bytes())) as z:
        name = next(n for n in z.namelist() if n.upper().startswith("FINANCIAL_STATEMENTS"))
        data = z.read(name)
    out = []
    if name.lower().endswith(".xlsx"):
        import openpyxl

        wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True, read_only=True)
        for ws in wb.worksheets:
            out.append((ws.title, [["" if c is None else c for c in row] for row in ws.iter_rows(values_only=True)]))
    else:
        import xlrd

        wb = xlrd.open_workbook(file_contents=data)
        for s in wb.sheets():
            out.append((s.name, [s.row_values(r) for r in range(s.nrows)]))
    return out


def _unit_to_million(rows: list[list]) -> float:
    for r in rows[:14]:
        t = _norm(" ".join(str(c) for c in r if isinstance(c, str))).upper()
        if "UNIT" in t or "IN THOUSAND" in t or "IN MILLION" in t or "BAHT" in t:
            if "THOUSAND" in t or "000" in t:
                return 1e-3
            if "MILLION" in t:
                return 1.0
            if "BAHT" in t:
                return 1e-6
    return math.nan


def _is_pl_sheet(rows: list[list]) -> tuple[bool, int | None]:
    head = _norm(" ".join(str(c) for r in rows[:8] for c in r if isinstance(c, str))).upper()
    if not ("PROFIT OR LOSS" in head or "INCOME" in head or "PROFIT (LOSS)" in head):
        return False, None
    if "FINANCIAL POSITION" in head or "CASH FLOW" in head or "CHANGES IN" in head:
        return False, None
    months = None
    m = re.search(r"1 ([A-Z]+) (\d{4}) TO (\d{1,2}) ([A-Z]+) (\d{4})", head)
    if m:
        mon = {k: i + 1 for i, k in enumerate(["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY",
                                               "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"])}
        if m.group(1) in mon and m.group(4) in mon:
            months = (int(m.group(5)) - int(m.group(2))) * 12 + mon[m.group(4)] - mon[m.group(1)] + 1
    if months is None:
        months = next((mm for k, mm in DURATION if k in head), None)
    return True, months


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _year_col(rows: list[list]) -> tuple[int, int] | None:
    """(row index, column) of the first current-year heading; years may be numbers or text."""
    for i, r in enumerate(rows[:20]):
        cols = []
        for j, c in enumerate(r):
            s = str(c).strip()
            if re.fullmatch(r"((\d{1,2} )?[A-Za-z]+ )?(19|20|25)\d\d(\.0)?", s):
                cols.append(j)
        if len(cols) >= 2:
            return i, cols[0]
    return None


def _label(row: list, col: int) -> str:
    for c in row[:col]:
        if isinstance(c, str) and c.strip():
            return _norm(c)
    return ""


OWNERS = re.compile(r"owners? of (the )?(parent|company)|equity holders of (the )?(parent|company)"
                    r"|shareholders of the (parent|company)|the parent company")
NET_LINE = re.compile(r"^(net )?(profit|loss)( \(loss\))?( \(profit\))? for the (year|period|quarter)")
STOP = re.compile(r"^(cost|expenses?|gross|operating cost|costs|other income|other revenue)")


def parse_income(rows: list[list]) -> dict:
    """Consolidated current-period revenue and owners' profit in million baht from one income sheet.

    Revenue is the first subtotal or total inside the revenue block (rows before costs or expenses),
    which excludes other income where a sales subtotal exists; a lone revenue line is used as is.
    """
    yc = _year_col(rows)
    unit = _unit_to_million(rows)
    if yc is None or math.isnan(unit):
        return {}
    start, col = yc
    block = []
    prof = None
    in_block = True
    per_share = False
    for r in rows[start + 1 :]:
        lab = _label(r, col)
        low = lab.lower()
        v = _num(r[col]) if col < len(r) else None
        if in_block and STOP.match(low):
            in_block = False
        if in_block and v is not None:
            block.append((low, v))
        if "per share" in low:
            per_share = True
        if prof is None and not per_share and v is not None and OWNERS.search(low):
            prof = v
    rev = None
    for low, v in block[1:]:
        if low.startswith("total") or low == "":
            rev = v
            break
    if rev is None and block:
        rev = block[0][1]
    if prof is None:
        for r in rows[start + 1 :]:
            low = _label(r, col).lower()
            v = _num(r[col]) if col < len(r) else None
            if v is not None and NET_LINE.match(low):
                prof = v
                break
    out = {}
    if rev is not None:
        out["revenue_mthb"] = rev * unit
    if prof is not None:
        out["net_profit_mthb"] = prof * unit
    return out


def parse_filing(zpath: Path) -> list[dict]:
    """One dict per income sheet found: months covered by the sheet and the parsed numbers."""
    res = []
    seen = set()
    for name, rows in sheets_from_zip(zpath):
        ok, months = _is_pl_sheet(rows)
        if not ok:
            continue
        p = parse_income(rows)
        if "revenue_mthb" not in p or "net_profit_mthb" not in p:
            continue
        key = months
        if key in seen:
            continue
        seen.add(key)
        res.append({"sheet": name, "months": months, **p})
    return res


def _period(asof: str):
    import pandas as pd

    d, m, y = asof.split("/")
    return pd.Period(f"{y}-{m}", freq="M")


def assemble(listing: dict, filings_dir: str | Path, tol: float = 0.005):
    """Quarterly revenue and owners' profit per ticker from primary filings.

    Returns (rows, log). A 3-month sheet is used directly; otherwise a quarter is the cumulative figure
    minus the earlier quarters of the same fiscal year. A quarter that cannot be built is dropped and logged.
    """
    filings_dir = Path(filings_dir)
    rows, log = [], []
    for tk, items in listing.items():
        cum: dict = {}
        fdate: dict = {}
        for r in items:
            if r["type"] != "Consolidated":
                continue
            zs = [f for f in pick_files(r) if f.lower().endswith(".zip")]
            if not zs:
                log.append((tk, r["asof"], "no zip in listing"))
                continue
            path = filings_dir / tk / zs[0].rsplit("/", 1)[-1]
            pe = _period(r["asof"])
            try:
                res = parse_filing(path)
            except Exception as exc:
                log.append((tk, r["asof"], f"parse error {type(exc).__name__}"))
                continue
            for x in res:
                if x["months"] in (3, 6, 9, 12):
                    cum[(pe, x["months"])] = (x["revenue_mthb"], x["net_profit_mthb"])
            if not res:
                log.append((tk, r["asof"], "no income sheet parsed"))
            fdate[pe] = filing_timestamp(zs[0])
        quarters: dict = {}
        for pe in sorted({k[0] for k in cum}):
            if (pe, 3) in cum:
                quarters[pe] = (cum[(pe, 3)], "direct 3-month sheet")
            else:
                for m in (6, 9, 12):
                    if (pe, m) in cum:
                        prev = [pe - 3 * i for i in range(1, m // 3)]
                        if all(p in quarters for p in prev):
                            rv = cum[(pe, m)][0] - sum(quarters[p][0][0] for p in prev)
                            pf = cum[(pe, m)][1] - sum(quarters[p][0][1] for p in prev)
                            quarters[pe] = ((rv, pf), f"{m}-month cumulative minus earlier quarters")
                        else:
                            log.append((tk, str(pe), f"cumulative {m}M present but an earlier quarter is missing"))
                        break
            # consistency: cumulative vs sum of direct quarters, where both exist
            for m in (6, 9, 12):
                if (pe, m) in cum:
                    prev = [pe - 3 * i for i in range(0, m // 3)]
                    if all(p in quarters for p in prev):
                        s_rev = sum(quarters[p][0][0] for p in prev)
                        if abs(s_rev - cum[(pe, m)][0]) > tol * abs(cum[(pe, m)][0]):
                            log.append((tk, str(pe), f"{m}M revenue differs from sum of quarters by "
                                        f"{s_rev - cum[(pe, m)][0]:.1f} mTHB (restatement or parse issue)"))
        for pe, ((rv, pf), how) in sorted(quarters.items()):
            rows.append({"ticker": tk, "period_end": str(pe), "revenue_mthb": round(rv, 3),
                         "net_profit_mthb": round(pf, 3), "filing_date": fdate.get(pe, ""), "method": how})
    return rows, log


def write_outputs(listing_path: str | Path, filings_dir: str | Path, out_dir: str | Path, aggregator_csv: str | Path):
    """Write quarterly_primary.csv and fundamentals_log.csv; cross-check against the pilot aggregator file."""
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, log = assemble(load_listing(listing_path), filings_dir)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "quarterly_primary.csv", index=False)
    agg = pd.read_csv(aggregator_csv)
    for tk in df["ticker"].unique():
        a, d = agg[agg["ticker"] == tk], df[df["ticker"] == tk]
        for _, r in a.iterrows():
            hit = d[(d["revenue_mthb"] - r["revenue_mthb"]).abs() <= 0.003 * abs(r["revenue_mthb"])]
            if hit.empty:
                log.append((tk, f"aggregator FY{r['fiscal']} Q{r['quarter']}",
                            f"aggregator revenue {r['revenue_mthb']} has no match in the primary quarters; "
                            "primary figure kept"))
    pd.DataFrame(log, columns=["ticker", "item", "note"]).to_csv(out_dir / "fundamentals_log.csv", index=False)
    return df, log
