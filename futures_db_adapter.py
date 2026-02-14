"""
futures_db_adapter.py — Bridge between futuresdatabase and MarkovIBViz.

Reads 1-minute OHLCV bars and day-level annotations from a futuresdatabase
SQLite database for a specified date range, then outputs CSV files that
MarkovIBViz's load_data.py can consume directly.

Outputs (saved to MarkovIBViz/output/):
  - futures_db_bars_{start}_{end}.csv
        Columns: datetime, open, high, low, close, volume
        Timestamps in Pacific Time, 1-minute resolution.
        Compatible with load_data.load_and_validate(filepath=...).

  - futures_db_annotations_{start}_{end}.csv
        Columns: session_date, annotation_id, annotation_type, content,
                 tags, source, status, created_at
        Tags are JSON arrays (e.g. '["breakout", "trend_day"]') for
        downstream labeling scripts to parse.

Usage:
    # Export a date range from an existing database
    python futures_db_adapter.py --db-path /path/to/market_data.db \
        --start-date 2025-06-01 --end-date 2025-06-30

    # Specify symbol and source
    python futures_db_adapter.py --db-path /path/to/market_data.db \
        --start-date 2025-06-01 --end-date 2025-06-30 \
        --symbol MNQ --source tradingview

    # Include halt-period bars (2-3 PM PT)
    python futures_db_adapter.py --db-path /path/to/market_data.db \
        --start-date 2025-06-01 --end-date 2025-06-30 --include-halt

    # Initialize a new database from CSV before exporting
    python futures_db_adapter.py --init-csv /path/to/data.csv \
        --db-path /path/to/market_data.db \
        --start-date 2025-06-01 --end-date 2025-06-30

Programmatic usage:
    from futures_db_adapter import export_bars, export_annotations

    bars_path = export_bars(
        db_path="market_data.db",
        symbol="MNQ",
        start_date="2025-06-01",
        end_date="2025-06-30",
    )

    ann_path = export_annotations(
        db_path="market_data.db",
        symbol="MNQ",
        start_date="2025-06-01",
        end_date="2025-06-30",
    )
"""

import argparse
import csv
import datetime
import json
import os
import sys
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Resolve paths so we can import from the sibling futuresdatabase repo
# regardless of the working directory.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FUTURES_DB_DIR = os.path.join(os.path.dirname(_THIS_DIR), "futuresdatabase")
if _FUTURES_DB_DIR not in sys.path:
    sys.path.insert(0, _FUTURES_DB_DIR)

from market_archivist import (
    get_bars,
    get_day_annotations,
    init_database,
    ingest_csv,
)

PT_TIMEZONE = ZoneInfo("America/Los_Angeles")
OUTPUT_DIR = os.path.join(_THIS_DIR, "output")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _epoch_to_pt_str(timestamp: int) -> str:
    """Convert Unix epoch seconds to a Pacific Time datetime string.

    Returns format ``YYYY-MM-DD HH:MM:SS`` which ``pd.to_datetime(...,
    format='mixed')`` in load_data.py parses correctly.
    """
    dt = datetime.datetime.fromtimestamp(timestamp, tz=PT_TIMEZONE)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _bars_filename(start_date: str, end_date: str) -> str:
    return f"futures_db_bars_{start_date}_{end_date}.csv"


def _annotations_filename(start_date: str, end_date: str) -> str:
    return f"futures_db_annotations_{start_date}_{end_date}.csv"


# ---------------------------------------------------------------------------
# Core export functions
# ---------------------------------------------------------------------------

def export_bars(
    db_path: str,
    symbol: str,
    start_date: str,
    end_date: str,
    source: str = "tradingview",
    include_halt: bool = False,
) -> str:
    """Export bars from futuresdatabase to a CSV consumable by MarkovIBViz.

    Args:
        db_path: Path to the SQLite database file.
        symbol: Instrument symbol (e.g. ``"MNQ"``).
        start_date: First session date to include (``YYYY-MM-DD``).
        end_date: Last session date to include (``YYYY-MM-DD``).
        source: Data source filter (default ``"tradingview"``).
        include_halt: If True, include halt-period bars (2-3 PM PT).

    Returns:
        Absolute path to the written CSV file.

    Raises:
        FileNotFoundError: If *db_path* does not exist.
        RuntimeError: If the query returns zero bars.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    bars = get_bars(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        source=source,
        include_halt=include_halt,
        db_path=db_path,
    )

    if not bars:
        raise RuntimeError(
            f"No bars found for {symbol} between {start_date} and {end_date} "
            f"(source={source}, db={db_path})"
        )

    _ensure_output_dir()
    filename = _bars_filename(start_date, end_date)
    out_path = os.path.join(OUTPUT_DIR, filename)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime", "open", "high", "low", "close", "volume"])
        for bar in bars:
            writer.writerow([
                _epoch_to_pt_str(bar["timestamp"]),
                bar["open"],
                bar["high"],
                bar["low"],
                bar["close"],
                bar["volume"],
            ])

    print(f"Exported {len(bars)} bars -> {out_path}")
    return out_path


def export_annotations(
    db_path: str,
    symbol: str,
    start_date: str,
    end_date: str,
) -> str | None:
    """Export day-level annotations from futuresdatabase to a CSV.

    The output preserves the ``tags`` column as a JSON array string so that
    downstream labeling scripts can parse individual tags (e.g.
    ``json.loads(row['tags'])``).

    Args:
        db_path: Path to the SQLite database file.
        symbol: Instrument symbol.
        start_date: First session date (``YYYY-MM-DD``).
        end_date: Last session date (``YYYY-MM-DD``).

    Returns:
        Absolute path to the written CSV, or ``None`` if no annotations exist
        for the date range.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")

    annotations = get_day_annotations(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        db_path=db_path,
    )

    if not annotations:
        print(f"No annotations found for {symbol} between {start_date} and {end_date}")
        return None

    _ensure_output_dir()
    filename = _annotations_filename(start_date, end_date)
    out_path = os.path.join(OUTPUT_DIR, filename)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "session_date",
            "annotation_id",
            "annotation_type",
            "content",
            "tags",
            "source",
            "status",
            "created_at",
        ])
        for ann in annotations:
            writer.writerow([
                ann["session_date"],
                ann["id"],
                ann["annotation_type"],
                ann["content"],
                json.dumps(ann["tags"]),
                ann["source"],
                ann["status"],
                ann["created_at"],
            ])

    print(f"Exported {len(annotations)} annotations -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Export a date range from a futuresdatabase SQLite DB into CSV "
            "files that MarkovIBViz can read for visualization."
        ),
    )
    p.add_argument(
        "--db-path",
        required=True,
        help="Path to the futuresdatabase SQLite file.",
    )
    p.add_argument(
        "--start-date",
        required=True,
        help="First session date to include (YYYY-MM-DD).",
    )
    p.add_argument(
        "--end-date",
        required=True,
        help="Last session date to include (YYYY-MM-DD).",
    )
    p.add_argument(
        "--symbol",
        default="MNQ",
        help="Instrument symbol (default: MNQ).",
    )
    p.add_argument(
        "--source",
        default="tradingview",
        help="Data source filter (default: tradingview).",
    )
    p.add_argument(
        "--include-halt",
        action="store_true",
        help="Include halt-period bars (2-3 PM PT).",
    )
    p.add_argument(
        "--init-csv",
        default=None,
        help=(
            "Path to a CSV file to ingest into the database before exporting. "
            "The database is created if it does not already exist."
        ),
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Optionally initialize the database from a CSV first.
    if args.init_csv:
        if not os.path.exists(args.init_csv):
            parser.error(f"CSV file not found: {args.init_csv}")
        print(f"Initializing database at {args.db_path} ...")
        init_database(args.db_path)
        print(f"Ingesting {args.init_csv} ...")
        result = ingest_csv(
            file_path=args.init_csv,
            symbol=args.symbol,
            timeframe="1m",
            source=args.source,
            db_path=args.db_path,
        )
        print(
            f"  inserted={result['inserted']}  "
            f"skipped={result['skipped']}  "
            f"conflicts={result['conflicts']}"
        )

    # Export bars.
    bars_path = export_bars(
        db_path=args.db_path,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        source=args.source,
        include_halt=args.include_halt,
    )

    # Export annotations (best-effort; may be empty).
    ann_path = export_annotations(
        db_path=args.db_path,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Print summary for integration with the pipeline.
    print()
    print("To run the MarkovIBViz pipeline on the exported data:")
    print(f'  from load_data import load_and_validate, save_cleaned')
    print(f'  df = load_and_validate(filepath="{bars_path}")')
    print(f'  save_cleaned(df)')
    if ann_path:
        print()
        print("Annotations are available for labeling at:")
        print(f"  {ann_path}")
        print("  Tags column contains JSON arrays, parse with json.loads().")


if __name__ == "__main__":
    main()
