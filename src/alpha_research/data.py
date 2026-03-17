"""Daily equity panel construction from synthetic or yfinance data sources.

The main entry point is :func:`build_dataset`, which dispatches to either a
fully deterministic synthetic generator (no network required) or a yfinance
downloader. Results are persisted in a pickle cache keyed by a hash of the
config, so repeated runs with the same config are instantaneous.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import DataConfig


DEFAULT_SECTOR_MAP = {
    "AAPL": "Technology",
    "ABBV": "Health Care",
    "ABT": "Health Care",
    "AMGN": "Health Care",
    "AMZN": "Consumer Discretionary",
    "AVGO": "Technology",
    "AXP": "Financials",
    "BA": "Industrials",
    "BAC": "Financials",
    "CAT": "Industrials",
    "COST": "Consumer Staples",
    "CRM": "Technology",
    "CSCO": "Technology",
    "CVX": "Energy",
    "DIS": "Communication Services",
    "GOOGL": "Communication Services",
    "GS": "Financials",
    "HD": "Consumer Discretionary",
    "HON": "Industrials",
    "IBM": "Technology",
    "INTC": "Technology",
    "JNJ": "Health Care",
    "JPM": "Financials",
    "KO": "Consumer Staples",
    "LIN": "Materials",
    "LLY": "Health Care",
    "LOW": "Consumer Discretionary",
    "MA": "Financials",
    "MCD": "Consumer Discretionary",
    "META": "Communication Services",
    "MMM": "Industrials",
    "MRK": "Health Care",
    "MS": "Financials",
    "MSFT": "Technology",
    "NEE": "Utilities",
    "NFLX": "Communication Services",
    "NKE": "Consumer Discretionary",
    "NVDA": "Technology",
    "ORCL": "Technology",
    "PEP": "Consumer Staples",
    "PFE": "Health Care",
    "PG": "Consumer Staples",
    "QCOM": "Technology",
    "RTX": "Industrials",
    "T": "Communication Services",
    "TMO": "Health Care",
    "TSLA": "Consumer Discretionary",
    "UNH": "Health Care",
    "UNP": "Industrials",
    "UPS": "Industrials",
    "USB": "Financials",
    "V": "Financials",
    "VZ": "Communication Services",
    "WFC": "Financials",
    "WMT": "Consumer Staples",
    "XOM": "Energy",
}


def build_dataset(config: DataConfig) -> pd.DataFrame:
    """Build or load a cached daily equity panel.

    Parameters
    ----------
    config:
        Data source settings including tickers, date range, benchmark symbol,
        and liquidity filters.

    Returns
    -------
    pd.DataFrame
        Panel with columns: ``date``, ``ticker``, ``open``, ``high``, ``low``,
        ``close``, ``adj_close``, ``volume``, ``sector``, ``is_benchmark``,
        ``dollar_volume``, ``daily_return``, ``benchmark_return``.

    Raises
    ------
    ValueError
        If ``config.source`` is not ``"synthetic"`` or ``"yfinance"``.
    """
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _cache_key(config)
    cache_path = cache_dir / f"panel_{cache_key}.pkl"
    if cache_path.exists():
        return pd.read_pickle(cache_path)

    if config.source == "synthetic":
        panel = _build_synthetic_dataset(config)
    elif config.source == "yfinance":
        panel = _build_yfinance_dataset(config)
    else:
        raise ValueError(f"Unsupported data source: {config.source}")

    panel.to_pickle(cache_path)
    return panel


def _cache_key(config: DataConfig) -> str:
    """Return a 16-character hex hash of the config dict for cache file naming."""
    payload = json.dumps(asdict(config), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _build_yfinance_dataset(config: DataConfig) -> pd.DataFrame:
    """Download OHLCV data via yfinance and normalise into the standard panel format."""
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "yfinance is required for the default market-data workflow. "
            "Install the project dependencies or use configs/synthetic.toml for offline tests."
        ) from exc

    tickers = sorted(set(config.tickers + [config.benchmark]))
    raw = yf.download(
        tickers=tickers,
        start=config.start_date,
        end=config.end_date,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("No data returned from yfinance; check the date range and network access.")

    panel = _normalize_download(raw, tickers)
    panel["sector"] = panel["ticker"].map(DEFAULT_SECTOR_MAP).fillna("Unknown")
    panel["is_benchmark"] = panel["ticker"].eq(config.benchmark)
    return _finalize_panel(panel, config)


def _normalize_download(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Reshape a multi-level yfinance DataFrame into a long-format panel."""
    frames: list[pd.DataFrame] = []
    raw = raw.sort_index()
    for ticker in tickers:
        if ticker not in raw.columns.get_level_values(0):
            continue
        frame = raw[ticker].copy()
        frame.columns = [column.lower().replace(" ", "_") for column in frame.columns]
        frame["ticker"] = ticker
        frame = frame.reset_index().rename(columns={"Date": "date", "date": "date"})
        frames.append(frame)

    if not frames:
        raise RuntimeError("Could not normalize downloaded market data.")

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.rename(columns={"adj_close": "adj_close"})
    return panel


def _build_synthetic_dataset(config: DataConfig) -> pd.DataFrame:
    """Generate a fully deterministic synthetic equity panel for offline testing.

    Returns a seeded random panel of ``config.synthetic_tickers`` stocks plus a
    benchmark, with returns drawn from a factor model: ``r = beta * r_bench +
    alpha + idio``. Suitable for CI and smoke testing without network access.
    """
    rng = np.random.default_rng(config.synthetic_seed)
    dates = pd.bdate_range(config.start_date, config.end_date)
    benchmark_returns = rng.normal(loc=0.0003, scale=0.012, size=len(dates))

    sectors = [
        "Technology",
        "Health Care",
        "Financials",
        "Industrials",
        "Consumer Discretionary",
        "Consumer Staples",
        "Energy",
        "Utilities",
        "Materials",
        "Communication Services",
    ]

    frames: list[pd.DataFrame] = []
    for idx in range(config.synthetic_tickers):
        ticker = f"STK{idx:03d}"
        sector = sectors[idx % len(sectors)]
        beta = rng.uniform(0.7, 1.4)
        alpha_signal = rng.normal(scale=0.0005, size=len(dates))
        idio = rng.normal(scale=0.018, size=len(dates))
        returns = beta * benchmark_returns + alpha_signal + idio
        close = 40.0 * np.cumprod(1.0 + returns)
        volume = rng.integers(500_000, 3_500_000, size=len(dates))

        frame = pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "open": close * (1.0 + rng.normal(0, 0.002, size=len(dates))),
                "high": close * (1.0 + np.abs(rng.normal(0, 0.006, size=len(dates)))),
                "low": close * (1.0 - np.abs(rng.normal(0, 0.006, size=len(dates)))),
                "close": close,
                "adj_close": close,
                "volume": volume,
                "sector": sector,
            }
        )
        frames.append(frame)

    benchmark = pd.DataFrame(
        {
            "date": dates,
            "ticker": config.benchmark,
            "open": 100.0 * np.cumprod(1.0 + benchmark_returns),
            "high": 100.0 * np.cumprod(1.0 + benchmark_returns + np.abs(rng.normal(0, 0.002, size=len(dates)))),
            "low": 100.0 * np.cumprod(1.0 + benchmark_returns - np.abs(rng.normal(0, 0.002, size=len(dates)))),
            "close": 100.0 * np.cumprod(1.0 + benchmark_returns),
            "adj_close": 100.0 * np.cumprod(1.0 + benchmark_returns),
            "volume": rng.integers(50_000_000, 90_000_000, size=len(dates)),
            "sector": "Benchmark",
            "is_benchmark": True,
        }
    )
    frames.append(benchmark)
    panel = pd.concat(frames, ignore_index=True)
    panel["is_benchmark"] = panel.get("is_benchmark", False)
    panel["is_benchmark"] = panel["is_benchmark"].map(lambda value: bool(value) if pd.notna(value) else False)
    return _finalize_panel(panel, config)


def _finalize_panel(panel: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Add derived columns, merge benchmark returns, and apply liquidity filters."""
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    if "is_benchmark" not in panel.columns:
        panel["is_benchmark"] = panel["ticker"].eq(config.benchmark)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["dollar_volume"] = panel["close"] * panel["volume"]
    panel["daily_return"] = panel.groupby("ticker", observed=True)["adj_close"].pct_change()

    benchmark_returns = (
        panel.loc[panel["ticker"] == config.benchmark, ["date", "daily_return"]]
        .rename(columns={"daily_return": "benchmark_return"})
        .drop_duplicates("date")
    )
    panel = panel.merge(benchmark_returns, on="date", how="left")

    equity_panel = panel.loc[~panel["is_benchmark"]].copy()
    stats = equity_panel.groupby("ticker", observed=True).agg(
        history_days=("date", "count"),
        median_close=("close", "median"),
        median_dollar_volume=("dollar_volume", "median"),
    )
    eligible = stats.loc[
        (stats["history_days"] >= config.min_history_days)
        & (stats["median_close"] >= config.min_price)
        & (stats["median_dollar_volume"] >= config.min_median_dollar_volume)
    ].index
    equity_panel = equity_panel.loc[equity_panel["ticker"].isin(eligible)].copy()
    panel = pd.concat([equity_panel, panel.loc[panel["is_benchmark"]]], ignore_index=True)
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel
