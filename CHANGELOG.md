# Changelog

All notable changes to this project will be documented here.

## [0.1.0] - 2026-04-04

### Added
- Initial release of the cross-sectional equity alpha research platform.
- `data.py`: synthetic and yfinance panel builders with pickle caching.
- `features.py`: momentum, reversal, volatility, volume, and beta cross-sectional features with leakage-safe forward-return targets.
- `modeling.py`: walk-forward ridge regression and rank-composite signal with strict out-of-sample splits.
- `portfolio.py`: sector-neutral, beta-neutral long/short portfolio construction via least-squares constraint projection.
- `backtest.py`: daily P&L attribution, turnover, drawdown, and volatility-regime metrics.
- `reporting.py`: equity curve and leg contribution plots, research summary report generator.
- `cli.py`: `alpha-research run --config <path>` entry point.
- `configs/synthetic.toml`: offline smoke test config.
- `configs/default.toml`: live market-data config using yfinance.
- Tests for forward-target accuracy, cross-sectional z-score centering, walk-forward split boundaries, portfolio neutrality, and transaction cost mechanics.
