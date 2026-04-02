# Research Summary

## Objective

This project evaluates whether cross-sectional signals derived from daily U.S. equity data can predict relative performance over the next few days and support a market-neutral long/short portfolio.

## What Was Tested

- Signal family: reversal, momentum, volatility, volume, rolling beta, beta instability, idiosyncratic volatility, and extreme-move mean reversion
- Prediction models: rank-composite baseline and ridge regression
- Portfolio: long top decile, short bottom decile, sector-aware construction, beta-neutral scaling, transaction costs applied

## Out-of-Sample Results

- Annualized return: 9.41%
- Annualized volatility: 13.70%
- Sharpe ratio: 0.69
- Max drawdown: -12.60%
- Hit rate: 50.38%
- Average turnover: 0.34

## Feature Diagnostics

Top absolute correlations between engineered features and realized returns in the evaluation window:

| feature                         |   abs_correlation |
|:--------------------------------|------------------:|
| momentum_60d_rank               |        0.0156981  |
| beta_instability_20d_z          |        0.00680553 |
| idio_vol_60d_z                  |        0.00635164 |
| extreme_move_reversal_flag_rank |        0.00382848 |
| reversal_5d_rank                |        0.0014295  |

## What Worked

- Cross-sectional ranking and basic preprocessing kept signals comparable across heterogeneous stocks.
- The portfolio construction layer explicitly controlled sector concentration and market beta drift.
- Transaction-cost-aware evaluation made the research output closer to a production-style alpha study.

## Limitations

- Free data introduces survivorship bias, stale sector labels, and occasional corporate-action quirks.
- The default universe is a liquid prototype universe rather than a full institutional coverage list.
- Targets are daily-bar based and do not model intraday execution or borrow constraints.

## Next Steps

- Add premium or point-in-time fundamentals and sector mappings.
- Extend the model set to boosted trees and residualized-return targets.
- Add richer risk-model neutralization and regime-aware allocation.
