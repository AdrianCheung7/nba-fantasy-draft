# Research v3 — injury risk, team composition, other factors (2026-09-29)

Three research agents backtested candidate factors on our own data: predict season N (2024–2026) from seasons ≤ N−1,
players with ≥ 20 GP, fantasy points per game (FP/g) MAE. Scratch scripts are not in the repo; decisions are.

## Adopted (user approved "go" 2026-09-29; each ships only if it survives the combined backtest)

| # | Change | Measured gain | Why |
|---|---|---|---|
| A | Blend model per-game stats 50/50 with ESPN preseason projections (weight refit from saved past seasons, fallback 0.5); keep model GP. Fixes rookie bug: ESPN proj has REB only, OREB/DREB were 0 → rookies underrated ~4–8 FP/g. | top-150 FP/g MAE 4.208 → 3.901 (−7.3%), all 3 seasons | Forecast combination (Bates & Granger 1969; Clemen 1989); ESPN ≈ as accurate as us but errs differently. |
| B | Team shot budget: `stat *= (rate_league / rate_team) ** lam` for PTS/FGM/FGA/FTM/FTA/3PM/3PA/TO; rate = Σstat/ΣMIN over a team's projected players; lam 1 for team-changers, 0.5 others. Team per game comes free from ESPN `proTeamId`. | FP/g −0.043 overall, −0.131 for team-changers, ~0 on stable rosters | Finite possessions ("skill curve", Oliver; SCHOENE team consistency, Pelton). Must be re-tested after A (ESPN already reflects trades). |
| C | GP model features → GP_1..3, MIN_1..3, age, age² | GP MAE 17.93 → 17.58, all 3 seasons (bootstrap CI −0.59..−0.09) | Injury-prone extras (absence streaks, seasons < 62 GP, minutes load, shutdowns) added nothing measurable. |
| D | Sim availability: beta-binomial, a ~ Beta(m·κ, (1−m)·κ), κ fit by method of moments on rotation players (~2.3) | P(GP ≤ 41) for draftable players: actual 14%, old sim 3%, new 17% | Old binomial draw had SD ≈ 4 games vs real ≈ 19. |
| E | Sim only: 2nd night of a back-to-back, players > 22 MIN: play prob × (1 − clip(0.012·(age−24), 0, 0.14)) | Extra miss rate +0.4 pt (≤25) … +13 pt (34+); stars +7–12 pt every season | Affects weekly distribution, not season totals (GP model already includes it). |

## Rejected (no measurable gain on our data)
Injury-prone flags; teammate "pressure" ridge feature (hurts stable rosters); roster-change delta; shooting % shrinkage
(ridge already regresses it); age curves by position / age hinge; minutes & role trends; games-started rate; team pace;
late-season tanking; contract year and coaching change (no free data). Preseason injury-status multipliers
(OUT .75 / IR .5 / SUSP .9) can't be fitted yet: ESPN exposes only current status — log it each season to fit later.
