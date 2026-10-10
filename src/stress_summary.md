# Stress test: Newton vs Adam vs hybrid (lowrank_glm.jl)

Threads: 8. Train objective = penalised NLL on the 90 % training entries; held-out = mean per-entry deviance on the 10 % masked entries (NB / ZINB: -2 log-lik); cor(η) on ≤ 4000 rows; angle = largest principal angle between true and fitted loadings; heap = peak live Julia heap during the fit above the pre-fit level (includes the fitter's internal copy of the data); RSS = process peak (not per fit).

## Key findings (hand-written, run of 2026-10-06, 8 threads; the tables below are auto-generated)

- **Adam wins at large k and for the expensive likelihoods.** Its cost per epoch is O(npk) and runs on BLAS, so time barely grows with k. At 20k×2k, Adam vs Newton took: Poisson k=20 29 s vs 134 s; Poisson k=50 30 s vs 542 s; logistic k=50 41 s vs 135 s; NB k=20 62 s vs 214 s. Held-out deviance and cor(η) are the same to 3–4 digits.
- **ZINB: Adam is the only practical solver beyond small data.** At 2k×500 it was 4–8× faster. At 20k×2k k=5 it converged in 106 s to a *lower* objective than Newton reached in its 600 s cap. At 100k×2k sparse it took 256 s (k=5) and 309 s (k=20); Newton was not attempted (predicted > 50 min).
- **Newton wins at small k (k=5) when n is moderate or large.** Examples: Poisson 20k×2k 5.7 s vs Adam 23 s, 100k×2k 29 s vs 68 s; logistic 100k×2k k=5/20 32/50 s vs 116/137 s; NB 20k×2k k=5 14 s vs 53 s. It needs only 5–15 sweeps there and reaches the lowest objective.
- **Newton is slow at small n with sparse counts:** 100–260 sweeps from slow alternating convergence (e.g. ZINB 2k×500 35 s vs Adam 4 s).
- **Train objective:** Adam typically ends 1e-5–1e-3 (relative) above Newton because of the stochastic noise floor. Held-out deviance is equal or slightly better (mild early-stopping regularisation), e.g. over-ranked ZINB/NB k=50 at 2k×500.
- **Hybrid** (Adam, then ≤ 20 Newton sweeps) gives the lowest objective in many settings at roughly 1.1–1.5× Adam time. Worth it when you need a tighter optimum than Adam alone. At large k its Newton polish dominates the cost.
- **Memory:** live-heap growth per fit is about 1–2× the data size (the fitter's internal copy plus buffers). It is similar for Newton and Adam (Adam adds O((n+p)k) moment arrays). Peak process RSS was about 4 GB at 100k×2k sparse (28 % nonzero, Int32 indices).
- **Skipped by prediction (> limit):** large Poisson Newton k=20/50; all large ZINB except Adam k=5/20; medium/large ZINB Newton k≥20; several hybrids at large k (rows are marked `skipped_pred_*`).

| size | family | k | solver | status | time (s) | iters | Δobj vs best (rel) | held-out dev | cor(η) | angle (°) | heap (MB) | alloc (GB) |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small 2000×500 | poisson | 5 | newton | converged | 1.78 | 151 | 0.0e+00 | 0.744848 | 0.98604 | 5.644 | 14.0 | 0.014 |
| small 2000×500 | poisson | 5 | adam | converged | 0.59 | 64 | 6.0e-04 | 0.74514 | 0.98589 | 5.66 | 17.0 | 0.017 |
| small 2000×500 | poisson | 5 | hybrid | maxiter | 0.79 | 84 | 3.7e-04 | 0.744768 | 0.98605 | 5.651 | 19.0 | 0.018 |
| small 2000×500 | poisson | 20 | newton | converged | 2.6 | 124 | 0.0e+00 | 0.797618 | 0.94332 | 14.727 | 18.0 | 0.017 |
| small 2000×500 | poisson | 20 | adam | converged | 1.32 | 63 | 8.0e-04 | 0.797323 | 0.9433 | 14.665 | 22.0 | 0.021 |
| small 2000×500 | poisson | 20 | hybrid | maxiter | 1.65 | 83 | 3.4e-04 | 0.797195 | 0.94335 | 14.701 | 23.0 | 0.022 |
| small 2000×500 | poisson | 50 | newton | converged | 7.15 | 102 | 0.0e+00 | 1.007246 | 0.8342 | 34.083 | 25.0 | 0.025 |
| small 2000×500 | poisson | 50 | adam | converged | 3.28 | 127 | 5.4e-04 | 1.006525 | 0.83346 | 38.326 | 34.0 | 0.034 |
| small 2000×500 | poisson | 50 | hybrid | maxiter | 4.65 | 147 | 8.1e-05 | 1.006348 | 0.83423 | 33.991 | 36.0 | 0.035 |
| small 2000×500 | bernoulli | 5 | newton | converged | 0.82 | 29 | 0.0e+00 | 0.818711 | 0.98414 | 5.183 | 13.0 | 0.012 |
| small 2000×500 | bernoulli | 5 | adam | converged | 0.97 | 68 | 7.5e-04 | 0.818626 | 0.98396 | 5.206 | 18.0 | 0.017 |
| small 2000×500 | bernoulli | 5 | hybrid | maxiter | 1.48 | 88 | 2.9e-05 | 0.818661 | 0.98413 | 5.178 | 19.0 | 0.018 |
| small 2000×500 | bernoulli | 20 | newton | converged | 0.9 | 24 | 6.2e-06 | 0.857477 | 0.94506 | 12.602 | 16.0 | 0.016 |
| small 2000×500 | bernoulli | 20 | adam | converged | 2.23 | 70 | 5.0e-04 | 0.856832 | 0.94527 | 12.517 | 22.0 | 0.021 |
| small 2000×500 | bernoulli | 20 | hybrid | converged | 2.87 | 90 | 0.0e+00 | 0.857353 | 0.94507 | 12.597 | 23.0 | 0.023 |
| small 2000×500 | bernoulli | 50 | newton | converged | 3.39 | 41 | 9.8e-06 | 1.038915 | 0.85676 | 28.587 | 25.0 | 0.024 |
| small 2000×500 | bernoulli | 50 | adam | converged | 3.86 | 105 | 4.5e-04 | 1.033199 | 0.85964 | 27.545 | 33.0 | 0.032 |
| small 2000×500 | bernoulli | 50 | hybrid | converged | 5.18 | 121 | 0.0e+00 | 1.038652 | 0.8569 | 28.514 | 34.0 | 0.033 |
| small 2000×500 | zinb | 5 | newton | converged | 35.3 | 264 | 0.0e+00 | 1.654365 | 0.97498 | 9.552 | 21.0 | 0.02 |
| small 2000×500 | zinb | 5 | adam | converged | 4.26 | 111 | 7.9e-04 | 1.653264 | 0.97673 | 8.706 | 26.0 | 0.025 |
| small 2000×500 | zinb | 5 | hybrid | maxiter | 7.07 | 131 | 4.2e-04 | 1.653628 | 0.97598 | 8.901 | 27.0 | 0.026 |
| small 2000×500 | zinb | 20 | newton | converged | 36.05 | 246 | 0.0e+00 | 1.781189 | 0.8916 | 26.144 | 25.0 | 0.024 |
| small 2000×500 | zinb | 20 | adam | converged | 8.72 | 142 | 1.2e-03 | 1.765176 | 0.90141 | 23.489 | 33.0 | 0.032 |
| small 2000×500 | zinb | 20 | hybrid | maxiter | 11.68 | 162 | 5.5e-04 | 1.773871 | 0.89695 | 24.885 | 34.0 | 0.033 |
| small 2000×500 | zinb | 50 | newton | maxiter | 60.74 | 300 | 0.0e+00 | 4.249387 | 0.56477 | 85.206 | 34.0 | 0.033 |
| small 2000×500 | zinb | 50 | adam | maxiter | 13.76 | 200 | 7.6e-03 | 3.008236 | 0.59851 | 88.992 | 46.0 | 0.045 |
| small 2000×500 | zinb | 50 | hybrid | maxiter | 18.49 | 220 | 3.7e-03 | 3.414132 | 0.58017 | 89.296 | 30.0 | 0.047 |
| medium 20000×2000 | poisson | 5 | newton | converged | 5.74 | 11 | 0.0e+00 | 0.729643 | 0.99709 | 1.757 | 343.0 | 0.369 |
| medium 20000×2000 | poisson | 5 | adam | converged | 23.21 | 50 | 7.7e-05 | 0.72975 | 0.99702 | 1.764 | 371.0 | 0.397 |
| medium 20000×2000 | poisson | 5 | hybrid | converged | 23.84 | 52 | 2.8e-06 | 0.729642 | 0.99708 | 1.756 | 379.0 | 0.405 |
| medium 20000×2000 | poisson | 20 | newton | converged | 133.72 | 153 | 1.9e-05 | 0.737706 | 0.98816 | 3.851 | 378.0 | 0.404 |
| medium 20000×2000 | poisson | 20 | adam | converged | 28.88 | 53 | 5.7e-05 | 0.737773 | 0.9881 | 3.852 | 407.0 | 0.431 |
| medium 20000×2000 | poisson | 20 | hybrid | converged | 30.19 | 56 | 0.0e+00 | 0.737675 | 0.98816 | 3.852 | 415.0 | 0.44 |
| medium 20000×2000 | poisson | 50 | newton | converged | 541.56 | 192 | 0.0e+00 | 0.758098 | 0.96989 | 6.821 | 415.0 | 0.471 |
| medium 20000×2000 | poisson | 50 | adam | converged | 29.85 | 53 | 1.1e-04 | 0.758101 | 0.96982 | 6.826 | 243.0 | 0.5 |
| medium 20000×2000 | poisson | 50 | hybrid | maxiter | 87.01 | 73 | 2.6e-05 | 0.758002 | 0.96988 | 6.817 | 252.0 | 0.509 |
| medium 20000×2000 | bernoulli | 5 | newton | converged | 14.66 | 14 | 0.0e+00 | 0.803577 | 0.99657 | 1.672 | 343.0 | 0.369 |
| medium 20000×2000 | bernoulli | 5 | adam | converged | 39.69 | 54 | 1.8e-04 | 0.803633 | 0.99639 | 1.674 | 372.0 | 0.397 |
| medium 20000×2000 | bernoulli | 5 | hybrid | maxiter | 57.3 | 74 | 3.4e-05 | 0.803566 | 0.99657 | 1.67 | 380.0 | 0.406 |
| medium 20000×2000 | bernoulli | 20 | newton | converged | 52.11 | 43 | 0.0e+00 | 0.802818 | 0.98813 | 3.254 | 376.0 | 0.401 |
| medium 20000×2000 | bernoulli | 20 | adam | converged | 35.52 | 47 | 1.6e-04 | 0.80285 | 0.98806 | 3.269 | 406.0 | 0.431 |
| medium 20000×2000 | bernoulli | 20 | hybrid | maxiter | 63.94 | 67 | 3.4e-05 | 0.802769 | 0.98813 | 3.251 | 415.0 | 0.439 |
| medium 20000×2000 | bernoulli | 50 | newton | converged | 135.43 | 43 | 0.0e+00 | 0.822793 | 0.97127 | 5.727 | 414.0 | 0.465 |
| medium 20000×2000 | bernoulli | 50 | adam | converged | 40.53 | 48 | 1.7e-04 | 0.822815 | 0.97121 | 5.732 | 243.0 | 0.5 |
| medium 20000×2000 | bernoulli | 50 | hybrid | maxiter | 95.94 | 68 | 2.1e-05 | 0.822666 | 0.97127 | 5.725 | 252.0 | 0.509 |
| medium 20000×2000 | zinb | 5 | newton | time_limit | 601.5 | 125 | 2.9e-04 | 1.624445 | 0.99343 | 3.173 | 177.0 | 0.525 |
| medium 20000×2000 | zinb | 5 | adam | converged | 106.0 | 61 | 7.3e-05 | 1.624191 | 0.99515 | 2.823 | 210.0 | 0.558 |
| medium 20000×2000 | zinb | 5 | hybrid | maxiter | 213.59 | 81 | 0.0e+00 | 1.624166 | 0.99521 | 2.819 | 219.0 | 0.567 |
| medium 20000×2000 | zinb | 20 | newton | skipped_pred_4812s |  |  |  |  |  |  |  |  |
| medium 20000×2000 | zinb | 20 | adam | converged | 124.81 | 71 | 0.0e+00 | 1.636493 | 0.98359 | 5.781 | 228.0 | 0.595 |
| medium 20000×2000 | zinb | 20 | hybrid | skipped_pred_1709s |  |  |  |  |  |  |  |  |
| medium 20000×2000 | zinb | 50 | newton | skipped_pred_19021s |  |  |  |  |  |  |  |  |
| medium 20000×2000 | zinb | 50 | adam | converged | 149.61 | 82 | 0.0e+00 | 1.661366 | 0.95706 | 10.623 | 231.0 | 0.667 |
| medium 20000×2000 | zinb | 50 | hybrid | skipped_pred_6754s |  |  |  |  |  |  |  |  |
| large 100000×2000 | poisson | 5 | newton | converged | 29.05 | 11 | 0.0e+00 | 0.722388 | 0.99717 | 0.861 | 890.0 | 1.356 |
| large 100000×2000 | poisson | 5 | adam | converged | 67.68 | 37 | 9.5e-05 | 0.722409 | 0.9971 | 0.894 | 936.0 | 1.401 |
| large 100000×2000 | poisson | 5 | hybrid | converged | 67.24 | 39 | 1.7e-05 | 0.722386 | 0.99717 | 0.841 | 1472.0 | 1.438 |
| large 100000×2000 | poisson | 20 | newton | skipped_pred_669s |  |  |  |  |  |  |  |  |
| large 100000×2000 | poisson | 20 | adam | converged | 88.34 | 44 | 3.2e-05 | 0.73889 | 0.98925 | 1.703 | 1495.0 | 1.599 |
| large 100000×2000 | poisson | 20 | hybrid | converged | 96.51 | 46 | 0.0e+00 | 0.738853 | 0.98929 | 1.698 | 1675.0 | 1.636 |
| large 100000×2000 | poisson | 50 | newton | skipped_pred_2708s |  |  |  |  |  |  |  |  |
| large 100000×2000 | poisson | 50 | adam | converged | 129.11 | 46 | 3.9e-05 | 0.760279 | 0.97263 | 3.064 | 1052.0 | 1.924 |
| large 100000×2000 | poisson | 50 | hybrid | converged | 151.02 | 48 | 0.0e+00 | 0.760106 | 0.97269 | 3.061 | 1085.0 | 1.96 |
| large 100000×2000 | bernoulli | 5 | newton | converged | 32.45 | 5 | 0.0e+00 | 0.811698 | 0.99696 | 0.718 | 1468.0 | 1.433 |
| large 100000×2000 | bernoulli | 5 | adam | converged | 116.32 | 38 | 1.3e-04 | 0.811701 | 0.99684 | 0.716 | 1515.0 | 1.479 |
| large 100000×2000 | bernoulli | 5 | hybrid | converged | 119.15 | 40 | 5.8e-05 | 0.811692 | 0.99696 | 0.71 | 1552.0 | 1.516 |
| large 100000×2000 | bernoulli | 20 | newton | converged | 49.76 | 5 | 0.0e+00 | 0.807055 | 0.98927 | 1.476 | 1601.0 | 1.611 |
| large 100000×2000 | bernoulli | 20 | adam | converged | 137.23 | 39 | 7.9e-05 | 0.806975 | 0.98925 | 1.479 | 1709.0 | 1.669 |
| large 100000×2000 | bernoulli | 20 | hybrid | maxiter | 285.56 | 59 | 2.1e-05 | 0.80702 | 0.98927 | 1.471 | 1748.0 | 1.707 |
| large 100000×2000 | bernoulli | 50 | adam | converged | 130.02 | 29 | 2.3e-04 | 0.817817 | 0.97369 | 2.544 | 2009.0 | 1.964 |
| large 100000×2000 | bernoulli | 50 | newton | converged | 351.95 | 22 | 0.0e+00 | 0.818843 | 0.97365 | 2.541 | 1711.0 | 1.889 |
| large 100000×2000 | bernoulli | 50 | hybrid | maxiter | 474.14 | 49 | 6.9e-05 | 0.818703 | 0.97365 | 2.537 | 1742.0 | 2.003 |
| large 100000×2000 | zinb | 5 | newton | skipped_pred_3008s |  |  |  |  |  |  |  |  |
| large 100000×2000 | zinb | 5 | adam | converged | 255.67 | 32 | 0.0e+00 | 1.591985 | 0.99607 | 1.372 | 966.0 | 2.183 |
| large 100000×2000 | zinb | 5 | hybrid | skipped_pred_1068s |  |  |  |  |  |  |  |  |
| large 100000×2000 | zinb | 20 | newton | skipped_pred_24060s |  |  |  |  |  |  |  |  |
| large 100000×2000 | zinb | 20 | hybrid | skipped_pred_8543s |  |  |  |  |  |  |  |  |
| large 100000×2000 | zinb | 20 | adam | converged | 309.09 | 39 | 0.0e+00 | 1.637562 | 0.98645 | 2.57 | 1040.0 | 2.377 |
| large 100000×2000 | zinb | 50 | newton | skipped_pred_95106s |  |  |  |  |  |  |  |  |
| large 100000×2000 | zinb | 50 | adam | skipped_pred_2557s |  |  |  |  |  |  |  |  |
| large 100000×2000 | zinb | 50 | hybrid | skipped_pred_33771s |  |  |  |  |  |  |  |  |
| small 2000×500 | nb | 5 | newton | converged | 6.38 | 109 | 0.0e+00 | 1.853058 | 0.9874 | 5.671 | 19.0 | 0.018 |
| small 2000×500 | nb | 5 | adam | converged | 1.45 | 57 | 3.7e-04 | 1.852809 | 0.98739 | 5.673 | 21.0 | 0.021 |
| small 2000×500 | nb | 5 | hybrid | maxiter | 2.52 | 77 | 1.7e-04 | 1.852987 | 0.98742 | 5.666 | 23.0 | 0.022 |
| small 2000×500 | nb | 20 | newton | converged | 5.25 | 89 | 0.0e+00 | 1.924877 | 0.95106 | 13.716 | 22.0 | 0.021 |
| small 2000×500 | nb | 20 | adam | converged | 2.99 | 67 | 4.3e-04 | 1.922096 | 0.95114 | 13.705 | 26.0 | 0.026 |
| small 2000×500 | nb | 20 | hybrid | maxiter | 4.28 | 87 | 1.2e-04 | 1.92457 | 0.95109 | 13.702 | 28.0 | 0.027 |
| small 2000×500 | nb | 50 | newton | converged | 13.38 | 138 | 0.0e+00 | 2.28244 | 0.85663 | 32.174 | 31.0 | 0.03 |
| small 2000×500 | nb | 50 | adam | converged | 5.08 | 115 | 1.7e-03 | 2.1923 | 0.86019 | 31.757 | 38.0 | 0.037 |
| small 2000×500 | nb | 50 | hybrid | maxiter | 7.13 | 135 | 4.1e-04 | 2.257268 | 0.85732 | 32.092 | 40.0 | 0.039 |
| medium 20000×2000 | nb | 5 | newton | converged | 13.99 | 6 | 9.5e-05 | 1.827237 | 0.99737 | 1.656 | 206.0 | 0.519 |
| medium 20000×2000 | nb | 5 | adam | converged | 53.05 | 45 | 6.2e-05 | 1.827278 | 0.99732 | 1.659 | 200.0 | 0.547 |
| medium 20000×2000 | nb | 5 | hybrid | converged | 61.89 | 48 | 0.0e+00 | 1.827232 | 0.99737 | 1.654 | 208.0 | 0.555 |
| medium 20000×2000 | nb | 20 | newton | converged | 213.52 | 116 | 0.0e+00 | 1.83156 | 0.98949 | 3.576 | 188.0 | 0.556 |
| medium 20000×2000 | nb | 20 | adam | converged | 61.84 | 49 | 2.7e-05 | 1.831396 | 0.98945 | 3.581 | 215.0 | 0.582 |
| medium 20000×2000 | nb | 20 | hybrid | skipped_pred_495s |  |  |  |  |  |  |  |  |
| medium 20000×2000 | nb | 50 | newton | skipped_pred_844s |  |  |  |  |  |  |  |  |
| medium 20000×2000 | nb | 50 | adam | converged | 67.45 | 52 | 0.0e+00 | 1.852502 | 0.97323 | 6.452 | 375.0 | 0.652 |
| medium 20000×2000 | nb | 50 | hybrid | skipped_pred_1957s |  |  |  |  |  |  |  |  |

## Fastest solver per setting (among runs within 1e-3 relative of the best train objective)

- small poisson k=5: **adam** 0.59 s (newton 1.8s, hybrid 0.8s)
- small poisson k=20: **adam** 1.32 s (newton 2.6s, hybrid 1.6s)
- small poisson k=50: **adam** 3.28 s (newton 7.2s, hybrid 4.7s)
- small bernoulli k=5: **newton** 0.82 s (adam 1.0s, hybrid 1.5s)
- small bernoulli k=20: **newton** 0.9 s (adam 2.2s, hybrid 2.9s)
- small bernoulli k=50: **newton** 3.39 s (adam 3.9s, hybrid 5.2s)
- small zinb k=5: **adam** 4.26 s (newton 35.3s, hybrid 7.1s)
- small zinb k=20: **hybrid** 11.68 s (newton 36.0s, adam 8.7s)
- small zinb k=50: **newton** 60.74 s (adam 13.8s, hybrid 18.5s)
- medium poisson k=5: **newton** 5.74 s (adam 23.2s, hybrid 23.8s)
- medium poisson k=20: **adam** 28.88 s (newton 133.7s, hybrid 30.2s)
- medium poisson k=50: **adam** 29.85 s (newton 541.6s, hybrid 87.0s)
- medium bernoulli k=5: **newton** 14.66 s (adam 39.7s, hybrid 57.3s)
- medium bernoulli k=20: **adam** 35.52 s (newton 52.1s, hybrid 63.9s)
- medium bernoulli k=50: **adam** 40.53 s (newton 135.4s, hybrid 95.9s)
- medium zinb k=5: **adam** 106.0 s (newton 601.5s, hybrid 213.6s)
- medium zinb k=20: **adam** 124.81 s ()
- medium zinb k=50: **adam** 149.61 s ()
- large poisson k=5: **newton** 29.05 s (adam 67.7s, hybrid 67.2s)
- large poisson k=20: **adam** 88.34 s (hybrid 96.5s)
- large poisson k=50: **adam** 129.11 s (hybrid 151.0s)
- large bernoulli k=5: **newton** 32.45 s (adam 116.3s, hybrid 119.2s)
- large bernoulli k=20: **newton** 49.76 s (adam 137.2s, hybrid 285.6s)
- large bernoulli k=50: **adam** 130.02 s (newton 351.9s, hybrid 474.1s)
- large zinb k=5: **adam** 255.67 s ()
- large zinb k=20: **adam** 309.09 s ()
- small nb k=5: **adam** 1.45 s (newton 6.4s, hybrid 2.5s)
- small nb k=20: **adam** 2.99 s (newton 5.2s, hybrid 4.3s)
- small nb k=50: **hybrid** 7.13 s (newton 13.4s, adam 5.1s)
- medium nb k=5: **newton** 13.99 s (adam 53.0s, hybrid 61.9s)
- medium nb k=20: **adam** 61.84 s (newton 213.5s)
- medium nb k=50: **adam** 67.45 s ()
