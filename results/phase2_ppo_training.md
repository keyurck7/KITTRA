# Kittra – PPO placement skill: training and results

In-training evaluation: every 25,000 environment steps, 50 episodes, randomized conditions, deterministic policy. Thresholds use a trailing mean over 3 evaluations. Final evaluation: 100 episodes per condition with fixed seeds 1000–1099, shared by all policies.

## Training

| | PPO seed 0 | PPO seed 1 |
|---|---|---|
| Environment steps trained | 1,875,000 | 3,000,000 |
| Training episodes | ≈ 451,314 (est.) | 657,289 |
| Wall-clock time (laptop CPU, 8 envs) | – | 5.4 h |
| Evaluations | 75 | 120 |
| First reaches 50 % success | 500 k steps / 159,573 ep. | 525 k steps / 194,353 ep. |
| First reaches 70 % success | 925 k steps / 270,579 ep. | 700 k steps / 238,735 ep. |
| First reaches 80 % success | 1.23 M steps / 334,060 ep. | 950 k steps / 294,449 ep. |
| First reaches 90 % success | not reached | 2.08 M steps / 508,399 ep. |
| Best in-training eval (selected checkpoint) | 96 % at 1.38 M | 100 % at 2.38 M |
| Mean of last 5 evaluations | 83 % | 87 % |

## Final evaluation (success rate, 95 % CI, mean placement error)

| Policy | Nominal | Randomized | Out-of-distribution |
|---|---|---|---|
| Scripted expert | 100 % [96, 100] · 0.5 mm | 100 % [96, 100] · 0.8 mm | 99 % [95, 100] · 1.4 mm |
| PPO seed 0 | 87 % [79, 92] · 4.8 mm | 78 % [69, 85] · 5.1 mm | 49 % [39, 59] · 5.9 mm |
| PPO seed 1 | 97 % [92, 99] · 5.1 mm | 90 % [83, 94] · 4.8 mm | 57 % [47, 66] · 5.4 mm |
| Behaviour cloning | 87 % [79, 92] · 3.4 mm | 57 % [47, 66] · 4.2 mm | 14 % [9, 22] · 4.6 mm |

**PPO mean over 2 seeds:** nominal: 92 %, randomized: 84 %, ood: 53 %

_Episode counts marked “est.” are estimated from evaluation episode lengths (TensorBoard logs not found)._
