# Kittra – Phase 2 results

Success rate with 95 % Wilson confidence interval; 100 episodes per cell unless noted.

## Imitation learning (pick-and-place)

_no runs found_

## Placement skill (success rate / mean placement error)

| Policy | nominal | randomized | ood |
|---|---|---|---|
| bc_place | 87.0 % / 3.4 mm | 57.0 % / 4.2 mm | 14.0 % / 4.6 mm |
| place_expert | 100.0 % / 0.5 mm | 100.0 % / 0.8 mm | 99.0 % / 1.4 mm |
| ppo_place | 87.0 % / 4.8 mm | 78.0 % / 5.1 mm | 49.0 % / 5.9 mm |

## Inference time per control step (nominal condition)

| Policy | ms / step |
|---|---|
| bc_place | 0.16 |
| place_expert | 0.27 |
| ppo_place | 0.52 |

