# FieldNet benchmark

## Accuracy — relative L2 vs FEM ground truth (%)

| model | split | sigma_vm | displacement | u_x | u_y | peak-stress err |
|---|---|---|---|---|---|---|
| FNO | in-distribution | 0.55 | 0.47 | 0.72 | 0.94 | 1.95 |
| FNO | OOD geometry | 2.20 | 1.59 | 2.14 | 2.76 | 6.82 |
| DEEPONET | in-distribution | 2.19 | 0.97 | 1.27 | 1.47 | 8.36 |
| DEEPONET | OOD geometry | 5.94 | 3.59 | 4.15 | 6.58 | 12.50 |

## Inference cost & speedup

| pipeline | mode | time per field (ms) | speedup vs FEM |
|---|---|---|---|
| FEM solve | one solve | 797.75 ± 19.33 | 1x |
| FNO (CPU) | single forward | 313.754 ± 7.248 | 3x |
| FNO (GPU) | single forward | 5.019 ± 0.404 | 159x |
| FNO (GPU) | batched throughput | 1.966 ± 0.001 | 406x |
| DEEPONET (CPU) | single forward | 15.630 ± 1.121 | 51x |
| DEEPONET (GPU) | single forward | 1.631 ± 0.185 | 489x |
| DEEPONET (GPU) | batched throughput | 1.136 ± 0.025 | 702x |

Best speedup: **702x**  
OOD sigma_vm rel L2 under 8%: **True**
