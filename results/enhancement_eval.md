# Enhancement filter comparison

Produced by `MATLAB Codes/Phase3_Enhancement.m`.

## Real data (OASBUD hold-out patient, A-line B-mode)

ROIs [row1 row2 col1 col2]: lesion [780 940 250 320], background [780 940 380 450], homogeneous [780 940 100 170]; boundary band from the OASBUD lesion outline.

| Filter | CNR | gCNR | Speckle SNR | Edge ratio |
|---|---|---|---|---|
| None | 2.0463 | 0.8969 | 4.5276 | 0.6978 |
| Median 3x3 | 2.2169 | 0.9234 | 4.8795 | 0.6583 |
| Wiener 5x5 | 2.3059 | 0.9334 | 5.0515 | 0.5402 |
| Bilateral | 2.5376 | 0.9579 | 5.4701 | 0.5674 |

## Synthetic phantom (scored against the clean phantom)

| Filter | Variance | PSNR (dB) | SSIM | Gradient corr | CNR |
|---|---|---|---|---|---|
| None | 0.01 | 26.05 | 0.3228 | 0.6444 | 6.6830 |
| Median 3x3 | 0.01 | 31.47 | 0.6593 | 0.7704 | 12.6044 |
| Wiener 5x5 | 0.01 | 36.52 | 0.9119 | 0.9051 | 32.8368 |
| Bilateral | 0.01 | 38.09 | 0.9679 | 0.9162 | 43.2072 |
| None | 0.05 | 19.18 | 0.1118 | 0.2889 | 2.9945 |
| Median 3x3 | 0.05 | 24.60 | 0.3190 | 0.3837 | 5.5573 |
| Wiener 5x5 | 0.05 | 29.97 | 0.7120 | 0.6597 | 12.8493 |
| Bilateral | 0.05 | 31.66 | 0.7470 | 0.7389 | 13.8743 |
| None | 0.10 | 16.20 | 0.0675 | 0.1718 | 2.1111 |
| Median 3x3 | 0.10 | 21.66 | 0.2070 | 0.2454 | 3.9741 |
| Wiener 5x5 | 0.10 | 27.16 | 0.5496 | 0.4886 | 9.0818 |
| Bilateral | 0.10 | 26.41 | 0.4154 | 0.4605 | 7.4657 |

Mean rank over the seven criteria (1 is best): Median 3x3 2.71, Wiener 5x5 2.14, Bilateral 1.14

Selected filter: **Bilateral**
