# CNN hold-out evaluation

Produced by `MATLAB Codes/Phase6_CNN_Evaluation.m` on the test partition stored in `MATLAB Codes/cnn_split.mat` (seed 42, 636 images). The test images were never used for training or validation.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| benign | 84.72% | 84.27% | 0.8449 | 375 |
| malignant | 74.65% | 75.00% | 0.7482 | 212 |
| normal | 90.00% | 91.84% | 0.9091 | 49 |

Overall accuracy: 81.76% (520/636)

Malignant vs rest: AUC-ROC 0.9105, AUC-PR 0.8154

Confusion matrix (rows true, columns predicted, order benign, malignant, normal):

| true \ pred | benign | malignant | normal |
|---|---|---|---|
| benign | 316 | 54 | 5 |
| malignant | 53 | 159 | 0 |
| normal | 4 | 0 | 45 |

Single-image MATLAB inference latency: mean 17.84 ms, std 14.24 ms (n=50)
