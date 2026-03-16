import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def compute_metrics(y_true_batch: np.ndarray, y_pred_batch: np.ndarray) -> dict[str, float]:
    if len(y_true_batch) == 0:
        return {"CharAccuracy": 0.0, "CharPrecision": 0.0, "CharRecall": 0.0, "CharF1": 0.0, "DotAccuracy": 0.0}

    powers = 1 << np.arange(y_true_batch.shape[1] - 1, -1, -1)
    y_true_char = y_true_batch.astype(int).dot(powers)
    y_pred_char = y_pred_batch.astype(int).dot(powers)

    char_acc = float(np.mean(y_true_char == y_pred_char))
    char_p, char_r, char_f1, _ = precision_recall_fscore_support(
        y_true_char, y_pred_char, average="macro", zero_division=0
    )

    dot_acc = float(np.mean(y_true_batch.astype(int) == y_pred_batch.astype(int)))

    return {
        "CharAccuracy": round(char_acc, 4),
        "CharPrecision": round(float(char_p), 4),
        "CharRecall": round(float(char_r), 4),
        "CharF1": round(float(char_f1), 4),
        "DotAccuracy": round(dot_acc, 4),
    }