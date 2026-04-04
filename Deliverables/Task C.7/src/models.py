import numpy as np

def predict_direction(model, X_row, threshold=0.5):
    """
    Returns predicted label and probability of UP (class 1).
    """
    prob_up = float(model.predict_proba(X_row.reshape(1, -1))[0][1])
    pred = 1 if prob_up >= threshold else 0
    return pred, prob_up