import numpy as np

def run_lls(X, y):
    """
    Performs Linear Least Squares training.
    Appends a bias column of ones to X and solves the normal equation.
    
    Args:
        X (np.ndarray): The input feature matrix.
        y (np.ndarray): The target labels.
        
    Returns:
        alpha (np.ndarray): The calculated weight vector.
        X_b (np.ndarray): The design matrix with bias column.
    """
    # Create the design matrix with a bias column
    X_b = np.concatenate((np.ones((len(X), 1)), X), axis=1)
    # Solve for weights (alpha) using the normal equation: (X^T * X) * alpha = X^T * y
    alpha = np.linalg.solve((X_b.T @ X_b), (X_b.T @ y))
    return alpha, X_b

def compute_confusion_matrix(X_b, y, alpha, threshold=0.5):
    """
    Computes the confusion matrix and accuracy for binary classification.
    
    Args:
        X_b (np.ndarray): The design matrix.
        y (np.ndarray): The ground truth labels.
        alpha (np.ndarray): The weight vector.
        threshold (float): Decision boundary threshold.
        
    Returns:
        c (np.ndarray): The confusion matrix [[TP, FP], [FN, TN]].
        accuracy (float): The calculated accuracy.
    """
    # Generate predictions based on the decision threshold
    predictions = (np.dot(X_b, alpha) >= threshold).astype(int)
    # Identify true positives, false positives, true negatives, and false negatives
    tp = np.sum((predictions == 1) & (y == 1))
    fp = np.sum((predictions == 1) & (y == 0))
    tn = np.sum((predictions == 0) & (y == 0))
    fn = np.sum((predictions == 0) & (y == 1))
    # Construct the confusion matrix: [[TP, FP], [FN, TN]]
    c = np.array([[tp, fp], [fn, tn]])
    accuracy = np.trace(c) / len(y)
    return c, accuracy