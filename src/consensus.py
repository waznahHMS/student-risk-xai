"""
Measures agreement between the Random Forest and Logistic Regression.
Agreement strengthens trust; disagreement is a red flag for the teacher.
"""

def compute_consensus(rf_proba, lr_proba):
    """
    Returns:
      rf_label, lr_label : 0/1 predictions
      agreement          : True/False
      gap                : absolute probability gap (0..1)
    """
    rf_label = int(rf_proba >= 0.5)
    lr_label = int(lr_proba >= 0.5)
    agreement = rf_label == lr_label
    gap = abs(rf_proba - lr_proba)
    return rf_label, lr_label, agreement, gap
