"""
Sepsis ML: Post-operative sepsis prediction pipeline

A reproducible machine learning framework for predicting post-operative sepsis
using nested cross-validation, comprehensive preprocessing, and ensemble methods.
"""

__version__ = "1.0.0"

# Explicit classification threshold used throughout the pipeline
CLASSIFICATION_THRESHOLD = 0.5

__all__ = [
    "__version__",
    "CLASSIFICATION_THRESHOLD",
]

