"""Stacking classifier, from scratch with numpy only."""

from stacking_ensemble.model import (
    DecisionStump,
    SimpleLogisticRegression,
    StackingClassifier,
)

__all__ = ["DecisionStump", "SimpleLogisticRegression", "StackingClassifier"]
