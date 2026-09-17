"""Compiled least-squares fitting and robust rejection for differential correction."""

from difforb.od.dc.lsq.lsq import (
    LSQTermination,
    LeastSquares,
    LeastSquaresResult,
    PriorCovarianceResult,
    RobustLeastSquares,
    RobustResult,
)

__all__ = [
    "LSQTermination", "LeastSquares", "LeastSquaresResult",
    "PriorCovarianceResult", "RobustLeastSquares", "RobustResult",
]
