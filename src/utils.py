import numpy as np


def distance_sq(p1: np.ndarray | float, p2: np.ndarray | float) -> float:
    """Calculate the squared Euclidean distance between two points."""
    return float(np.sum((p1 - p2) ** 2))


def euclidean(p1: np.ndarray | float, p2: np.ndarray | float) -> float:
    """Calculate the Euclidean distance between two points."""
    return float(np.linalg.norm(p1 - p2))


def manhattan(p1: np.ndarray | float, p2: np.ndarray | float) -> float:
    """Calculate the Manhattan distance between two points."""
    return float(np.sum(np.abs(p1 - p2)))
