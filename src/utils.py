import numpy as np


def distance_sq(
    p1: np.ndarray | list | tuple | int | float,
    p2: np.ndarray | list | tuple | int | float,
) -> float:
    """Calculate the squared Euclidean distance between two points."""
    if not isinstance(p1, np.ndarray) or not isinstance(p1, np.ndarray):
        p1, p2 = np.asarray(p1), np.asarray(p2)

    return float(np.sum((p1 - p2) ** 2))


def euclidean(
    p1: np.ndarray | list | tuple | int | float,
    p2: np.ndarray | list | tuple | int | float,
) -> float:
    """Calculate the Euclidean distance between two points."""
    if not isinstance(p1, np.ndarray) or not isinstance(p1, np.ndarray):
        p1, p2 = np.asarray(p1), np.asarray(p2)

    return float(np.linalg.norm(p1 - p2))


def manhattan(
    p1: np.ndarray | list | tuple | int | float,
    p2: np.ndarray | list | tuple | int | float,
) -> float:
    """Calculate the Manhattan distance between two points."""
    if not isinstance(p1, np.ndarray) or not isinstance(p1, np.ndarray):
        p1, p2 = np.asarray(p1), np.asarray(p2)

    return float(np.sum(np.abs(p1 - p2)))
