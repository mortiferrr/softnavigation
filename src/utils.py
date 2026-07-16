import numpy as np


def distance_sq(p1: np.ndarray | float, p2: np.ndarray | float) -> float:
    """
    Calculate the squared Euclidean distance between two points.

    Args:
        p1 (np.ndarray | float): The first point.
        p2 (np.ndarray | float): The second point.

    Returns:
        float: The squared Euclidean distance between the two points.
    """
    if isinstance(p1, float) and isinstance(p2, float):
        return (p1 - p2) ** 2
    elif isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
        return float(np.sum((p1 - p2) ** 2))
    else:
        raise TypeError("p1 and p2 must be both float or both np.ndarray")


def euclidean(p1: np.ndarray | float, p2: np.ndarray | float) -> float:
    """
    Calculate the Euclidean distance between two points.

    Args:
        p1 (np.ndarray | float): The first point.
        p2 (np.ndarray | float): The second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    if isinstance(p1, float) and isinstance(p2, float):
        return abs(p1 - p2)
    elif isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
        return float(np.linalg.norm(p1 - p2))
    else:
        raise TypeError("p1 and p2 must be both float or both np.ndarray")


def manhattan(p1: np.ndarray | float, p2: np.ndarray | float) -> float:
    """
    Calculate the Manhattan distance between two points.

    Args:
        p1 (np.ndarray | float): The first point.
        p2 (np.ndarray | float): The second point.

    Returns:
        float: The Manhattan distance between the two points.
    """
    if isinstance(p1, float) and isinstance(p2, float):
        return abs(p1 - p2)
    elif isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray):
        return float(np.sum(np.abs(p1 - p2)))
    else:
        raise TypeError("p1 and p2 must be both float or both np.ndarray")
