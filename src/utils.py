import numpy as np


def distance_sq(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate the squared Euclidean distance between two points.

    Args:
        p1 (np.ndarray): The first point.
        p2 (np.ndarray): The second point.

    Returns:
        float: The squared Euclidean distance between the two points.
    """
    return float(np.sum((p1 - p2) ** 2))


def euclidean(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two points.

    Args:
        p1 (np.ndarray): The first point.
        p2 (np.ndarray): The second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    return float(np.linalg.norm(p1 - p2))


def manhattan(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate the Manhattan distance between two points.

    Args:
        p1 (np.ndarray): The first point.
        p2 (np.ndarray): The second point.

    Returns:
        float: The Manhattan distance between the two points.
    """
    return float(np.sum(np.abs(p1 - p2)))
