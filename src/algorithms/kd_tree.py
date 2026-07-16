from __future__ import annotations

from typing import Literal

import numpy as np

from src.utils import distance_sq, manhetten


class KDNode:
    """
    A node in a kd-tree.

    Attributes:
        point (np.ndarray): The coordinate point stored in the node.
        axis (int): The splitting axis (dimension index) for this node.
        left (KDNode | None): The left child node.
        right (KDNode | None): The right child node.
    """

    def __init__(
        self,
        point: np.ndarray,
        axis: int,
        left: KDNode | None = None,
        right: KDNode | None = None,
    ) -> None:
        """
        Initializes a KDNode with a point, axis, and optional children.

        Args:
            point (np.ndarray): The coordinate point stored in the node.
            axis (int): The splitting axis (dimension index).
            left (KDNode | None, optional): The left child node. Defaults to None.
            right (KDNode | None, optional): The right child node. Defaults to None.
        """
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right


class KDTree:
    """A kd-tree for fast spatial search.

    Attributes:
        root (KDNode | None): The root node of the tree.
        k (int): The number of dimensions.
    """

    def __init__(
        self,
        root: KDNode | None = None,
        k: int = 2,
        dist: Literal["manhatten", "euclidian"] = "euclidian",
    ) -> None:
        """
        Initializes a KDTree.

        Args:
            root (KDNode | None, optional): The root node of the tree. Defaults to None.
            k (int, optional): The number of dimensions. Defaults to 2.
            dist (Literal["manhatten", "euclidian"], optional): The distance metric to use
                for nearest neighbor search ("manhatten" or "euclidian"). Defaults to "euclidian".
        """
        self._dist = dist
        self.root = root
        self.k = k

    def build(self, points: list[tuple[float, ...]]) -> None:
        """
        Builds the KDTree from a list of points.

        Args:
            points (list[tuple[float, ...]]): A list of points (tuples of coordinates)
                to construct the tree from.
        """
        if not points:
            self.root = None
        else:
            points_array = np.asarray(points, dtype=float)
            self.root = self._build(points_array, depth=0)

    def _build(self, points: np.ndarray, depth: int) -> KDNode | None:
        if points.size == 0:
            return None

        axis = depth % self.k
        sorted_indices = np.argsort(points[:, axis])
        sorted_points = points[sorted_indices]

        median_index = len(sorted_points) // 2
        median_point = sorted_points[median_index]

        left_points = sorted_points[:median_index]
        right_points = sorted_points[median_index + 1 :]

        left_child = self._build(left_points, depth + 1)
        right_child = self._build(right_points, depth + 1)

        return KDNode(median_point, axis, left_child, right_child)

    def insert(self, point: tuple[float, ...] | np.ndarray) -> None:
        """
        Inserts a new point into the KDTree.

        Args:
            point (tuple[float, ...] | np.ndarray): The point (as a tuple or numpy array)
                to insert into the tree.
        """
        point_array = np.asarray(point, dtype=float)
        self.root = self._insert(node=self.root, point=point_array)

    def _insert(self, node: KDNode | None, point: np.ndarray, depth: int = 0) -> KDNode:
        if node is None:
            axis = depth % self.k
            return KDNode(point, axis)
        else:
            axis = node.axis
            if point[axis] < node.point[axis]:
                node.left = self._insert(node.left, point, depth + 1)
            else:
                node.right = self._insert(node.right, point, depth + 1)
            return node

    def find_nearest_point(self, point: tuple[float, ...] | np.ndarray) -> np.ndarray:
        """
        Finds the nearest point in the KDTree to the query point.

        Args:
            point (tuple[float, ...] | np.ndarray): The query point as a tuple
                or numpy array.

        Returns:
            np.ndarray: The point in the KDTree that is closest to the query point.

        Raises:
            ValueError: If the KDTree is empty (has no root).
        """
        if self.root is None:
            raise ValueError("Tree is empty")

        target = np.asarray(point, dtype=float)
        best_node, _ = self._find_nearest_node(self.root, target)

        return best_node.point

    def _find_nearest_node(
        self, node: KDNode, point: np.ndarray
    ) -> tuple[KDNode, float]:
        axis = node.axis

        if point[axis] < node.point[axis]:
            next_node, other_node = node.left, node.right
        else:
            next_node, other_node = node.right, node.left

        best_node = node
        if self._dist == "euclidian":
            best_dist = distance_sq(node.point, point)
        else:
            best_dist = manhetten(node.point, point)

        if next_node is not None:
            candidate_node, candidate_dist = self._find_nearest_node(next_node, point)
            if candidate_dist < best_dist:
                best_node, best_dist = candidate_node, candidate_dist

        if self._dist == "euclidian":
            plane_dist = (point[axis] - node.point[axis]) ** 2
        else:
            plane_dist = np.abs(point[axis] - node.point[axis])

        if other_node is not None and plane_dist < best_dist:
            candidate_node, candidate_dist = self._find_nearest_node(other_node, point)
            if candidate_dist < best_dist:
                best_node, best_dist = candidate_node, candidate_dist

        return best_node, best_dist
