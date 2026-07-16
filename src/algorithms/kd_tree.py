from __future__ import annotations

from typing import Literal

import numpy as np

from src.utils import distance_sq, manhattan


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
    """
    A kd-tree for fast spatial search.

    Attributes:
        root (KDNode | None): The root node of the tree.
        k (int): The number of dimensions.
    """

    def __init__(
        self,
        k: int = 2,
        dist: Literal["manhattan", "euclidean"] = "euclidean",
    ) -> None:
        """
        Initializes a KDTree.

        Args:
            k (int, optional): The number of dimensions. Defaults to 2.
            dist (Literal["manhattan", "euclidean"], optional): The distance metric to use
                for nearest neighbor search ("manhattan" or "euclidean"). Defaults to "euclidean".
        """
        self.root = None
        self._dist = distance_sq if dist == "euclidean" else manhattan
        self.k = k

    def build(self, points: list[tuple[float, ...]] | np.ndarray) -> None:
        """
        Builds the KDTree from a list of points.

        Args:
            points (list[tuple[float, ...]] | np.ndarray): A list of points (tuples of coordinates)
                to construct the tree from.
        """
        if len(points) == 0:
            raise ValueError("Points list is empty")
        points_array = np.asarray(points, dtype=float)
        indices = np.arange(len(points_array))
        self.root = self._build(points_array, indices, 0)

    def _build(
        self, points: np.ndarray, indices: np.ndarray, depth: int = 0
    ) -> KDNode | None:
        if indices.size == 0:
            return None

        axis = depth % self.k
        median_index = len(indices) // 2

        partitioned = np.argpartition(points[indices, axis], median_index)
        sorted_indices = indices[partitioned]

        median_point = points[sorted_indices[median_index]]

        left_indices = sorted_indices[:median_index]
        right_indices = sorted_indices[median_index + 1 :]

        left_child = self._build(points, left_indices, depth + 1)
        right_child = self._build(points, right_indices, depth + 1)

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
            return KDNode(point, depth % self.k)

        next_node = node

        while True:
            axis = next_node.axis
            if point[axis] < next_node.point[axis]:
                if next_node.left is None:
                    next_node.left = KDNode(point, depth % self.k)
                    break
                next_node = next_node.left
            else:
                if next_node.right is None:
                    next_node.right = KDNode(point, depth % self.k)
                    break
                next_node = next_node.right
            depth += 1

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
        best_dist = float("inf")

        if next_node is not None:
            best_node, best_dist = self._find_nearest_node(next_node, point)

        dist_to_node = self._dist(node.point, point)
        if dist_to_node < best_dist:
            best_node = node
            best_dist = dist_to_node

        plane_dist = self._dist(point[axis], node.point[axis])

        if other_node is not None and plane_dist < best_dist:
            candidate_node, candidate_dist = self._find_nearest_node(other_node, point)
            if candidate_dist < best_dist:
                best_node, best_dist = candidate_node, candidate_dist

        return best_node, best_dist
