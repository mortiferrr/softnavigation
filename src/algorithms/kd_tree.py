from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

from src.utils import distance_sq, manhattan


class KDNode:
    """
    A node in a kd-tree.

    Attributes:
        point (tuple[int | float, ...]): The coordinate point stored in the node.
        axis (int): The splitting axis (dimension index) for this node.
        left (KDNode | None): The left child node.
        right (KDNode | None): The right child node.
        node_id (int | None): The ID of current node.
    """

    __slots__ = ("point", "axis", "left", "right", "node_id")

    def __init__(
        self,
        point: tuple[int | float, ...],
        axis: int,
        left: KDNode | None = None,
        right: KDNode | None = None,
        node_id: int | None = None,
    ) -> None:
        """Initializes a KDNode with a point, axis, and optional children."""
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right
        self.node_id = node_id


# TODO: Add to rebalance of kd-tree
class KDTree:
    """
    A kd-tree for fast spatial search.

    Attributes:
        root (KDNode | None): The root node of the tree.
        k (int): The number of dimensions.
        dist ("manhattan" | "euclidean"]): Function for calculating distance to target.
    """

    def __init__(
        self,
        k: int = 2,
        dist: Literal["manhattan", "euclidean"] = "euclidean",
    ) -> None:
        """Initializes a KDTree."""
        self.root = None
        self.k = k
        self.points: set[tuple[int | float, ...]] = set()
        self._dist = distance_sq if dist == "euclidean" else manhattan

    def build(
        self, points: list[tuple[int | float, ...]], ids: list[int] | None = None
    ) -> None:
        """Builds the KDTree from a list of points."""
        if len(points) == 0:
            raise ValueError("Points list is empty")
        if ids and len(ids) != len(points):
            raise ValueError("Lengths of `ids` and `points` must be same")

        points_array = np.array(points, dtype=np.float64)
        indices = np.arange(len(points_array))
        self.root = self._build(points_array, indices, 0, ids)

    def _build(
        self, points: np.ndarray, indices: np.ndarray, depth: int, ids: list[int] | None
    ) -> KDNode | None:
        if indices.size == 0:
            return None

        axis = depth % self.k
        median_index = len(indices) // 2

        # NOTE: O(N) instead of O(N*log(N))
        partitioned = np.argpartition(points[indices, axis], median_index)
        sorted_indices = indices[partitioned]

        # NOTE: Median point - dividing line
        median_point = points[sorted_indices[median_index]]
        median_point = tuple(median_point.tolist())
        node_id = sorted_indices[median_index] if ids else None

        left_indices = sorted_indices[:median_index]
        right_indices = sorted_indices[median_index + 1 :]

        left_child = self._build(points, left_indices, depth + 1, ids)
        right_child = self._build(points, right_indices, depth + 1, ids)

        self.points.add(median_point)
        return KDNode(median_point, axis, left_child, right_child, node_id)

    def insert(
        self, point: tuple[int | float, ...], node_id: int | None = None
    ) -> None:
        """Inserts a new point into the KDTree."""
        self.root = self._insert(self.root, point, 0, node_id)
        self.points.add(point)

    def _insert(
        self,
        node: KDNode | None,
        point: tuple[int | float, ...],
        depth: int,
        node_id: int | None,
    ) -> KDNode:
        if node is None:
            return KDNode(point, depth % self.k, node_id=node_id)

        next_node = node

        # The loop is needed to avoid a `RecursionError` if the tree is too deep
        while True:
            axis = next_node.axis
            if point[axis] < next_node.point[axis]:
                if next_node.left is None:
                    next_node.left = KDNode(
                        point, (depth + 1) % self.k, node_id=node_id
                    )
                    break
                next_node = next_node.left
            else:
                if next_node.right is None:
                    next_node.right = KDNode(
                        point, (depth + 1) % self.k, node_id=node_id
                    )
                    break
                next_node = next_node.right
            depth += 1

        return node

    def find_nearest_point(
        self, target: tuple[int | float, ...], return_id: bool = False
    ) -> int | tuple[int | float, ...]:
        """Finds the nearest point in the KDTree to the query point."""
        if self.root is None:
            raise ValueError("Tree is empty")

        best_node, _ = self._find_nearest_node(self.root, target)
        if return_id:
            if best_node.node_id is None:
                warnings.warn(
                    message="Nearest node found has no ID; coordinates will be returned",
                    stacklevel=2,
                )
                return best_node.point
            return best_node.node_id

        return best_node.point

    def _find_nearest_node(
        self, node: KDNode, point: tuple[int | float, ...]
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
