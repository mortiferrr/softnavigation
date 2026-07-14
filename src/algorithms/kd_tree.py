from __future__ import annotations
import numpy as np


class KDNode:
    def __init__(
        self,
        point: np.ndarray,
        axis: int,
        left: KDNode | None = None,
        right: KDNode | None = None,
    ) -> None:
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right


class KDTree:
    def __init__(self, root: KDNode | None = None, k: int = 2) -> None:
        self.root = root
        self.k = k

    def build(self, points: list[tuple[float, ...]]) -> None:
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

    def insert(self, point: tuple[float, ...]) -> None:
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

    def find_nearest_point(self, point: tuple[float, ...]) -> np.ndarray:
        if self.root is None:
            raise ValueError("Tree is empty")

        target = np.asarray(point, dtype=float)
        best_node, _ = self._find_nearest_node(self.root, target)

        return best_node.point

    def _distance_sq(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.sum((p1 - p2) ** 2))

    def _find_nearest_node(
        self, node: KDNode, point: np.ndarray
    ) -> tuple[KDNode, float]:
        axis = node.axis

        if point[axis] < node.point[axis]:
            next_node, other_node = node.left, node.right
        else:
            next_node, other_node = node.right, node.left

        best_node = node
        best_dist = self._distance_sq(node.point, point)

        if next_node is not None:
            candidate_node, candidate_dist = self._find_nearest_node(next_node, point)
            if candidate_dist < best_dist:
                best_node, best_dist = candidate_node, candidate_dist

        plane_dist = (point[axis] - node.point[axis]) ** 2
        if other_node is not None and plane_dist < best_dist:
            candidate_node, candidate_dist = self._find_nearest_node(other_node, point)
            if candidate_dist < best_dist:
                best_node, best_dist = candidate_node, candidate_dist

        return best_node, best_dist
