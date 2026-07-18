from __future__ import annotations

import numpy as np

from src.algorithms.kd_tree import KDTree
from src.exceptions import ObstacleCollisionError, TargetNotReachedError


class Node:
    """The node of RRT"""
    def __init__(self, point: np.ndarray | tuple[float, ...],
                 path_to_node: list[tuple[int, ...]]) -> None:
        """
        Args:
            point (np.ndarray | tuple[float, ...]): Node coordinates in space
            path_to_node (list[tuple[int, ...]]): The path to this node,
                starting from the start position
        """
        self.point = np.asarray(point, dtype=np.float64)
        # FIXME (mortiferrr): High memory usage
        self.path_to_node = path_to_node


# TODO (mortiferrr): Add support for diagonal movements
# TODO (mortiferrr): Add the option to specify the step size
# TODO (mortiferrr): Add POMDP support
class DiscreteRRTAgent:
    """
    Class of agent that operate according to
    the RRT algorithm in a discrete space (on a grid)
    NOTE: The optimal path is not guaranteed
    """
    def __init__(
        self,
        k: int = 2,
        random_state: int | None = None,
    ) -> None:
        """
        Args:
            k (int): Dimensionality of space
            random_state (int | None): Random number generator seed
        """
        self._nodes: dict[tuple[float, ...], Node] = {}
        self._k = k
        self._kd_tree = KDTree(k=k)
        self._rng = np.random.default_rng(random_state)

    # FIXME (mortiferrr): Too many allocations associated with the conversion to a tuple
    # mortifer (from Latin) - the bringer of death to RAM. It is my Alter Ego
    def _create_new_node(self, obstacles: np.ndarray,
                         random_point: np.ndarray,
                         nearest_node: Node) -> Node:
        # TODO (mortiferrr): This solution allows movement only along the axes,
        # because `np.argmax` finds and returns only the first max element
        diff = random_point - nearest_node.point
        max_axis = np.argmax(np.abs(diff))
        move = np.zeros_like(nearest_node.point)
        move[max_axis] = np.sign(diff[max_axis])

        try:
            new_point = nearest_node.point + move
            if obstacles[tuple(new_point.astype(np.int32))] == 1:
                raise ObstacleCollisionError

        except IndexError as e:
            raise e

        path_to_new_node = nearest_node.path_to_node.copy()
        path_to_new_node.append(tuple(move.astype(np.int32)))

        new_node = Node(point=new_point, path_to_node=path_to_new_node)
        return new_node

    # FIXME (mortiferrr): Too many allocations associated with the conversion to a tuple
    # mortifer (from Latin) - the bringer of death to RAM. It is my Alter Ego
    def _build(self, obstacles: np.ndarray,
               current: np.ndarray,
               target: np.ndarray,
               max_steps: int):
        assert self._nodes, "The root node must be initialized in `self._nodes` before building"
        assert tuple(current) in self._nodes, "The root node must be initialized in `self._nodes` before building"

        curr_node = self._nodes[tuple(current)]
        curr_step = 0

        while not np.allclose(curr_node.point, target) and curr_step < max_steps:
            curr_step += 1
            # This is necessary to avoid unnecessary memory allocation
            random_point = self._rng.uniform(0, obstacles.shape)
            np.floor(random_point, out=random_point)
            # Keep selecting a random point until one is chosen
            # that isn't already in the dictionary
            if tuple(random_point) in self._nodes:
                continue

            nearest_point = self._kd_tree.find_nearest_point(random_point)
            nearest_node = self._nodes[tuple(nearest_point)]

            try:
                new_node = self._create_new_node(obstacles, random_point, nearest_node)
            except (ObstacleCollisionError, IndexError):
                continue

            self._nodes[tuple(new_node.point)] = new_node
            self._kd_tree.insert(new_node.point)

            curr_node = new_node

        # This might work if the loop exit condition
        # was based on a limit on the number of iterations
        if not np.allclose(curr_node.point, target) and len(self._nodes):
            raise TargetNotReachedError("The target was not reached")

    def act(self, obstacles: np.ndarray,
            target: np.ndarray,
            current: np.ndarray,
            max_steps: int = 1000) -> list:
        """
        Generates a sequence of actions for the agent
        Args:
            obstacles (np.ndarray): A map with obstacles, where a “1”
                represents an obstacle and a “0” represents an empty cell
            target (np.ndarray): Target position for the agent
            current (np.ndarray): Cuurent position for the agent
            max_steps (int): Maximum number of iterations during tree building.
                Default 100

         Raises:
            TargetNotFoundError: Raise if the target point has not been
                reached within the specified maximum number of iterations
            ValueError: Raise if the specified maximum number of iterations
                is less than or equal to zero
        """
        if max_steps < 1:
            raise ValueError("The `max_steps` parameter must be greater than zero")

        root = Node(point=current, path_to_node=[])
        self._nodes[tuple(current)] = root
        self._kd_tree.insert(root.point)

        self._build(obstacles, current, target, max_steps)
        assert self._nodes, "After the tree is built, the `self._nodes` must be not empty"

        return self._nodes[tuple(target)].path_to_node

