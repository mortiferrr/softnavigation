from __future__ import annotations

import numpy as np

from src.algorithms.kd_tree import KDTree
from src.exceptions import ObstacleCollisionError, TargetNotReachedError


class Node:
    """The node of RRT"""

    __slots__ = ("point", "parent", "move")

    def __init__(
        self,
        point: tuple[int, ...],
        move: tuple[int, ...] | None = None,
        parent: Node | None = None,
    ) -> None:
        """
        Args:
            point (tuple[int, ...]): Node coordinates in space.
            parent (Node | None): Parent node for this node.
            move (tuple[int, ...] | None): The move that led from the parent node to this one.
        """
        self.point = point
        self.parent = parent
        self.move = move

    def restore_path(self) -> list[tuple[int, ...]]:
        """Restores the path from the root node to the current node"""
        if not self.move:
            raise ValueError("Path to root node cannot be restores")

        moves = [self.move]
        curr_node = self.parent
        while curr_node and curr_node.move:
            moves.append(curr_node.move)
            curr_node = curr_node.parent

        return moves


# TODO (mortiferrr): Add support for diagonal movements
# TODO (mortiferrr): Add the option to specify the step size
# TODO (mortiferrr): Add POMDP support
# TODO (mortiferrr): Rewrite to C++
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
        self._nodes: list[Node] = []
        self._k = k
        self._kd_tree = KDTree(k=k)
        self._rng = np.random.default_rng(random_state)
        # This is true if the new observation passed to the act() method
        # does not require a path change
        # NOTE: This will come in handy for future POMDPs
        self._is_actual = False
        self._actual_path_to_target: list[tuple[int, ...]] = []

    def _create_new_node(
        self, obstacles: np.ndarray, random_point: tuple[int, ...], nearest_node: Node
    ) -> Node:
        # TODO (mortiferrr): This solution allows movement only along the axes,
        # because `np.argmax` finds and returns only the first max element
        diff = np.array(random_point, dtype=np.int32) - nearest_node.point
        max_axis = np.argmax(np.abs(diff))
        move = np.zeros_like(nearest_node.point)
        move[max_axis] = np.sign(diff[max_axis])

        try:
            new_point = nearest_node.point + move
            new_point = tuple(new_point.tolist())
            if obstacles[new_point] == 1:
                raise ObstacleCollisionError

        except IndexError as e:
            raise e

        new_node = Node(point=new_point, move=tuple(move.tolist()), parent=nearest_node)
        return new_node

    def _build(
        self,
        obstacles: np.ndarray,
        target: tuple[int, ...],
        max_steps: int,
    ) -> Node:
        assert self._nodes, (
            "The root node must be initialized in `self._nodes` before building"
        )
        assert len(self._nodes), "The 'self._nodes' should contain only the root node"

        new_node = self._nodes[0]
        curr_step = 0

        while not np.allclose(new_node.point, target) and curr_step < max_steps:
            curr_step += 1
            random_point = self._rng.integers(low=obstacles.shape)
            random_point = tuple(random_point.tolist())
            # Keep selecting a random point until one is chosen
            # that isn't already in the dictionary
            if random_point in self._kd_tree.points:
                continue

            nearest_id = self._kd_tree.find_nearest_point(random_point, return_id=True)
            nearest_node = self._nodes[nearest_id]  # type: ignore

            try:
                new_node = self._create_new_node(obstacles, random_point, nearest_node)
            except (ObstacleCollisionError, IndexError):
                continue

            self._nodes.append(new_node)
            self._kd_tree.insert(new_node.point, len(self._nodes) - 1)

        # This might work if the loop exit condition
        # was based on a limit on the number of iterations
        if not np.allclose(new_node.point, target) and len(self._nodes):
            raise TargetNotReachedError("The target was not reached")

        return new_node

    def act(
        self,
        obstacles: np.ndarray,
        target: tuple[int, ...],
        current: tuple[int, ...],
        max_steps: int = 1000,
    ) -> tuple[int, ...] | None:
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

        # NOTE: A path validation check for GridMemory will be added here later;
        # this is required for POMDP and agent mode (returns a single action rather than a list).
        if not self._is_actual:
            self._kd_tree.clear()
            root = Node(point=current)
            self._nodes.append(root)
            self._kd_tree.insert(root.point, 0)

            target_node = self._build(obstacles, target, max_steps)
            assert self._nodes, (
                "After the tree is built, the `self._nodes` must be not empty"
            )
            self._actual_path_to_target = target_node.restore_path()
            self._is_actual = True

        move = self._actual_path_to_target.pop(0) if len(self._actual_path_to_target) != 0 else None
        return move
