import numpy as np
import pytest

from src.algorithms.discrete_rrt import DiscreteRRTAgent
from src.exceptions import TargetNotReachedError


def follow_path(act, start) -> list[int]:
    curr_coords = start
    for move in act:
        for i in range(len(curr_coords)):
            curr_coords[i] += move[i]
    return curr_coords
        

@pytest.mark.parametrize(
    "obstacles,start,target,k",
    [
        ([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [0, 0], [3, 3], 2),

        ([[0, 0, 0, 0]], [0, 0], [0, 3], 2),

        ([[0], [0], [0], [0]], [0, 0], [3, 0], 2),

        ([[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 0, 1], [1, 0, 0, 0]], [0, 2], [3, 3], 2),

        ([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
          [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], [0, 0, 0], [3, 3, 3], 3),

        ([[[0, 0, 1, 1], [1, 0, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1]],
          [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 1]],
          [[1, 1, 0, 0], [0, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 1]],
          [[1, 1, 0, 1], [1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]]], [0, 0, 0], [3, 3, 3], 3),

         ([[[0, 0, 0, 0]]], [0, 0, 0], [0, 0, 3], 3),

         ([[[0]], [[0]], [[0]], [[0]],], [0, 0, 0], [3, 0, 0], 3),
    ],
)
def test_search_target_2d(obstacles, start, target, k):
    obstacles_arr = np.array(obstacles, dtype=np.int32)
    start_arr = np.array(start, dtype=np.int32)
    target_arr = np.array(target, dtype=np.int32)

    agent = DiscreteRRTAgent(k=k, random_state=67)
    act = agent.act(obstacles_arr, target_arr, start_arr)

    act_coords = follow_path(act, start)

    assert act_coords == target


@pytest.mark.parametrize(
    "obstacles,start,target,k",
    [
        ([[0, 0, 0], [1, 1, 1], [0, 0, 0]], (0, 0), (3, 3), 2),

        ([[[0, 0, 0], [1, 0, 1], [1, 0, 1]],
          [[0, 0, 0], [1, 1, 1], [0, 1, 0]]], (0, 0, 0), (1, 2, 2), 3)
    ]
)
def test_target_not_found_error(obstacles, start, target, k):
    obstacles_arr = np.array(obstacles, dtype=np.int32)
    start_arr = np.array(start, dtype=np.int32)
    target_arr = np.array(target, dtype=np.int32)

    agent = DiscreteRRTAgent(k=k, random_state=67)
    with pytest.raises(TargetNotReachedError, match="The target was not reached"):
        act = agent.act(obstacles_arr, target_arr, start_arr, 100)
