from typing import cast
import numpy as np
import pytest
from unittest.mock import patch

from src.algorithms.kd_tree import KDTree


rng = np.random.default_rng(seed=67)


def test_build_single_point():
    tree = KDTree()
    tree.build([(1.0, 2.0)])
    assert tree.root is not None
    assert np.allclose(tree.root.point, [1.0, 2.0])
    assert tree.root.left is None
    assert tree.root.right is None


def test_clear_tree():
    tree = KDTree(k=2)
    tree.build([(3, 1), (2, 3), (2, 6), (4, 2)])
    tree.clear()
    assert tree.root is None
    assert len(tree.points) == 0


def test_build_2d_structure():
    tree = KDTree(k=2)
    tree.build([(3, 1), (2, 3), (2, 6), (4, 2)])

    root = tree.root
    assert root is not None
    assert np.allclose(root.point, [3.0, 1.0])
    assert root.axis == 0

    assert root.left is not None
    assert np.allclose(root.left.point, [2.0, 6.0])

    assert root.right is not None
    assert np.allclose(root.right.point, [4.0, 2.0])

    assert root.left.left is not None
    assert np.allclose(root.left.left.point, [2.0, 3.0])


def test_insert_into_empty():
    tree = KDTree()
    tree.insert((5.0, 5.0))
    assert tree.root is not None
    assert np.allclose(tree.root.point, [5.0, 5.0])


def test_insert_existing_tree():
    tree = KDTree(k=2)
    tree.insert((5.0, 5.0))  # root, axis 0
    tree.insert((3.0, 8.0))  # left of root, axis 1
    tree.insert((7.0, 2.0))  # right of root, axis 1

    assert np.allclose(tree.root.point, [5.0, 5.0])  # type: ignore
    assert np.allclose(tree.root.left.point, [3.0, 8.0])  # type: ignore
    assert np.allclose(tree.root.right.point, [7.0, 2.0])  # type: ignore


def test_find_nearest_if_empty():
    tree = KDTree()
    with pytest.raises(ValueError, match="Tree is empty"):
        tree.find_nearest_point((1.0, 2.0))


def test_build_if_points_empty():
    tree = KDTree()
    with pytest.raises(ValueError, match="Points list is empty"):
        tree.build([])


def test_build_if_points_and_ids_len_not_be_same():
    tree = KDTree()
    with pytest.raises(ValueError, match="Lengths of `ids` and `points` must be same"):
        tree.build([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)], [1, 2])


def test_find_nearest_exact_match():
    tree = KDTree()
    tree.build([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])
    nearest = tree.find_nearest_point((3.0, 4.0))
    assert np.allclose(nearest, [3.0, 4.0])


def test_find_nearest_backtracking():
    tree = KDTree(k=2)
    tree.build([(2, 5), (5, 5), (6, 6), (6.1, 7)])

    nearest = tree.find_nearest_point((5, 7))
    assert np.allclose(nearest, [6.1, 7.0])


def brute_force_nearest(
    points: np.ndarray, target: np.ndarray, dist: str = "euclidean"
) -> np.ndarray:
    if dist == "euclidean":
        distances = np.sum((points - target) ** 2, axis=1)
    elif dist == "manhattan":
        distances = np.sum(np.abs(points - target), axis=1)
    else:
        raise ValueError(f"Unknown distance metric: {dist}")
    best_idx = np.argmin(distances)
    return points[best_idx]


@pytest.mark.parametrize("k", [2, 3, 5])
@pytest.mark.parametrize("dist", ["euclidean", "manhattan"])
def test_random_points_brute_force_comparison(k, dist):
    points = rng.random(size=(100, k)) * 100.0
    queries = rng.random(size=(20, k)) * 100.0

    tree = KDTree(k=k, dist=dist)
    points_list = [tuple(p) for p in points]
    tree.build(points_list)

    for query in queries:
        tree_nearest = tree.find_nearest_point(tuple(query))
        brute_nearest = brute_force_nearest(points, query, dist=dist)
        assert np.allclose(tree_nearest, brute_nearest)


def test_nearest_on_inserted_tree():
    # Tree degenerated into a list due to sequential diagonal insertion
    tree = KDTree(k=2)
    points = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
    for p in points:
        tree.insert(p)

    # Search should work correctly even on an unbalanced structure
    nearest = tree.find_nearest_point((2.1, 2.1))
    assert np.allclose(nearest, [2.0, 2.0])


def test_duplicate_points():
    # Duplicates during build
    tree_build = KDTree(k=2)
    tree_build.build([(1.0, 1.0), (1.0, 1.0), (3.0, 3.0)])
    nearest = tree_build.find_nearest_point((1.1, 1.1))
    assert np.allclose(nearest, [1.0, 1.0])

    # Duplicates during insertion
    tree_insert = KDTree(k=2)
    tree_insert.insert((1.0, 1.0))
    tree_insert.insert((1.0, 1.0))
    tree_insert.insert((3.0, 3.0))
    nearest_insert = tree_insert.find_nearest_point((1.1, 1.1))
    assert np.allclose(nearest_insert, [1.0, 1.0])


@pytest.mark.parametrize(
    "query,expected",
    [
        (3.2, 3.0),
        (7.5, 8.0),
    ],
)
def test_1d_tree(query, expected):
    tree = KDTree(k=1)
    tree.build([(5.0,), (3.0,), (8.0,), (1.0,), (4.0,)])

    nearest = tree.find_nearest_point((query,))
    assert np.allclose(nearest, expected)


@pytest.mark.parametrize(
    "points,query,expected",
    [
        # Negative coordinates
        ([(-10.0, -10.0), (-5.0, -5.0), (0.0, 0.0)], (-7.0, -7.0), (-5.0, -5.0)),
        # Large scale difference
        ([(1e10, 1e10), (-1e10, -1e10)], (1e9, 1e9), (1e10, 1e10)),
        # Very close values
        ([(1.0000001, 1.0), (1.0000002, 1.0)], (1.0, 1.0), (1.0000001, 1.0)),
    ],
)
def test_extreme_coordinates(points, query, expected):
    tree = KDTree(k=2)
    tree.build(points)
    nearest = tree.find_nearest_point(query)
    assert np.allclose(nearest, expected)


@pytest.mark.parametrize(
    "points,query,expected,dist",
    [
        ([(5.0, 0.0), (4.0, 2.9)], (0.0, 0.0), (4.0, 2.9), "euclidean"),
        ([(5.0, 0.0), (4.0, 2.9)], (0.0, 0.0), (5.0, 0.0), "manhattan"),
    ],
)
def test_nearest_differs_by_metric(points, query, expected, dist):
    tree = KDTree(k=2, dist=dist)
    tree.build(points)
    nearest = tree.find_nearest_point(query)
    assert np.allclose(nearest, expected)


def test_search_o_log_n():
    pts1 = [(float(i), float(i)) for i in range(2000)]
    pts2 = [(float(i), float(i)) for i in range(4000)]

    tree1 = KDTree(k=2)
    tree1.build(pts1)

    tree2 = KDTree(k=2)
    tree2.build(pts2)

    queries = rng.random(size=(100, 2))

    with patch.object(tree1, "_dist", wraps=tree1._dist) as mock_dist1:
        for q in queries:
            tree1.find_nearest_point(tuple(q.tolist()))
        calls1 = max(mock_dist1.call_count, 1)

    with patch.object(tree2, "_dist", wraps=tree2._dist) as mock_dist2:
        for q in queries:
            tree2.find_nearest_point(tuple(q.tolist()))
        calls2 = mock_dist2.call_count

    ratio = calls2 / calls1
    assert ratio < 1.5


@pytest.mark.parametrize("k", [2, 3, 5])
def test_build_tree_with_ids(k):
    ids = rng.integers(low=1000, size=(100,)).tolist()
    points = rng.random(size=(100, k)).tolist()

    tree = KDTree(k=k)
    tree.build(points, ids)

    for p in points:
        nearest_id = tree.find_nearest_point(p, return_id=True)
        p_id = points.index(p)
        assert nearest_id == p_id
