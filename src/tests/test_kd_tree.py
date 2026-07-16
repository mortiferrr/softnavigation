import numpy as np
import pytest
import time
from unittest.mock import patch

from src.algorithms.kd_tree import KDTree


def test_build_single_point():
    tree = KDTree()
    tree.build([(1.0, 2.0)])
    assert tree.root is not None
    assert np.allclose(tree.root.point, np.array([1.0, 2.0]))
    assert tree.root.left is None
    assert tree.root.right is None


def test_build_known_2d_structure():
    # Points: (3, 1), (2, 3), (2, 6), (4, 2)
    # k=2, so depth 0 splits by x.
    # Sorted by x: (2, 3), (2, 6), (3, 1), (4, 2)
    # Median index: 4 // 2 = 2. So median is (3, 1).
    # Left: (2, 3), (2, 6)
    # Right: (4, 2)

    # Left child (depth 1, split by y):
    # Sorted by y: (2, 3), (2, 6).
    # Median index: 2 // 2 = 1. So median is (2, 6).
    # Left-left: (2, 3)
    # Left-right: None

    tree = KDTree(k=2)
    tree.build([(3, 1), (2, 3), (2, 6), (4, 2)])

    root = tree.root
    assert root is not None
    assert np.allclose(root.point, np.array([3.0, 1.0]))
    assert root.axis == 0

    assert root.right is not None
    assert np.allclose(root.right.point, np.array([4.0, 2.0]))

    assert root.left is not None
    assert np.allclose(root.left.point, np.array([2.0, 6.0]))

    assert root.left.left is not None
    assert np.allclose(root.left.left.point, np.array([2.0, 3.0]))


def test_insert_into_empty():
    tree = KDTree()
    tree.insert((5.0, 5.0))
    assert tree.root is not None
    assert np.allclose(tree.root.point, np.array([5.0, 5.0]))


def test_insert_existing_tree():
    tree = KDTree(k=2)
    tree.insert((5.0, 5.0))  # root, axis 0
    tree.insert((3.0, 8.0))  # left of root, axis 1
    tree.insert((7.0, 2.0))  # right of root, axis 1

    assert np.allclose(tree.root.point, np.array([5.0, 5.0]))
    assert np.allclose(tree.root.left.point, np.array([3.0, 8.0]))
    assert np.allclose(tree.root.right.point, np.array([7.0, 2.0]))


def test_find_nearest_empty():
    tree = KDTree()
    with pytest.raises(ValueError, match="Tree is empty"):
        tree.find_nearest_point((1.0, 2.0))


def test_build_empty():
    tree = KDTree()
    with pytest.raises(ValueError, match="Points list is empty"):
        tree.build([])


def test_find_nearest_exact_match():
    tree = KDTree()
    tree.build([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])
    nearest = tree.find_nearest_point((3.0, 4.0))
    assert np.allclose(nearest, np.array([3.0, 4.0]))


def test_find_nearest_backtracking():
    # Construct a specific tree where greedy search goes to the wrong branch.
    # Points: (2, 5), (5, 5), (6, 6), (6.1, 7)
    # Root will be (6, 6), axis 0.
    # Left child: (5, 5), axis 1.
    # Right child: (6.1, 7), axis 1.
    # Query: (5, 7).
    # Greedy goes to left child (5, 5) because 5 < 6.
    # Distance to (5, 5) is 4.
    # Distance to Root (6, 6) is 2. Best is now Root.
    # Distance to splitting plane (x=6) is 1. Since 1 < 2, it must check right branch.
    # Distance to right child (6.1, 7) is 1.21. This is the true nearest neighbor!

    tree = KDTree(k=2)
    tree.build([(2, 5), (5, 5), (6, 6), (6.1, 7)])

    nearest = tree.find_nearest_point((5, 7))
    assert np.allclose(nearest, np.array([6.1, 7.0]))


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
    np.random.seed(42 + k)
    num_points = 100
    num_queries = 20

    points = np.random.rand(num_points, k) * 100.0
    tree = KDTree(k=k, dist=dist)
    points_list = [tuple(p) for p in points]
    tree.build(points_list)

    queries = np.random.rand(num_queries, k) * 100.0

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
    assert np.allclose(nearest, np.array([2.0, 2.0]))


def test_duplicate_points():
    # Duplicates during build
    tree_build = KDTree(k=2)
    tree_build.build([(1.0, 1.0), (1.0, 1.0), (3.0, 3.0)])
    nearest = tree_build.find_nearest_point((1.1, 1.1))
    assert np.allclose(nearest, np.array([1.0, 1.0]))

    # Duplicates during insertion
    tree_insert = KDTree(k=2)
    tree_insert.insert((1.0, 1.0))
    tree_insert.insert((1.0, 1.0))
    tree_insert.insert((3.0, 3.0))
    nearest_insert = tree_insert.find_nearest_point((1.1, 1.1))
    assert np.allclose(nearest_insert, np.array([1.0, 1.0]))


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
    assert np.allclose(nearest, np.array([expected]))


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
    assert np.allclose(nearest, np.array(expected))


def test_nearest_differs_by_metric():
    # Points: P1 = (5.0, 0.0), P2 = (4.0, 2.9)
    # Query: Q = (0.0, 0.0)
    # Under Euclidean distance:
    # d_sq(Q, P1) = 25.0
    # d_sq(Q, P2) = 16.0 + 8.41 = 24.41 -> P2 is closer.
    # Under Manhattan distance:
    # d_man(Q, P1) = 5.0
    # d_man(Q, P2) = 4.0 + 2.9 = 6.9 -> P1 is closer.
    points = [(5.0, 0.0), (4.0, 2.9)]
    query = (0.0, 0.0)

    tree_euclidian = KDTree(k=2, dist="euclidean")
    tree_euclidian.build(points)
    nearest_euclidian = tree_euclidian.find_nearest_point(query)
    assert np.allclose(nearest_euclidian, np.array([4.0, 2.9]))

    tree_manhattan = KDTree(k=2, dist="manhattan")
    tree_manhattan.build(points)
    nearest_manhattan = tree_manhattan.find_nearest_point(query)
    assert np.allclose(nearest_manhattan, np.array([5.0, 0.0]))


def test_build_worst_case_o_n_log_n():
    N1 = 2000
    N2 = 4000

    pts1 = np.array([(float(i), float(i)) for i in range(N1)])
    pts2 = np.array([(float(i), float(i)) for i in range(N2)])

    tree1 = KDTree(k=2)
    tree2 = KDTree(k=2)

    t0 = time.perf_counter()
    tree1.build(pts1)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    tree2.build(pts2)
    t3 = time.perf_counter()

    time1 = max(t1 - t0, 1e-5)
    time2 = max(t3 - t2, 1e-5)

    # We expect time2 / time1 to be strictly less than 4.0 (which would be O(N^2))
    # It should ideally be around 2.2, but we give it a generous bound of 3.5 to avoid flakiness in CI.
    assert time2 / time1 < 3.5


def test_search_worst_case_o_n_log_n():
    N1 = 1000
    N2 = 2000

    # Worst case search can happen with highly overlapping bounds, but we test
    # the average worst-case with random points where we expect O(log N) per query.
    np.random.seed(42)
    pts1 = np.random.rand(N1, 2)
    pts2 = np.random.rand(N2, 2)

    tree1 = KDTree(k=2)
    tree1.build(pts1)

    tree2 = KDTree(k=2)
    tree2.build(pts2)

    queries = np.random.rand(100, 2)

    with patch.object(tree1, "_dist", wraps=tree1._dist) as mock_dist1:
        for q in queries:
            tree1.find_nearest_point(q)
        calls1 = max(mock_dist1.call_count, 1)

    with patch.object(tree2, "_dist", wraps=tree2._dist) as mock_dist2:
        for q in queries:
            tree2.find_nearest_point(q)
        calls2 = mock_dist2.call_count

    # The number of distance calculations for N2 should be roughly:
    # calls1 * (log(N2) / log(N1)) = calls1 * (11 / 10) = 1.1 * calls1
    # We assert that the number of distance checks grows sublinearly compared to the tree size.
    ratio = calls2 / calls1
    assert ratio < 1.5  # Much smaller than 2.0 (which would mean O(N) search)
