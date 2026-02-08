"""
Optimized DP waypoint selection (pos_only mode).

- Precomputed error matrix: O(1) lookup in DP loop (方案1)
- Vectorized point-line distance: numpy batch ops (方案2)
- Only depends on numpy — safe for multiprocessing workers on Windows
"""
import numpy as np


def _precompute_err_matrix(actions, gt_states):
    """
    Precompute max point-to-line-segment distance for all (start, end) pairs.

    err_matrix[k, i] = max distance of gt_states[k:i] from
                        line segment (actions[k] -> actions[i])

    Replaces repeated calls to pos_only_geometric_waypoint_trajectory()
    in the DP inner loop with O(1) table lookups.
    """
    n = len(actions)
    err_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        line_start = actions[i]
        for j in range(i + 2, n):
            line_end = actions[j]

            # All points in the segment [i, j)  — vectorized
            segment_points = gt_states[i:j]          # shape (j-i, D)
            line_vec = line_end - line_start          # shape (D,)
            denom = np.dot(line_vec, line_vec)

            if denom < 1e-30:
                # Degenerate: start ≈ end
                distances = np.linalg.norm(segment_points - line_start, axis=1)
            else:
                point_vecs = segment_points - line_start          # (m, D)
                t = point_vecs @ line_vec / denom                 # (m,)
                np.clip(t, 0, 1, out=t)
                projections = line_start + t[:, np.newaxis] * line_vec  # (m, D)
                distances = np.linalg.norm(segment_points - projections, axis=1)

            err_matrix[i, j] = distances.max()

    return err_matrix


def dp_waypoint_selection_fast(actions, err_threshold):
    """
    Optimized DP waypoint selection for pos_only=True mode.

    Mathematically equivalent to the original dp_waypoint_selection()
    when called with pos_only=True and actions == gt_states.

    Parameters
    ----------
    actions : np.ndarray, shape (T, D)
        Joint position trajectory (used as both actions and gt_states).
    err_threshold : float
        Maximum allowed geometric error.

    Returns
    -------
    list[int]
        Sorted waypoint indices.
    """
    actions = np.asarray(actions, dtype=np.float64)
    num_frames = len(actions)

    if num_frames <= 1:
        return list(range(num_frames))

    # pos_only=True → only last frame is forced waypoint
    initial_waypoints = [num_frames - 1]

    # ---- Step 1: Precompute error matrix (vectorized) ----
    err_matrix = _precompute_err_matrix(actions, actions)

    # ---- Step 2: Threshold sanity check ----
    # With gt_states == actions, using ALL points as waypoints gives error = 0.
    # So err_threshold <= 0 is the only pathological case.
    if err_threshold <= 0:
        print("Error threshold <= 0, returning all points as waypoints.")
        return list(range(1, num_frames))

    # ---- Step 3: DP with O(1) lookups ----
    memo = {}
    for i in range(num_frames):
        memo[i] = (0, [])
    memo[1] = (1, [1])

    for i in range(1, num_frames):
        min_wp_count = float("inf")
        best_wp = []

        for k in range(1, i):
            total_traj_err = err_matrix[k, i]        # O(1) lookup!

            if total_traj_err < err_threshold:
                sub_count, sub_wp = memo[k - 1]
                total_count = 1 + sub_count

                if total_count < min_wp_count:
                    min_wp_count = total_count
                    best_wp = sub_wp + [i]

        memo[i] = (min_wp_count, best_wp)

    _, waypoints = memo[num_frames - 1]
    waypoints += initial_waypoints
    waypoints = sorted(set(waypoints))

    print(f"Minimum number of waypoints: {len(waypoints)}")
    print(f"waypoint positions: {waypoints}")
    return waypoints


# ---- Multiprocessing worker (top-level, picklable) ----

def _extract_worker(args):
    """
    Worker function for ProcessPoolExecutor.

    Defined at module top-level so it's picklable on Windows (spawn).
    Only uses numpy — no TensorFlow, no mocked modules.
    """
    joint_positions, err_threshold = args
    return dp_waypoint_selection_fast(joint_positions, err_threshold)