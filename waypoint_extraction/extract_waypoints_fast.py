"""
Optimized DP waypoint selection (pos_only mode).

- Numba JIT compiled precompute + DP (方案3)
- Precomputed error matrix: O(1) lookup in DP loop (方案1)
- Vectorized point-line distance (方案2)
- Only depends on numpy + numba — safe for multiprocessing on Windows
"""
import numpy as np
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ================================================================
# Numba-accelerated core (方案3)
# ================================================================
if HAS_NUMBA:
    @numba.njit(cache=True)
    def _precompute_err_matrix_numba(actions):
        """Numba JIT: precompute max point-line distance for all (start, end) pairs."""
        n = actions.shape[0]
        D = actions.shape[1]
        err_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 2, n):
                # line vector
                lv = actions[j] - actions[i]
                denom = 0.0
                for d in range(D):
                    denom += lv[d] * lv[d]

                max_dist = 0.0
                for p in range(i, j):  # points i .. j-1
                    dot_pv_lv = 0.0
                    for d in range(D):
                        dot_pv_lv += (actions[p, d] - actions[i, d]) * lv[d]

                    if denom < 1e-30:
                        t = 0.0
                    else:
                        t = dot_pv_lv / denom
                        if t < 0.0:
                            t = 0.0
                        elif t > 1.0:
                            t = 1.0

                    dist_sq = 0.0
                    for d in range(D):
                        proj_d = actions[i, d] + t * lv[d]
                        diff = actions[p, d] - proj_d
                        dist_sq += diff * diff
                    dist = np.sqrt(dist_sq)

                    if dist > max_dist:
                        max_dist = dist

                err_matrix[i, j] = max_dist

        return err_matrix

    @numba.njit(cache=True)
    def _dp_core_numba(err_matrix, num_frames, err_threshold):
        """Numba JIT: DP core with O(1) lookups + early termination."""
        INF = 1e18
        memo_count = np.full(num_frames, INF)
        memo_from = np.full(num_frames, -1, dtype=np.int64)

        memo_count[0] = 0.0
        if num_frames > 1:
            memo_count[1] = 1.0
            memo_from[1] = 1

        for i in range(2, num_frames):
            for k in range(1, i):
                if err_matrix[k, i] < err_threshold:
                    candidate = 1.0 + memo_count[k - 1]
                    if candidate < memo_count[i]:
                        memo_count[i] = candidate
                        memo_from[i] = k
                        if memo_count[i] == 1.0:
                            break  # can't do better than 1 waypoint

        return memo_count, memo_from


# ================================================================
# Numpy fallback (no numba)
# ================================================================
def _precompute_err_matrix_numpy(actions):
    """Numpy vectorized fallback."""
    n = len(actions)
    err_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        line_start = actions[i]
        for j in range(i + 2, n):
            line_end = actions[j]
            segment_points = actions[i:j]
            line_vec = line_end - line_start
            denom = np.dot(line_vec, line_vec)

            if denom < 1e-30:
                distances = np.linalg.norm(segment_points - line_start, axis=1)
            else:
                point_vecs = segment_points - line_start
                t = point_vecs @ line_vec / denom
                np.clip(t, 0, 1, out=t)
                projections = line_start + t[:, np.newaxis] * line_vec
                distances = np.linalg.norm(segment_points - projections, axis=1)

            err_matrix[i, j] = distances.max()

    return err_matrix


def _dp_core_python(err_matrix, num_frames, err_threshold):
    """Pure Python DP fallback."""
    INF = float("inf")
    memo_count = [INF] * num_frames
    memo_from = [-1] * num_frames

    memo_count[0] = 0
    if num_frames > 1:
        memo_count[1] = 1
        memo_from[1] = 1

    for i in range(2, num_frames):
        for k in range(1, i):
            if err_matrix[k, i] < err_threshold:
                candidate = 1 + memo_count[k - 1]
                if candidate < memo_count[i]:
                    memo_count[i] = candidate
                    memo_from[i] = k
                    if memo_count[i] == 1:
                        break

    return memo_count, memo_from


# ================================================================
# Unified interface
# ================================================================
def _reconstruct_waypoints(memo_from, num_frames):
    """Backtrack memo_from[] to recover waypoint list."""
    waypoints = []
    i = num_frames - 1
    while i > 0 and memo_from[i] >= 0:
        waypoints.append(i)
        k = memo_from[i]
        i = k - 1
    waypoints.reverse()
    return waypoints


def dp_waypoint_selection_fast(actions, err_threshold):
    """
    Optimized DP waypoint selection (pos_only=True, actions == gt_states).

    Uses Numba JIT if available, else falls back to numpy vectorized version.
    """
    actions = np.asarray(actions, dtype=np.float64)
    num_frames = len(actions)

    if num_frames <= 1:
        return list(range(num_frames))

    initial_waypoints = [num_frames - 1]

    if err_threshold <= 0:
        return list(range(1, num_frames))

    # --- Precompute error matrix ---
    if HAS_NUMBA:
        err_matrix = _precompute_err_matrix_numba(actions)
    else:
        err_matrix = _precompute_err_matrix_numpy(actions)

    # --- DP ---
    if HAS_NUMBA:
        memo_count, memo_from = _dp_core_numba(err_matrix, num_frames, err_threshold)
    else:
        memo_count, memo_from = _dp_core_python(err_matrix, num_frames, err_threshold)

    # --- Reconstruct ---
    if HAS_NUMBA:
        memo_from_list = memo_from.tolist()
    else:
        memo_from_list = memo_from

    waypoints = _reconstruct_waypoints(memo_from_list, num_frames)
    waypoints += initial_waypoints
    waypoints = sorted(set(waypoints))

    print(f"Minimum number of waypoints: {len(waypoints)}")
    print(f"waypoint positions: {waypoints}")
    return waypoints


# ---- Multiprocessing worker ----
def _extract_worker(args):
    joint_positions, err_threshold = args
    return dp_waypoint_selection_fast(joint_positions, err_threshold)