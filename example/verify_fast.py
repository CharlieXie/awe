"""
verify_fast.py
==============
Verify that dp_waypoint_selection_fast produces identical results
to the original dp_waypoint_selection.
"""
import sys, os, types
import numpy as np
import time

# ---- Mock setup (same as rlds_wp_extract.py) ----
for _mod_name in list(sys.modules.keys()):
    if 'mujoco_py' in _mod_name:
        del sys.modules[_mod_name]

class _MockModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        mock_cls = type(name, (), {
            '__init__': lambda self, *a, **kw: None,
            '__call__': lambda self, *a, **kw: None,
            '__getattr__': lambda self, n: None,
        })
        setattr(self, name, mock_cls)
        return mock_cls

for _mod_name in [
    'mujoco_py', 'mujoco_py.builder', 'mujoco_py.utils',
    'mujoco_py.generated', 'mujoco_py.generated.const', 'mujoco_py.cymj',
]:
    sys.modules[_mod_name] = _MockModule(_mod_name)

try:
    import glfw
except (ImportError, ModuleNotFoundError):
    sys.modules['glfw'] = _MockModule('glfw')

awe_root = r"C:\Users\chuanlia\Documents\learning_space\ntu\projects\awe"
sys.path.insert(0, awe_root)

# ---- Import both versions ----
from waypoint_extraction.extract_waypoints import dp_waypoint_selection
from waypoint_extraction.extract_waypoints_fast import (
    dp_waypoint_selection_fast,
    _precompute_err_matrix,
)
from waypoint_extraction.traj_reconstruction import pos_only_geometric_waypoint_trajectory


def verify_err_matrix(actions, verbose=True):
    """
    层级1: 逐个验证 err_matrix[k,i] 是否与原始函数计算结果一致。
    """
    n = len(actions)
    err_matrix = _precompute_err_matrix(actions, actions)

    max_diff = 0.0
    num_checked = 0

    for i in range(1, n):
        for k in range(1, i):
            # ---- 原始函数的计算方式 ----
            # waypoints = [i-k] (relative to subsequence)
            original_err = pos_only_geometric_waypoint_trajectory(
                actions=actions[k : i + 1],
                gt_states=actions[k : i + 1],
                waypoints=[i - k],
            )

            # ---- 预计算矩阵的值 ----
            fast_err = err_matrix[k, i]

            diff = abs(original_err - fast_err)
            max_diff = max(max_diff, diff)
            num_checked += 1

            if diff > 1e-10 and verbose:
                print(f"  MISMATCH at k={k}, i={i}: "
                      f"original={original_err:.15e}, fast={fast_err:.15e}, "
                      f"diff={diff:.2e}")

    return max_diff, num_checked


def verify_waypoints(actions, err_threshold):
    """
    层级2: 验证两个函数输出的 waypoint 列表是否完全一致。
    """
    # ---- 原始版本 ----
    t0 = time.perf_counter()
    wp_original = dp_waypoint_selection(
        env=None,
        actions=actions,
        gt_states=actions,
        err_threshold=err_threshold,
        pos_only=True,
    )
    time_original = time.perf_counter() - t0

    # ---- 优化版本 ----
    t0 = time.perf_counter()
    wp_fast = dp_waypoint_selection_fast(actions, err_threshold)
    time_fast = time.perf_counter() - t0

    return wp_original, wp_fast, time_original, time_fast


# ============================================================
# Main: run on real data or synthetic data
# ============================================================
if __name__ == "__main__":
    import tensorflow_datasets as tfds

    SRC_DATA_DIR = r"C:\Users\chuanlia\Documents\learning_space\ntu\projects\awe\example\rlds_data"
    ERR_THRESHOLD = 0.008

    src_builder = tfds.builder_from_directory(SRC_DATA_DIR)
    src_dataset = src_builder.as_dataset(split="train")

    all_pass = True

    for ep_idx, episode in enumerate(src_dataset):
        steps = list(episode["steps"])
        T = len(steps)
        if T == 0:
            continue

        joint_positions = np.array([
            s["observation"]["joint_position_arm_left"].numpy()
            for s in steps
        ])

        print(f"\n{'='*60}")
        print(f"Episode {ep_idx}: {T} frames, {joint_positions.shape[1]}D")
        print(f"{'='*60}")

        # ---- 层级1: 误差矩阵验证 ----
        print("\n[Level 1] Verifying error matrix...")
        max_diff, num_checked = verify_err_matrix(joint_positions, verbose=True)
        print(f"  Checked {num_checked} (k,i) pairs")
        print(f"  Max absolute difference: {max_diff:.2e}")
        if max_diff < 1e-10:
            print(f"  ✓ PASS — error matrix is identical")
        else:
            print(f"  ✗ FAIL — differences detected!")
            all_pass = False

        # ---- 层级2: Waypoint 结果验证 ----
        print(f"\n[Level 2] Verifying waypoint results (threshold={ERR_THRESHOLD})...")
        wp_orig, wp_fast, t_orig, t_fast = verify_waypoints(
            joint_positions, ERR_THRESHOLD
        )

        match = (wp_orig == wp_fast)
        speedup = t_orig / t_fast if t_fast > 0 else float('inf')

        print(f"  Original:  {len(wp_orig)} waypoints, {t_orig:.3f}s")
        print(f"  Fast:      {len(wp_fast)} waypoints, {t_fast:.3f}s")
        print(f"  Speedup:   {speedup:.1f}x")

        if match:
            print(f"  ✓ PASS — waypoint indices are identical")
        else:
            print(f"  ✗ FAIL — waypoint indices differ!")
            print(f"    Original: {wp_orig}")
            print(f"    Fast:     {wp_fast}")
            # 找出差异
            set_orig = set(wp_orig)
            set_fast = set(wp_fast)
            only_orig = sorted(set_orig - set_fast)
            only_fast = sorted(set_fast - set_orig)
            if only_orig:
                print(f"    Only in original: {only_orig}")
            if only_fast:
                print(f"    Only in fast:     {only_fast}")
            all_pass = False

    # ---- 总结 ----
    print(f"\n{'='*60}")
    if all_pass:
        print("ALL EPISODES PASSED — results are identical.")
    else:
        print("SOME EPISODES FAILED — please investigate.")
    print(f"{'='*60}")