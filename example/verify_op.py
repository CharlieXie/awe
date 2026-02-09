"""
verify_optimization.py
======================
Compare the original dp_waypoint_selection (slow) against
dp_waypoint_selection_fast (optimized) on actual RLDS data.

Reports per-episode comparison and confirms results are identical.
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

import tensorflow_datasets as tfds
from waypoint_extraction.extract_waypoints import dp_waypoint_selection     # original (slow)
from waypoint_extraction.extract_waypoints_fast import dp_waypoint_selection_fast  # optimized

# ---- Config ----
SRC_DATA_DIR = r"C:\Users\chuanlia\Documents\learning_space\ntu\projects\awe\example\rlds_data"
ERR_THRESHOLD = 0.008
MAX_EPISODES = 20   # check first N episodes (set to None for all)

# ---- Run comparison ----
if __name__ == "__main__":
    src_builder = tfds.builder_from_directory(SRC_DATA_DIR)
    src_dataset = src_builder.as_dataset(split="train")

    all_match = True
    total_orig_time = 0.0
    total_fast_time = 0.0

    for ep_idx, episode in enumerate(src_dataset):
        if MAX_EPISODES is not None and ep_idx >= MAX_EPISODES:
            break

        steps = list(episode["steps"])
        T = len(steps)
        if T == 0:
            continue

        jp = np.array([
            s["observation"]["joint_position_arm_left"].numpy() for s in steps
        ])

        # ---- Original (slow) ----
        t0 = time.perf_counter()
        wp_orig = dp_waypoint_selection(
            env=None, actions=jp.copy(), gt_states=jp.copy(),
            err_threshold=ERR_THRESHOLD, pos_only=True,
        )
        t_orig = time.perf_counter() - t0

        # ---- Fast (optimized) ----
        t0 = time.perf_counter()
        wp_fast = dp_waypoint_selection_fast(jp.copy(), ERR_THRESHOLD)
        t_fast = time.perf_counter() - t0

        total_orig_time += t_orig
        total_fast_time += t_fast

        # ---- Compare ----
        match = (wp_orig == wp_fast)
        status = "MATCH" if match else "MISMATCH"
        speedup = t_orig / t_fast if t_fast > 0 else float('inf')

        if not match:
            all_match = False
            print(f"\n  *** MISMATCH at Episode {ep_idx} ***")
            print(f"      Original : {wp_orig}")
            print(f"      Fast     : {wp_fast}")
            # Find differences
            set_orig, set_fast = set(wp_orig), set(wp_fast)
            only_orig = sorted(set_orig - set_fast)
            only_fast = sorted(set_fast - set_orig)
            if only_orig:
                print(f"      Only in original: {only_orig}")
            if only_fast:
                print(f"      Only in fast    : {only_fast}")

        print(f"  Episode {ep_idx:>4} | T={T:>4} | orig={len(wp_orig):>3} wp | fast={len(wp_fast):>3} wp | "
              f"orig={t_orig:>8.3f}s | fast={t_fast:>8.3f}s | {speedup:>6.1f}x | {status}")

    # ---- Summary ----
    print(f"\n{'='*70}")
    if all_match:
        print("  ALL EPISODES MATCH — optimization is correct!")
    else:
        print("  WARNING: Some episodes have MISMATCHES!")

    speedup_total = total_orig_time / total_fast_time if total_fast_time > 0 else float('inf')
    print(f"  Total original time : {total_orig_time:.2f}s")
    print(f"  Total fast time     : {total_fast_time:.2f}s")
    print(f"  Overall speedup     : {speedup_total:.1f}x")
    print(f"{'='*70}")