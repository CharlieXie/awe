"""
rlds_wp_extract.py
==================
Extract waypoints from RLDS dataset using AWE dp_waypoint_selection
on joint_position_arm_left + joint_position_arm_right (6D joint space each),
with gripper state change detection, err_threshold=0.008.

All other features at waypoint indices are preserved.
Output is a valid RLDS dataset loadable with tfds.builder_from_directory().

Usage:
    python rlds_wp_extract.py
"""

import sys
import os
import types
import shutil
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import json

# 用于收集每个 episode 的统计信息
WAYPOINT_MAP = []  # 保存每个 episode 的 waypoint 原始索引
# ============================================================
# 0. Mock mujoco_py / glfw (same as rlds_wp_traj.py)
# ============================================================
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

awe_root = r".."
sys.path.insert(0, awe_root)

from waypoint_extraction.extract_waypoints_fast import dp_waypoint_selection_fast


def detect_gripper_changes(actions, left_idx=6, right_idx=13, atol=1.0):
    """
    Detect gripper command transitions from the ACTION signal.
    Consecutive changing frames are grouped into a single transition event;
    only ONE index per event is returned (the first frame with the new value).

    For Episode 4 (4 transitions): returns exactly 4 waypoints.
    """
    change_indices = set()
    for idx in [left_idx, right_idx]:
        gripper_cmd = actions[:, idx]
        if len(gripper_cmd) < 2:
            continue

        diffs = np.abs(np.diff(gripper_cmd))  # (T-1,)
        is_changing = diffs > atol

        # Group consecutive "changing" frames into one event
        # Mark only the FIRST changing frame's target (i+1)
        prev_changing = False
        for i in range(len(is_changing)):
            if is_changing[i] and not prev_changing:
                change_indices.add(i + 1)   # first frame with the new value
            prev_changing = is_changing[i]

    return sorted(change_indices)

# ============================================================
# 1. Configuration
# ============================================================
ERR_THRESHOLD = 0.008     # joint position error threshold (rad)
SRC_DATA_DIR = r"rlds_data"
DST_DATA_DIR = r"rlds_data_waypoint"
# 用于收集每个 episode 的统计信息
EP_STATS = []  # [(ep_idx, original_steps, waypoint_steps), ...]

# ============================================================
# 2. Define the new RLDS DatasetBuilder
#    Feature spec mirrors the original features.json exactly.
# ============================================================
class WaypointFilteredRLDS(tfds.core.GeneratorBasedBuilder):
    # '1.0.0': f'Waypoint-filtered with err_threshold={ERR_THRESHOLD} on both arms + gripper changes.',

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': f'Waypoint-filtered with err_threshold={ERR_THRESHOLD} on both arms + gripper changes.',
    }
    @property
    def _disable_shuffling(self):
        """Disable tfds' hash-based shuffler to preserve episode order."""
        return True
    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            disable_shuffling=True,   # ← ADD THIS
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        # --- Joint positions ---
                        'joint_position_arm_left':  tfds.features.Tensor(shape=(6,),  dtype=tf.float32),
                        'joint_position_arm_right': tfds.features.Tensor(shape=(6,),  dtype=tf.float32),
                        'joint_position_torso':     tfds.features.Tensor(shape=(4,),  dtype=tf.float32),
                        # --- Joint velocities ---
                        'joint_velocity_arm_left':  tfds.features.Tensor(shape=(6,),  dtype=tf.float32),
                        'joint_velocity_arm_right': tfds.features.Tensor(shape=(6,),  dtype=tf.float32),
                        # --- Gripper states ---
                        'gripper_state_left':       tfds.features.Tensor(shape=(1,),  dtype=tf.float32),
                        'gripper_state_right':      tfds.features.Tensor(shape=(1,),  dtype=tf.float32),
                        # --- Base ---
                        'base_velocity':            tfds.features.Tensor(shape=(3,),  dtype=tf.float32),
                        'last_action':              tfds.features.Tensor(shape=(26,), dtype=tf.float32),
                        # --- RGB images ---
                        'image_camera_head':        tfds.features.Image(shape=(224, 224, 3), dtype=tf.uint8),
                        'image_camera_wrist_left':  tfds.features.Image(shape=(224, 224, 3), dtype=tf.uint8),
                        'image_camera_wrist_right': tfds.features.Image(shape=(224, 224, 3), dtype=tf.uint8),
                        'image_camera_head':        tfds.features.Image(shape=(224, 224, 3), dtype=tf.uint8, encoding_format='jpeg'),
                        'image_camera_wrist_left':  tfds.features.Image(shape=(224, 224, 3), dtype=tf.uint8, encoding_format='jpeg'),
                        'image_camera_wrist_right': tfds.features.Image(shape=(224, 224, 3), dtype=tf.uint8, encoding_format='jpeg'),
                        # # --- Depth images ---
                        # 'depth_camera_head':        tfds.features.Image(shape=(224, 224, 1), dtype=tf.uint16),
                        # 'depth_camera_wrist_left':  tfds.features.Image(shape=(224, 224, 1), dtype=tf.uint16),
                        # 'depth_camera_wrist_right': tfds.features.Image(shape=(224, 224, 1), dtype=tf.uint16),
                    }),
                    'action':               tfds.features.Tensor(shape=(26,), dtype=tf.float32),
                    'is_first':             tfds.features.Scalar(dtype=tf.bool),
                    'is_last':              tfds.features.Scalar(dtype=tf.bool),
                    'language_instruction':  tfds.features.Text(),
                    'variant_idx':          tfds.features.Scalar(dtype=tf.int32),
                    'segment_idx':          tfds.features.Scalar(dtype=tf.int32),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(),
                }),
            }),
        )

    def _split_generators(self, dl_manager):
        return {
            'train': self._generate_examples(),
        }

    def _generate_examples(self):
        """Single-pass: read data once, run fast DP inline, yield filtered."""
        print(f"\n{'='*60}")
        print(f"Loading source dataset: {SRC_DATA_DIR}")
        print(f"Waypoint extraction: both arms + gripper, err_threshold={ERR_THRESHOLD}")
        print(f"{'='*60}\n")

        src_builder = tfds.builder_from_directory(SRC_DATA_DIR)
        src_dataset = src_builder.as_dataset(split="train")

        total_src_steps = 0
        total_dst_steps = 0

        for ep_idx, episode in enumerate(src_dataset):
            steps = list(episode["steps"])
            T = len(steps)

            if T == 0:
                print(f"  Episode {ep_idx}: empty, skipping")
                continue

            # --- Extract joint positions + gripper (from already-loaded steps) ---
            jp_left = np.array([
                s["observation"]["joint_position_arm_left"].numpy() for s in steps
            ])
            jp_right = np.array([
                s["observation"]["joint_position_arm_right"].numpy() for s in steps
            ])
            actions_all = np.array([s["action"].numpy() for s in steps])  # (T, 26)
            # gl = np.array([

            wp_left = set(dp_waypoint_selection_fast(jp_left, ERR_THRESHOLD))
            wp_right = set(dp_waypoint_selection_fast(jp_right, ERR_THRESHOLD))
            wp_gripper = set(detect_gripper_changes(actions_all))
            
            waypoints = sorted(wp_left | wp_right | wp_gripper)
            print(f"    [Debug] Waypoint breakdown -> Left: {len(wp_left)}, Right: {len(wp_right)}, Gripper: {len(wp_gripper)}")


            # Ensure start point (index 0) is included
            wp_indices = sorted(set([0] + waypoints))
            n_wp = len(wp_indices)

            total_src_steps += T
            total_dst_steps += n_wp
            EP_STATS.append((ep_idx, T, n_wp))
            print(f"  Episode {ep_idx}: {T} steps -> {n_wp} waypoints "
                f"({T / n_wp:.1f}x compression)")

            file_path = episode["episode_metadata"]["file_path"].numpy().decode("utf-8")
            WAYPOINT_MAP.append({
                "src_ep_idx": ep_idx,
                "file_path": file_path,
                "original_steps": T,
                "waypoint_steps": n_wp,
                "waypoint_indices": [int(x) for x in wp_indices],
            })

            # --- Build filtered episode ---
            ep_steps = []
            for new_i, orig_i in enumerate(wp_indices):
                step = steps[orig_i]
                obs = step["observation"]

                step_dict = {
                    "observation": {
                        "joint_position_arm_left":  obs["joint_position_arm_left"].numpy(),
                        "joint_position_arm_right": obs["joint_position_arm_right"].numpy(),
                        "joint_position_torso":     obs["joint_position_torso"].numpy(),
                        "joint_velocity_arm_left":  obs["joint_velocity_arm_left"].numpy(),
                        "joint_velocity_arm_right": obs["joint_velocity_arm_right"].numpy(),
                        "gripper_state_left":       obs["gripper_state_left"].numpy(),
                        "gripper_state_right":      obs["gripper_state_right"].numpy(),
                        "base_velocity":            obs["base_velocity"].numpy(),
                        "last_action":              obs["last_action"].numpy(),
                        "image_camera_head":        obs["image_camera_head"].numpy(),
                        "image_camera_wrist_left":  obs["image_camera_wrist_left"].numpy(),
                        "image_camera_wrist_right": obs["image_camera_wrist_right"].numpy(),
                    },
                    "action": step["action"].numpy(),
                    "is_first": bool(new_i == 0),
                    "is_last":  bool(new_i == n_wp - 1),
                    "language_instruction": step["language_instruction"].numpy().decode("utf-8"),
                    "variant_idx": int(step["variant_idx"].numpy()),
                    "segment_idx": int(step["segment_idx"].numpy()),
                }
                ep_steps.append(step_dict)

            ep_metadata = {"file_path": file_path}

            yield ep_idx, {
                "steps": ep_steps,
                "episode_metadata": ep_metadata,
            }

        print(f"\n{'='*60}")
        print(f"Summary: {total_src_steps} total steps -> {total_dst_steps} waypoints "
            f"({total_src_steps / max(total_dst_steps, 1):.1f}x overall compression)")
        print(f"{'='*60}\n")

# ============================================================
# 3. Main: build and save the new dataset
# ============================================================
if __name__ == "__main__":
    output_path = os.path.join(DST_DATA_DIR, "waypoint_filtered_rlds", "1.0.0")

    # Clear existing output to allow regeneration
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        print(f"Cleared existing output: {output_path}")

    # Build the dataset (this triggers _generate_examples)
    print(f"Output directory: {DST_DATA_DIR}")
    builder = WaypointFilteredRLDS(data_dir=DST_DATA_DIR)
    builder.download_and_prepare()

    # ============================================================
    # Save waypoint index mapping (for verification)
    # ============================================================
    wp_index_path = os.path.join(DST_DATA_DIR, "waypoint_indices.json")
    wp_index_data = {
        "config": {
            "err_threshold": ERR_THRESHOLD,
            # "extraction_key": "joint_position_arm_left",
            "extraction_key": "joint_position_arm_left + joint_position_arm_right + gripper_changes",
            "src_data_dir": SRC_DATA_DIR,
            "dst_data_dir": DST_DATA_DIR,
            "total_episodes": len(WAYPOINT_MAP),
            "total_src_steps": sum(ep["original_steps"] for ep in WAYPOINT_MAP),
            "total_wp_steps": sum(len(ep["waypoint_indices"]) for ep in WAYPOINT_MAP),
        },
        "episodes": WAYPOINT_MAP,
    }
    with open(wp_index_path, "w", encoding="utf-8") as f:
        json.dump(wp_index_data, f, indent=2, ensure_ascii=False)
    print(f"\nWaypoint indices saved to: {wp_index_path}")
    
    # # ============================================================
    # # 4. Verify: load the new dataset and print statistics
    # # ============================================================
    # print(f"\n{'='*60}")
    # print("Verification: loading the new dataset...")
    # print(f"{'='*60}")

    # new_builder = tfds.builder_from_directory(output_path)
    # new_dataset = new_builder.as_dataset(split="train")

    # for ep_idx, episode in enumerate(new_dataset):
    #     steps = list(episode["steps"])
    #     n = len(steps)

    #     # Spot-check first step's features
    #     first_step = steps[0]
    #     obs = first_step["observation"]
    #     joint_left = obs["joint_position_arm_left"].numpy()
    #     img_shape  = obs["image_camera_head"].shape
    #     # depth_shape = obs["depth_camera_head"].shape

    #     print(f"  Episode {ep_idx}: {n} steps | "
    #           f"joint_left[0]={joint_left[:3]}... | "
    #           f"img={img_shape}")
    #     # ============================================================
    #     # 5. Print per-episode comparison table
    #     # ============================================================
    #     print(f"\n{'='*60}")
    #     print(f"  Waypoint Extraction Summary (err_threshold={ERR_THRESHOLD})")
    #     print(f"{'='*60}")
    #     print(f"  {'Episode':>8} | {'Original':>10} | {'Waypoints':>10} | {'Compression':>12}")
    #     print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")

    #     total_orig = 0
    #     total_wp = 0
    #     for ep_idx, orig, wp in EP_STATS:
    #         ratio = orig / wp if wp > 0 else float('inf')
    #         print(f"  {ep_idx:>8} | {orig:>10} | {wp:>10} | {ratio:>11.1f}x")
    #         total_orig += orig
    #         total_wp += wp

    #     print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
    #     total_ratio = total_orig / total_wp if total_wp > 0 else float('inf')
    #     print(f"  {'Total':>8} | {total_orig:>10} | {total_wp:>10} | {total_ratio:>11.1f}x")
    #     print(f"{'='*60}\n")

    # print(f"\nDone! New RLDS dataset saved at:\n  {output_path}")
    # print(f"\nTo load in your code:")
    # print(f'  builder = tfds.builder_from_directory(r"{output_path}")')
    # print(f'  dataset = builder.as_dataset(split="train")')