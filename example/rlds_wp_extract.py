"""
rlds_wp_extract.py
==================
Extract waypoints from RLDS dataset using AWE dp_waypoint_selection,
with gripper state change detection.

Supports two dataset modes (set DATASET_MODE below):
  - "r1_lite"  : Galaxea R1-Lite dual-arm robot
                  Extracts on joint_position_arm_left + joint_position_arm_right (6D each)
                  Gripper: 1D per arm in action[6] and action[13], range 0-100
  - "libero"   : LIBERO Franka Panda single-arm robot
                  Extracts on observation/state[:6] (EEF 6D pose: 3D pos + 3D axis-angle)
                  Gripper: 1D binary in action[6], values {-1, 1}

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
# Temp file for persisting WAYPOINT_MAP from inside the generator,
# since tfds may consume the generator in a context where global state
# is not propagated back to the main block.
_WP_MAP_TMP = None  # set in __main__
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

awe_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, awe_root)

# Import dp_waypoint_selection_fast directly from file to avoid
# __init__.py dependency chain (imageio, robosuite, etc.)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "extract_waypoints_fast",
    os.path.join(awe_root, "waypoint_extraction", "extract_waypoints_fast.py"),
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
dp_waypoint_selection_fast = _mod.dp_waypoint_selection_fast


def detect_gripper_changes(actions, gripper_indices, atol=1.0):
    """
    Detect gripper command transitions from the ACTION signal.
    Consecutive changing frames are grouped into a single transition event;
    only ONE index per event is returned (the first frame with the new value).

    Parameters
    ----------
    actions : np.ndarray, shape (T, action_dim)
    gripper_indices : list[int]
        Column indices in the action array that contain gripper commands.
        - R1-Lite: [6, 13] (left and right gripper, range 0-100)
        - LIBERO:  [6]     (single gripper, binary {-1, 1})
    atol : float
        Minimum absolute difference to count as a change.
        - R1-Lite: 1.0  (range 0-100, so 1.0 is 1% change)
        - LIBERO:  1.0  (binary {-1,1}, diff=2.0 on transition, always > 1.0)
    """
    change_indices = set()
    for idx in gripper_indices:
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
# --- Dataset mode: "r1_lite" or "libero" ---
DATASET_MODE = "libero"

if DATASET_MODE == "r1_lite":
    ERR_THRESHOLD = 0.008     # joint position error threshold (rad)
    SRC_DATA_DIR = r"/workspace/galaxea_data/rlds/part5_r1_lite/1.0.0"
    DST_DATA_DIR = r"rlds_part5"
    GRIPPER_INDICES = [6, 13]   # left_idx, right_idx in action (range 0-100)
    GRIPPER_ATOL = 1.0
elif DATASET_MODE == "libero":
    ERR_THRESHOLD = 0.01     # EEF pose error threshold (tune as needed)
    SRC_DATA_DIR = r"C:\Users\chuanlia\Documents\learning_space\ntu\projects\check_rlds_data\libero_object_no_noops\libero_object_no_noops\1.0.0"
    DST_DATA_DIR = r"libero_object_wp_001"
    GRIPPER_INDICES = [6]       # single gripper in action (binary {-1, 1})
    GRIPPER_ATOL = 1.0          # diff on transition = 2.0, always > 1.0
else:
    raise ValueError(f"Unknown DATASET_MODE: {DATASET_MODE!r}. Use 'r1_lite' or 'libero'.")

# 用于收集每个 episode 的统计信息
EP_STATS = []  # [(ep_idx, original_steps, waypoint_steps), ...]

# ============================================================
# 2. Define the new RLDS DatasetBuilder
#    Feature spec mirrors the original features.json exactly.
# ============================================================

# --- R1-Lite builder (dual-arm, 3 cameras) ---
class WaypointFilteredRLDS_R1Lite(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': f'Waypoint-filtered with err_threshold={ERR_THRESHOLD} on both arms + gripper changes.',
    }
    @property
    def _disable_shuffling(self):
        return True

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            disable_shuffling=True,
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'joint_position_arm_left':  tfds.features.Tensor(shape=(6,),  dtype=tf.float32),
                        'joint_position_arm_right': tfds.features.Tensor(shape=(6,),  dtype=tf.float32),
                        'joint_position_torso':     tfds.features.Tensor(shape=(4,),  dtype=tf.float32),
                        'joint_velocity_arm_left':  tfds.features.Tensor(shape=(6,),  dtype=tf.float32),
                        'joint_velocity_arm_right': tfds.features.Tensor(shape=(6,),  dtype=tf.float32),
                        'gripper_state_left':       tfds.features.Tensor(shape=(1,),  dtype=tf.float32),
                        'gripper_state_right':      tfds.features.Tensor(shape=(1,),  dtype=tf.float32),
                        'base_velocity':            tfds.features.Tensor(shape=(3,),  dtype=tf.float32),
                        'last_action':              tfds.features.Tensor(shape=(26,), dtype=tf.float32),
                        'image_camera_head':        tfds.features.Image(shape=(224, 224, 3), dtype=tf.uint8, encoding_format='jpeg'),
                        'image_camera_wrist_left':  tfds.features.Image(shape=(224, 224, 3), dtype=tf.uint8, encoding_format='jpeg'),
                        'image_camera_wrist_right': tfds.features.Image(shape=(224, 224, 3), dtype=tf.uint8, encoding_format='jpeg'),
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
        return {'train': self._generate_examples()}

    def _generate_examples(self):
        print(f"\n{'='*60}")
        print(f"[R1-Lite] Loading source dataset: {SRC_DATA_DIR}")
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

            jp_left = np.array([s["observation"]["joint_position_arm_left"].numpy() for s in steps])
            jp_right = np.array([s["observation"]["joint_position_arm_right"].numpy() for s in steps])
            actions_all = np.array([s["action"].numpy() for s in steps])

            wp_left = set(dp_waypoint_selection_fast(jp_left, ERR_THRESHOLD))
            wp_right = set(dp_waypoint_selection_fast(jp_right, ERR_THRESHOLD))
            wp_gripper = set(detect_gripper_changes(actions_all, GRIPPER_INDICES, GRIPPER_ATOL))

            waypoints = sorted(wp_left | wp_right | wp_gripper)
            print(f"    [Debug] Waypoint breakdown -> Left: {len(wp_left)}, Right: {len(wp_right)}, Gripper: {len(wp_gripper)}")

            wp_indices = sorted(set([0] + waypoints))
            n_wp = len(wp_indices)

            total_src_steps += T
            total_dst_steps += n_wp
            EP_STATS.append((ep_idx, T, n_wp))
            print(f"  Episode {ep_idx}: {T} steps -> {n_wp} waypoints ({T / n_wp:.1f}x compression)")

            file_path = episode["episode_metadata"]["file_path"].numpy().decode("utf-8")
            wp_entry = {
                "src_ep_idx": ep_idx, "file_path": file_path,
                "original_steps": T, "waypoint_steps": n_wp,
                "waypoint_indices": [int(x) for x in wp_indices],
            }
            WAYPOINT_MAP.append(wp_entry)
            # Persist incrementally to temp file
            if _WP_MAP_TMP:
                with open(_WP_MAP_TMP, "a", encoding="utf-8") as _f:
                    _f.write(json.dumps(wp_entry, ensure_ascii=False) + "\n")

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

            yield ep_idx, {
                "steps": ep_steps,
                "episode_metadata": {"file_path": file_path},
            }

        print(f"\n{'='*60}")
        print(f"Summary: {total_src_steps} total steps -> {total_dst_steps} waypoints "
              f"({total_src_steps / max(total_dst_steps, 1):.1f}x overall compression)")
        print(f"{'='*60}\n")


# --- LIBERO builder (single-arm Franka Panda, 2 cameras) ---
class WaypointFilteredRLDS_Libero(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': f'Waypoint-filtered with err_threshold={ERR_THRESHOLD} on EEF pose + gripper changes.',
    }
    @property
    def _disable_shuffling(self):
        return True

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            disable_shuffling=True,
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'state':       tfds.features.Tensor(shape=(8,), dtype=tf.float32),
                        'joint_state': tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                        'image':       tfds.features.Image(shape=(224, 224, 3), dtype=tf.uint8, encoding_format='jpeg'),
                        'wrist_image': tfds.features.Image(shape=(224, 224, 3), dtype=tf.uint8, encoding_format='jpeg'),
                    }),
                    'action':               tfds.features.Tensor(shape=(7,), dtype=tf.float32),
                    'is_first':             tfds.features.Scalar(dtype=tf.bool),
                    'is_last':              tfds.features.Scalar(dtype=tf.bool),
                    'is_terminal':          tfds.features.Scalar(dtype=tf.bool),
                    'language_instruction':  tfds.features.Text(),
                    'discount':             tfds.features.Scalar(dtype=tf.float32),
                    'reward':               tfds.features.Scalar(dtype=tf.float32),
                    # Waypoint metadata
                    'waypoint_duration':    tfds.features.Scalar(dtype=tf.int32),
                    'is_waypoint_end':      tfds.features.Scalar(dtype=tf.bool),
                    'original_step_index':  tfds.features.Scalar(dtype=tf.int32),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(),
                }),
            }),
        )

    def _split_generators(self, dl_manager):
        return {'train': self._generate_examples()}

    def _generate_examples(self):
        print(f"\n{'='*60}")
        print(f"[LIBERO] Loading source dataset: {SRC_DATA_DIR}")
        print(f"Waypoint extraction: EEF pose (state[:6]) + gripper, err_threshold={ERR_THRESHOLD}")
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

            # Extract EEF 6D pose from observation/state[:6]
            # state = [3D position, 3D axis-angle orientation, 2D gripper]
            ee_poses = np.array([s["observation"]["state"].numpy()[:6] for s in steps])  # (T, 6)
            actions_all = np.array([s["action"].numpy() for s in steps])  # (T, 7)

            # Waypoint extraction on EEF 6D pose
            wp_ee = set(dp_waypoint_selection_fast(ee_poses, ERR_THRESHOLD))
            # Gripper change detection from action[6] (binary: {-1, 1})
            wp_gripper = set(detect_gripper_changes(actions_all, GRIPPER_INDICES, GRIPPER_ATOL))

            waypoints = sorted(wp_ee | wp_gripper)
            print(f"    [Debug] Waypoint breakdown -> EEF: {len(wp_ee)}, Gripper: {len(wp_gripper)}")

            wp_indices = sorted(set([0] + waypoints))
            n_wp = len(wp_indices)

            total_src_steps += T
            total_dst_steps += n_wp
            EP_STATS.append((ep_idx, T, n_wp))
            print(f"  Episode {ep_idx}: {T} steps -> {n_wp} waypoints ({T / n_wp:.1f}x compression)")

            file_path = episode["episode_metadata"]["file_path"].numpy().decode("utf-8")
            wp_entry = {
                "src_ep_idx": ep_idx, "file_path": file_path,
                "original_steps": T, "waypoint_steps": n_wp,
                "waypoint_indices": [int(x) for x in wp_indices],
            }
            WAYPOINT_MAP.append(wp_entry)
            # Persist incrementally to temp file
            if _WP_MAP_TMP:
                with open(_WP_MAP_TMP, "a", encoding="utf-8") as _f:
                    _f.write(json.dumps(wp_entry, ensure_ascii=False) + "\n")

            ep_steps = []
            for new_i, orig_i in enumerate(wp_indices):
                step = steps[orig_i]
                obs = step["observation"]
                # Compute waypoint metadata
                if new_i < n_wp - 1:
                    duration = wp_indices[new_i + 1] - wp_indices[new_i]
                else:
                    duration = 0  # last waypoint has no next waypoint
                is_wp_end = bool(new_i == n_wp - 1)
                step_dict = {
                    # "observation": {
                    #     "state":       obs["state"].numpy(),
                    #     "joint_state": obs["joint_state"].numpy(),
                    #     "image":       obs["image"].numpy(),
                    #     "wrist_image": obs["wrist_image"].numpy(),
                    # },
                    "observation": {
                        "state":       obs["state"].numpy(),
                        "joint_state": obs["joint_state"].numpy(),
                        "image":       tf.cast(tf.image.resize(obs["image"], [224, 224]), tf.uint8).numpy(),
                        "wrist_image": tf.cast(tf.image.resize(obs["wrist_image"], [224, 224]), tf.uint8).numpy(),
                    },
                    "action":              step["action"].numpy(),
                    "is_first":            bool(new_i == 0),
                    "is_last":             bool(new_i == n_wp - 1),
                    "is_terminal":         bool(step["is_terminal"].numpy()),
                    "language_instruction": step["language_instruction"].numpy().decode("utf-8"),
                    "discount":            float(step["discount"].numpy()),
                    "reward":              float(step["reward"].numpy()),
                    # Waypoint metadata
                    "waypoint_duration":   int(duration),
                    "is_waypoint_end":     is_wp_end,
                    "original_step_index": int(orig_i),
                }
                ep_steps.append(step_dict)

            yield ep_idx, {
                "steps": ep_steps,
                "episode_metadata": {"file_path": file_path},
            }

        print(f"\n{'='*60}")
        print(f"Summary: {total_src_steps} total steps -> {total_dst_steps} waypoints "
              f"({total_src_steps / max(total_dst_steps, 1):.1f}x overall compression)")
        print(f"{'='*60}\n")


# ============================================================
# 3. Main: build and save the new dataset
# ============================================================
if __name__ == "__main__":
    # Select builder class based on dataset mode
    if DATASET_MODE == "r1_lite":
        BuilderClass = WaypointFilteredRLDS_R1Lite
        extraction_key = "joint_position_arm_left + joint_position_arm_right + gripper_changes"
    elif DATASET_MODE == "libero":
        BuilderClass = WaypointFilteredRLDS_Libero
        extraction_key = "state[:6] (EEF 6D pose) + gripper_changes (action[6])"
    else:
        raise ValueError(f"Unknown DATASET_MODE: {DATASET_MODE!r}")

    # Set up temp file for persisting waypoint map from inside the generator
    _WP_MAP_TMP = os.path.join(DST_DATA_DIR, ".waypoint_map_tmp.jsonl")
    os.makedirs(DST_DATA_DIR, exist_ok=True)
    # Clear any leftover temp file
    if os.path.exists(_WP_MAP_TMP):
        os.remove(_WP_MAP_TMP)

    # Build the dataset (this triggers _generate_examples)
    print(f"Dataset mode: {DATASET_MODE}")
    print(f"Output directory: {DST_DATA_DIR}")
    builder = BuilderClass(data_dir=DST_DATA_DIR)
    builder.download_and_prepare()

    # ============================================================
    # Recover WAYPOINT_MAP from temp file (in case global list is empty)
    # ============================================================
    if not WAYPOINT_MAP and os.path.exists(_WP_MAP_TMP):
        print("Recovering waypoint map from temp file...")
        with open(_WP_MAP_TMP, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    WAYPOINT_MAP.append(json.loads(line))
        print(f"  Recovered {len(WAYPOINT_MAP)} episodes from temp file")

    # Clean up temp file
    if os.path.exists(_WP_MAP_TMP):
        os.remove(_WP_MAP_TMP)

    # ============================================================
    # Save waypoint index mapping (for verification)
    # ============================================================
    wp_index_path = os.path.join(DST_DATA_DIR, "waypoint_indices.json")
    wp_index_data = {
        "config": {
            "dataset_mode": DATASET_MODE,
            "err_threshold": ERR_THRESHOLD,
            "extraction_key": extraction_key,
            "gripper_indices": GRIPPER_INDICES,
            "gripper_atol": GRIPPER_ATOL,
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
    print(f"  Total episodes: {len(WAYPOINT_MAP)}")
    if WAYPOINT_MAP:
        total_src = sum(ep["original_steps"] for ep in WAYPOINT_MAP)
        total_wp = sum(ep["waypoint_steps"] for ep in WAYPOINT_MAP)
        print(f"  Total source steps: {total_src}")
        print(f"  Total waypoint steps: {total_wp}")
        print(f"  Overall compression: {total_src / max(total_wp, 1):.1f}x")