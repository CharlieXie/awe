"""
rlds_wp_extract_parallel.py
============================
Parallel waypoint extraction from RLDS dataset.

Key optimisations over previous version:
  - Direct TFRecordWriter instead of tfds GeneratorBasedBuilder
    → each episode is written to disk immediately, no builder buffering
    → peak memory per worker ≈ TF runtime (~3 GB) + 1 episode (~200 MB)
  - Aggressive memory release: del + gc.collect() after every episode
  - Shard-aligned tfds percentage splits: zero-waste I/O per worker
  - TF thread limiting: 2 intra/inter-op threads per worker

Strategy:
  1. Read source dataset_info.json to get shardLengths (e.g. 2048 shards)
  2. Assign shard ranges to N workers via tfds percentage splits
  3. Each worker:
     - Loads its assigned shards via  split="train[X%:Y%]"
     - Runs waypoint extraction on each episode
     - Serialises filtered episode as tf.train.Example (same format as tfds)
     - Writes directly to TFRecord files, 1 episode per record
  4. Main process merges all worker TFRecord files into one RLDS dataset:
     - Renames/renumbers shard files
     - Writes dataset_info.json + features.json

Output is a valid RLDS dataset loadable with tfds.builder_from_directory().

Memory budget: designed for 172 GB.  Default 16 workers ≈ 80-100 GB peak.

Usage:
    python rlds_wp_extract_parallel.py [--num_workers 16]
"""

import sys
import os
import types
import shutil
import json
import glob
import time
import gc
import argparse
import multiprocessing as mp
import numpy as np

# ============================================================
# 0. Mock mujoco_py / glfw  (must happen before any awe import)
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

# ============================================================
# Configuration
# ============================================================
ERR_THRESHOLD = 0.008
SRC_DATA_DIR  = r"/workspace/galaxea_data/rlds/part5_r1_lite/1.0.0"
DST_DATA_DIR  = r"rlds_part5"
EPISODES_PER_SHARD = 8   # how many episodes to pack into each output shard


# ============================================================
# Gripper change detection (same as original)
# ============================================================
def detect_gripper_changes(actions, left_idx=6, right_idx=13, atol=1.0):
    change_indices = set()
    for idx in [left_idx, right_idx]:
        gripper_cmd = actions[:, idx]
        if len(gripper_cmd) < 2:
            continue
        diffs = np.abs(np.diff(gripper_cmd))
        is_changing = diffs > atol
        prev_changing = False
        for i in range(len(is_changing)):
            if is_changing[i] and not prev_changing:
                change_indices.add(i + 1)
            prev_changing = is_changing[i]
    return sorted(change_indices)


# ============================================================
# TFRecord serialisation helpers
# ============================================================
def _bytes_feature(value):
    """Returns a bytes_list from a list of bytes / strings."""
    if isinstance(value, (bytes, bytearray)):
        value = [value]
    return {"bytes_list": {"value": list(value)}}


def _float_feature(value):
    """Returns a float_list from a numpy array (flattened)."""
    return {"float_list": {"value": list(np.asarray(value, dtype=np.float32).flat)}}


def _int64_feature(value):
    """Returns an int64_list from a numpy array (flattened)."""
    return {"int64_list": {"value": list(np.asarray(value, dtype=np.int64).flat)}}


def _serialize_episode(ep_steps, file_path, tf):
    """Serialise one filtered episode as tf.train.Example bytes.

    The feature key layout exactly matches what tfds produces for RLDS:
      - steps/observation/... , steps/action, steps/is_first, ...
      - episode_metadata/file_path
    Sequence features (steps/*) are stored as variable-length lists in a flat
    tf.train.Example (same as tfds Dataset feature).
    Images are re-encoded as JPEG bytes.
    """
    n_steps = len(ep_steps)

    # ---- Collect per-step sequences ----
    # Float tensors  (each stored as concatenated float_list)
    joint_pos_left_vals   = []
    joint_pos_right_vals  = []
    joint_pos_torso_vals  = []
    joint_vel_left_vals   = []
    joint_vel_right_vals  = []
    gripper_left_vals     = []
    gripper_right_vals    = []
    base_velocity_vals    = []
    last_action_vals      = []
    action_vals           = []

    # Int64 tensors
    is_first_vals = []
    is_last_vals  = []
    variant_idx_vals = []
    segment_idx_vals = []

    # Bytes (images as JPEG, text as UTF-8)
    img_head_vals        = []
    img_wrist_left_vals  = []
    img_wrist_right_vals = []
    lang_vals            = []

    _encode_jpeg = tf.image.encode_jpeg

    for step in ep_steps:
        obs = step["observation"]

        # Float tensors → extend flat list
        joint_pos_left_vals.extend(obs["joint_position_arm_left"].flat)
        joint_pos_right_vals.extend(obs["joint_position_arm_right"].flat)
        joint_pos_torso_vals.extend(obs["joint_position_torso"].flat)
        joint_vel_left_vals.extend(obs["joint_velocity_arm_left"].flat)
        joint_vel_right_vals.extend(obs["joint_velocity_arm_right"].flat)
        gripper_left_vals.extend(obs["gripper_state_left"].flat)
        gripper_right_vals.extend(obs["gripper_state_right"].flat)
        base_velocity_vals.extend(obs["base_velocity"].flat)
        last_action_vals.extend(obs["last_action"].flat)
        action_vals.extend(step["action"].flat)

        # Bool → int64 (tf.train.Example has no bool type)
        is_first_vals.append(int(step["is_first"]))
        is_last_vals.append(int(step["is_last"]))

        # Int scalars
        variant_idx_vals.append(step["variant_idx"])
        segment_idx_vals.append(step["segment_idx"])

        # Images → JPEG bytes
        img_head_vals.append(
            _encode_jpeg(obs["image_camera_head"]).numpy())
        img_wrist_left_vals.append(
            _encode_jpeg(obs["image_camera_wrist_left"]).numpy())
        img_wrist_right_vals.append(
            _encode_jpeg(obs["image_camera_wrist_right"]).numpy())

        # Text
        lang_vals.append(step["language_instruction"].encode("utf-8"))

    # ---- Build the flat feature dict ----
    feature_dict = {
        # Observations (float)
        "steps/observation/joint_position_arm_left":
            _float_feature(joint_pos_left_vals),
        "steps/observation/joint_position_arm_right":
            _float_feature(joint_pos_right_vals),
        "steps/observation/joint_position_torso":
            _float_feature(joint_pos_torso_vals),
        "steps/observation/joint_velocity_arm_left":
            _float_feature(joint_vel_left_vals),
        "steps/observation/joint_velocity_arm_right":
            _float_feature(joint_vel_right_vals),
        "steps/observation/gripper_state_left":
            _float_feature(gripper_left_vals),
        "steps/observation/gripper_state_right":
            _float_feature(gripper_right_vals),
        "steps/observation/base_velocity":
            _float_feature(base_velocity_vals),
        "steps/observation/last_action":
            _float_feature(last_action_vals),
        # Observations (image bytes)
        "steps/observation/image_camera_head":
            _bytes_feature(img_head_vals),
        "steps/observation/image_camera_wrist_left":
            _bytes_feature(img_wrist_left_vals),
        "steps/observation/image_camera_wrist_right":
            _bytes_feature(img_wrist_right_vals),
        # Action (float)
        "steps/action":
            _float_feature(action_vals),
        # Scalars (int64)
        "steps/is_first":
            _int64_feature(is_first_vals),
        "steps/is_last":
            _int64_feature(is_last_vals),
        "steps/variant_idx":
            _int64_feature(variant_idx_vals),
        "steps/segment_idx":
            _int64_feature(segment_idx_vals),
        # Text (bytes)
        "steps/language_instruction":
            _bytes_feature(lang_vals),
        # Episode metadata
        "episode_metadata/file_path":
            _bytes_feature(file_path.encode("utf-8")),
    }

    # ---- Construct tf.train.Example proto ----
    # Build using tf proto helpers for correctness
    features = {}
    for key, feat_spec in feature_dict.items():
        if "float_list" in feat_spec:
            features[key] = tf.train.Feature(
                float_list=tf.train.FloatList(value=feat_spec["float_list"]["value"]))
        elif "int64_list" in feat_spec:
            features[key] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=feat_spec["int64_list"]["value"]))
        elif "bytes_list" in feat_spec:
            features[key] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=feat_spec["bytes_list"]["value"]))

    example = tf.train.Example(
        features=tf.train.Features(feature=features))
    return example.SerializeToString()


# ============================================================
# Worker function  (runs in a separate process)
# ============================================================
def worker_process(args):
    """
    Each worker:
      - Loads episodes via shard-aligned split="train[X%:Y%]"
      - Runs waypoint extraction
      - Writes filtered episodes directly as TFRecord files
      - Returns (worker_id, worker_dst, elapsed, shard_lengths)
    """
    (worker_id, pct_start, pct_end, ep_offset,
     src_data_dir, tmp_dir, err_threshold, eps_per_shard) = args

    # --- Limit TF threads BEFORE importing TF ---
    os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
    os.environ['TF_NUM_INTEROP_THREADS'] = '2'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    try:
        tf.config.set_visible_devices([], 'GPU')
    except RuntimeError:
        pass  # GPUs already initialised

    import tensorflow_datasets as tfds
    from waypoint_extraction.extract_waypoints_fast import dp_waypoint_selection_fast

    worker_dst = os.path.join(tmp_dir, f"worker_{worker_id:03d}")
    os.makedirs(worker_dst, exist_ok=True)

    split_str = f"train[{pct_start}%:{pct_end}%]"
    print(f"[Worker {worker_id}] Starting: split={split_str}  "
          f"ep_offset={ep_offset} -> {worker_dst}", flush=True)

    # ---- Load source dataset (only this worker's shards) ----
    src_builder = tfds.builder_from_directory(src_data_dir)
    src_dataset = src_builder.as_dataset(split=split_str)

    t0 = time.time()
    ep_stats_local = []
    wp_map_local = []

    # ---- Process episodes, write TFRecords directly ----
    shard_idx = 0
    ep_in_shard = 0
    shard_lengths = []   # number of episodes per shard file
    writer = None

    def _open_new_shard():
        nonlocal writer, shard_idx, ep_in_shard
        if writer is not None:
            writer.close()
            shard_lengths.append(ep_in_shard)
        shard_path = os.path.join(
            worker_dst,
            f"worker_{worker_id:03d}-train.tfrecord-{shard_idx:05d}")
        writer = tf.io.TFRecordWriter(shard_path)
        shard_idx += 1
        ep_in_shard = 0

    _open_new_shard()

    for local_idx, episode in enumerate(src_dataset):
        global_ep_idx = ep_offset + local_idx
        steps = list(episode["steps"])
        T = len(steps)

        if T == 0:
            print(f"  [W{worker_id}] Episode {global_ep_idx}: empty, skipping",
                  flush=True)
            del steps
            continue

        # --- Extract joint positions for waypoint computation ---
        jp_left = np.array([
            s["observation"]["joint_position_arm_left"].numpy() for s in steps
        ])
        jp_right = np.array([
            s["observation"]["joint_position_arm_right"].numpy() for s in steps
        ])
        actions_all = np.array([s["action"].numpy() for s in steps])

        wp_left = set(dp_waypoint_selection_fast(jp_left, err_threshold))
        wp_right = set(dp_waypoint_selection_fast(jp_right, err_threshold))
        wp_gripper = set(detect_gripper_changes(actions_all))

        waypoints = sorted(wp_left | wp_right | wp_gripper)
        wp_indices = sorted(set([0] + waypoints))
        n_wp = len(wp_indices)

        # Free waypoint intermediates
        del jp_left, jp_right, actions_all, wp_left, wp_right, wp_gripper, waypoints

        file_path = episode["episode_metadata"]["file_path"].numpy().decode("utf-8")

        ep_stats_local.append((global_ep_idx, T, n_wp))
        wp_map_local.append({
            "src_ep_idx": global_ep_idx,
            "file_path": file_path,
            "original_steps": T,
            "waypoint_steps": n_wp,
            "waypoint_indices": [int(x) for x in wp_indices],
        })

        # --- Build filtered step dicts (numpy, not TF tensors) ---
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

        # --- Free the original TF steps immediately ---
        del steps, episode, wp_indices

        # --- Serialise and write ---
        serialized = _serialize_episode(ep_steps, file_path, tf)
        writer.write(serialized)
        ep_in_shard += 1
        del ep_steps, serialized

        # Start new shard if full
        if ep_in_shard >= eps_per_shard:
            _open_new_shard()

        # --- Periodic GC and logging ---
        if local_idx % 10 == 0:
            gc.collect()
        if local_idx % 50 == 0 or local_idx == 0:
            elapsed_so_far = time.time() - t0
            print(f"  [W{worker_id}] Episode {global_ep_idx}: "
                  f"{T} -> {n_wp} waypoints  "
                  f"({local_idx+1} done, {elapsed_so_far:.0f}s elapsed)",
                  flush=True)

    # Close last shard
    if writer is not None:
        writer.close()
        if ep_in_shard > 0:
            shard_lengths.append(ep_in_shard)

    total_local = local_idx + 1 if 'local_idx' in dir() else 0
    elapsed = time.time() - t0

    # Save per-worker stats
    stats_path = os.path.join(worker_dst, "_worker_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "ep_stats": ep_stats_local,
            "wp_map": wp_map_local,
            "shard_lengths": shard_lengths,
        }, f)

    print(f"[Worker {worker_id}] Done: {total_local} episodes, "
          f"{len(shard_lengths)} shards, {elapsed:.1f}s ({elapsed/60:.1f} min)",
          flush=True)
    return worker_id, worker_dst, elapsed


# ============================================================
# Merge all worker TFRecord files into one RLDS dataset
# ============================================================
def merge_worker_outputs(tmp_dir, final_output_dir, num_workers, dataset_name):
    """
    Merge N worker TFRecord outputs into a single RLDS-compatible dataset.

    Each worker produced:
      <tmp_dir>/worker_XXX/
        - worker_XXX-train.tfrecord-00000
        - worker_XXX-train.tfrecord-00001
        - ...
        - _worker_stats.json  (contains shard_lengths)

    We:
      1. Collect all tfrecord shards from all workers (in order)
      2. Rename them with global sequential numbering
      3. Concatenate shard_lengths
      4. Write dataset_info.json + features.json
    """
    print(f"\n{'='*60}")
    print("Merging worker outputs...")
    print(f"{'='*60}")

    if os.path.exists(final_output_dir):
        shutil.rmtree(final_output_dir)
    os.makedirs(final_output_dir, exist_ok=True)

    all_shard_lengths = []
    all_shard_files = []  # (src_path, shard_length)

    for wid in range(num_workers):
        worker_dir = os.path.join(tmp_dir, f"worker_{wid:03d}")

        # Read shard_lengths from worker stats
        stats_path = os.path.join(worker_dir, "_worker_stats.json")
        if not os.path.exists(stats_path):
            print(f"  WARNING: Worker {wid} has no stats, skipping")
            continue
        with open(stats_path, "r") as f:
            stats = json.load(f)
        worker_shard_lengths = stats.get("shard_lengths", [])

        # Find all tfrecord shard files for this worker (sorted)
        shard_files = sorted(glob.glob(
            os.path.join(worker_dir, f"worker_{wid:03d}-train.tfrecord-*")))

        if len(shard_files) != len(worker_shard_lengths):
            print(f"  WARNING: Worker {wid}: {len(shard_files)} files vs "
                  f"{len(worker_shard_lengths)} shard_lengths")

        for sf in shard_files:
            all_shard_files.append(sf)
        all_shard_lengths.extend(worker_shard_lengths)

    total_shards = len(all_shard_files)
    print(f"  Total shards across all workers: {total_shards}")

    # Rename and copy/link
    for idx, src_path in enumerate(all_shard_files):
        dst_name = (f"{dataset_name}-train.tfrecord-"
                    f"{idx:05d}-of-{total_shards:05d}")
        dst_path = os.path.join(final_output_dir, dst_name)
        try:
            os.link(src_path, dst_path)
        except OSError:
            shutil.copy2(src_path, dst_path)

    # Write dataset_info.json
    total_episodes = sum(all_shard_lengths)
    dataset_info = {
        "fileFormat": "tfrecord",
        "moduleName": f"{dataset_name}.{dataset_name}",
        "name": dataset_name,
        "releaseNotes": {
            "1.0.0": f"Waypoint-filtered with err_threshold={ERR_THRESHOLD} "
                     f"on both arms + gripper changes."
        },
        "splits": [
            {
                "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
                "name": "train",
                "numBytes": "0",
                "shardLengths": [str(x) for x in all_shard_lengths],
            }
        ],
        "version": "1.0.0",
    }
    with open(os.path.join(final_output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"  Merged {total_shards} shards, {total_episodes} episodes")
    print(f"  Output: {final_output_dir}")
    return total_episodes


def write_features_json(final_output_dir):
    """Write features.json that matches the output schema (no depth images)."""
    features = {
        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
        "featuresDict": {
            "features": {
                "steps": {
                    "pythonClassName": "tensorflow_datasets.core.features.dataset_feature.Dataset",
                    "sequence": {
                        "feature": {
                            "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                            "featuresDict": {
                                "features": {
                                    "is_last": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                                        "description": "True on last step of the episode."
                                    },
                                    "language_instruction": {
                                        "pythonClassName": "tensorflow_datasets.core.features.text_feature.Text",
                                        "text": {},
                                        "description": "Language Instruction."
                                    },
                                    "variant_idx": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "int32", "encoding": "none"},
                                        "description": "Bowl order index."
                                    },
                                    "observation": {
                                        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                                        "featuresDict": {
                                            "features": {
                                                "last_action": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {"shape": {"dimensions": ["26"]}, "dtype": "float32", "encoding": "none"},
                                                    "description": "Last robot action."
                                                },
                                                "joint_position_arm_right": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {"shape": {"dimensions": ["6"]}, "dtype": "float32", "encoding": "none"},
                                                    "description": "Right arm joint positions."
                                                },
                                                "image_camera_wrist_right": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
                                                    "image": {"shape": {"dimensions": ["224", "224", "3"]}, "dtype": "uint8", "encodingFormat": "jpeg"},
                                                    "description": "RGB image from right wrist camera."
                                                },
                                                "gripper_state_right": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {"shape": {"dimensions": ["1"]}, "dtype": "float32", "encoding": "none"},
                                                    "description": "Right gripper state."
                                                },
                                                "joint_velocity_arm_right": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {"shape": {"dimensions": ["6"]}, "dtype": "float32", "encoding": "none"},
                                                    "description": "Right arm joint velocities."
                                                },
                                                "image_camera_wrist_left": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
                                                    "image": {"shape": {"dimensions": ["224", "224", "3"]}, "dtype": "uint8", "encodingFormat": "jpeg"},
                                                    "description": "RGB image from left wrist camera."
                                                },
                                                "image_camera_head": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
                                                    "image": {"shape": {"dimensions": ["224", "224", "3"]}, "dtype": "uint8", "encodingFormat": "jpeg"},
                                                    "description": "RGB image from head camera."
                                                },
                                                "joint_position_torso": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {"shape": {"dimensions": ["4"]}, "dtype": "float32", "encoding": "none"},
                                                    "description": "Torso joint positions."
                                                },
                                                "gripper_state_left": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {"shape": {"dimensions": ["1"]}, "dtype": "float32", "encoding": "none"},
                                                    "description": "Left gripper state."
                                                },
                                                "joint_velocity_arm_left": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {"shape": {"dimensions": ["6"]}, "dtype": "float32", "encoding": "none"},
                                                    "description": "Left arm joint velocities."
                                                },
                                                "joint_position_arm_left": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {"shape": {"dimensions": ["6"]}, "dtype": "float32", "encoding": "none"},
                                                    "description": "Left arm joint positions."
                                                },
                                                "base_velocity": {
                                                    "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                                    "tensor": {"shape": {"dimensions": ["3"]}, "dtype": "float32", "encoding": "none"},
                                                    "description": "Base velocity."
                                                }
                                            }
                                        }
                                    },
                                    "is_first": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"},
                                        "description": "True on first step of the episode."
                                    },
                                    "segment_idx": {
                                        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
                                        "tensor": {"shape": {}, "dtype": "int32", "encoding": "none"},
                                        "description": "Segment index."
                                    },
                                    "action": {
                                        "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
                                        "tensor": {"shape": {"dimensions": ["26"]}, "dtype": "float32", "encoding": "none"},
                                        "description": "Robot action, consists of [6x arm joint position, 1x gripper absolute (0...100, 0 close, 100 open)] x 2. + [6x torso cmd] + [6x chasis cmd]"
                                    }
                                }
                            }
                        },
                        "length": "-1"
                    }
                },
                "episode_metadata": {
                    "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                    "featuresDict": {
                        "features": {
                            "file_path": {
                                "pythonClassName": "tensorflow_datasets.core.features.text_feature.Text",
                                "text": {},
                                "description": "Path to the original data file."
                            }
                        }
                    }
                }
            }
        }
    }
    with open(os.path.join(final_output_dir, "features.json"), "w") as f:
        json.dump(features, f, indent=4)


def merge_worker_stats(tmp_dir, num_workers, dst_data_dir):
    """Merge all worker _worker_stats.json into a single waypoint_indices.json."""
    all_ep_stats = []
    all_wp_map = []

    for wid in range(num_workers):
        worker_dir = os.path.join(tmp_dir, f"worker_{wid:03d}")
        stats_path = os.path.join(worker_dir, "_worker_stats.json")
        if not os.path.exists(stats_path):
            continue
        with open(stats_path, "r") as f:
            data = json.load(f)
        all_ep_stats.extend(data.get("ep_stats", []))
        all_wp_map.extend(data.get("wp_map", []))

    # Sort by global episode index
    all_wp_map.sort(key=lambda x: x["src_ep_idx"])

    wp_index_data = {
        "config": {
            "err_threshold": ERR_THRESHOLD,
            "extraction_key": "joint_position_arm_left + joint_position_arm_right + gripper_changes",
            "src_data_dir": SRC_DATA_DIR,
            "dst_data_dir": dst_data_dir,
            "total_episodes": len(all_wp_map),
            "total_src_steps": sum(ep["original_steps"] for ep in all_wp_map),
            "total_wp_steps": sum(ep["waypoint_steps"] for ep in all_wp_map),
        },
        "episodes": all_wp_map,
    }
    wp_index_path = os.path.join(dst_data_dir, "waypoint_indices.json")
    with open(wp_index_path, "w", encoding="utf-8") as f:
        json.dump(wp_index_data, f, indent=2, ensure_ascii=False)
    print(f"Waypoint indices saved to: {wp_index_path}")

    if all_wp_map:
        total_src = sum(ep["original_steps"] for ep in all_wp_map)
        total_wp = sum(ep["waypoint_steps"] for ep in all_wp_map)
        ratio = total_src / max(total_wp, 1)
        print(f"\nSummary: {total_src} total steps -> {total_wp} waypoints "
              f"({ratio:.1f}x overall compression)")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel waypoint extraction from RLDS dataset "
                    "(direct TFRecord writer, low memory)."
    )
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel workers (default: 16)")
    parser.add_argument("--eps_per_shard", type=int, default=EPISODES_PER_SHARD,
                        help=f"Episodes per output shard (default: {EPISODES_PER_SHARD})")
    parser.add_argument("--keep_tmp", action="store_true",
                        help="Keep temporary worker directories after merge")
    args = parser.parse_args()

    NUM_WORKERS = args.num_workers

    # ---- Auto-detect total episodes and shards from dataset_info.json ----
    info_path = os.path.join(SRC_DATA_DIR, "dataset_info.json")
    with open(info_path, "r") as f:
        src_info = json.load(f)
    shard_lengths = [int(x) for x in src_info["splits"][0]["shardLengths"]]
    NUM_SHARDS = len(shard_lengths)
    TOTAL_EPISODES = sum(shard_lengths)

    print(f"Source dataset: {TOTAL_EPISODES} episodes across {NUM_SHARDS} shards")
    print(f"Parallel workers: {NUM_WORKERS}")
    print(f"Episodes per output shard: {args.eps_per_shard}")
    print(f"Estimated peak memory per worker: ~5-8 GB  "
          f"(total ~{NUM_WORKERS * 6} GB)")

    # ---- Compute shard-aligned percentage splits ----
    shards_per_worker = NUM_SHARDS // NUM_WORKERS
    remainder_shards = NUM_SHARDS % NUM_WORKERS

    worker_assignments = []  # (worker_id, pct_start, pct_end, ep_offset)
    shard_offset = 0
    ep_offset = 0

    for w in range(NUM_WORKERS):
        n_shards = shards_per_worker + (1 if w < remainder_shards else 0)
        shard_start = shard_offset
        shard_end = shard_offset + n_shards

        pct_start = int(round(shard_start * 100.0 / NUM_SHARDS))
        pct_end   = int(round(shard_end   * 100.0 / NUM_SHARDS))
        if w == NUM_WORKERS - 1:
            pct_end = 100

        worker_episodes = sum(shard_lengths[shard_start:shard_end])
        worker_assignments.append((w, pct_start, pct_end, ep_offset))

        shard_offset = shard_end
        ep_offset += worker_episodes

    print(f"\nWorker shard assignments:")
    for w, pct_s, pct_e, ep_off in worker_assignments:
        print(f"  Worker {w:2d}: split=train[{pct_s}%:{pct_e}%]  ep_offset={ep_off}")

    # ---- Temporary directory for worker outputs ----
    tmp_dir = os.path.join(DST_DATA_DIR, "_tmp_parallel")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    # ---- Launch workers ----
    worker_args = [
        (w, pct_s, pct_e, ep_off, SRC_DATA_DIR, tmp_dir, ERR_THRESHOLD,
         args.eps_per_shard)
        for w, pct_s, pct_e, ep_off in worker_assignments
    ]

    print(f"\n{'='*60}")
    print(f"Launching {NUM_WORKERS} workers...")
    print(f"{'='*60}\n")

    t_start = time.time()

    ctx = mp.get_context('spawn')
    with ctx.Pool(NUM_WORKERS) as pool:
        results = pool.map(worker_process, worker_args)

    t_parallel = time.time() - t_start
    print(f"\nAll workers done in {t_parallel:.1f}s "
          f"({t_parallel/60:.1f} min)")

    for wid, wdir, elapsed in results:
        print(f"  Worker {wid}: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ---- Merge ----
    dataset_name = "waypoint_filtered_rlds"
    final_output_dir = os.path.join(DST_DATA_DIR, dataset_name, "1.0.0")

    merge_worker_outputs(tmp_dir, final_output_dir, NUM_WORKERS, dataset_name)
    write_features_json(final_output_dir)
    merge_worker_stats(tmp_dir, NUM_WORKERS, DST_DATA_DIR)

    # ---- Cleanup ----
    if not args.keep_tmp:
        print(f"\nCleaning up temp directory: {tmp_dir}")
        shutil.rmtree(tmp_dir)
    else:
        print(f"\nTemp directory kept at: {tmp_dir}")

    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {t_total:.1f}s ({t_total/60:.1f} min)")
    print(f"{'='*60}")
    print(f"\nDone! New RLDS dataset saved at:\n  {final_output_dir}")
    print(f"\nTo load in your code:")
    print(f'  builder = tfds.builder_from_directory(r"{final_output_dir}")')
    print(f'  dataset = builder.as_dataset(split="train")')
