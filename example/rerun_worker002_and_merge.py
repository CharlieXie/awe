"""
rerun_worker002_and_merge.py
=============================
1. Clean up incomplete worker_002 data
2. Re-run worker_002's shard range  (train[25%:38%], ep_offset=4054)
3. Merge all 8 workers into the final RLDS dataset

Usage (inside awe_venv):
    conda activate awe_venv
    cd /workspace/awe/example
    python rerun_worker002_and_merge.py
"""

import sys
import os
import types
import shutil
import json
import glob
import time
import gc
import numpy as np

# ============================================================
# 0. Mock mujoco_py / glfw
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
# Configuration (must match rlds_wp_extract_parallel.py)
# ============================================================
ERR_THRESHOLD = 0.008
SRC_DATA_DIR  = r"/workspace/galaxea_data/rlds/part5_r1_lite/1.0.0"
DST_DATA_DIR  = r"rlds_part5"
TMP_DIR       = os.path.join(DST_DATA_DIR, "_tmp_parallel")
EPISODES_PER_SHARD = 8

# Worker 002 parameters (from the 8-worker run)
WORKER_ID  = 2
PCT_START  = 25
PCT_END    = 38
EP_OFFSET  = 4054

os.environ['TF_NUM_INTRAOP_THREADS'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
try:
    tf.config.set_visible_devices([], 'GPU')
except RuntimeError:
    pass

import tensorflow_datasets as tfds
from waypoint_extraction.extract_waypoints_fast import dp_waypoint_selection_fast


# ============================================================
# Helpers (same as rlds_wp_extract_parallel.py)
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


def _bytes_feature(value):
    if isinstance(value, (bytes, bytearray)):
        value = [value]
    return {"bytes_list": {"value": list(value)}}

def _float_feature(value):
    return {"float_list": {"value": list(np.asarray(value, dtype=np.float32).flat)}}

def _int64_feature(value):
    return {"int64_list": {"value": list(np.asarray(value, dtype=np.int64).flat)}}


def _serialize_episode(ep_steps, file_path):
    n_steps = len(ep_steps)
    joint_pos_left_vals, joint_pos_right_vals = [], []
    joint_pos_torso_vals = []
    joint_vel_left_vals, joint_vel_right_vals = [], []
    gripper_left_vals, gripper_right_vals = [], []
    base_velocity_vals, last_action_vals, action_vals = [], [], []
    is_first_vals, is_last_vals = [], []
    variant_idx_vals, segment_idx_vals = [], []
    img_head_vals, img_wrist_left_vals, img_wrist_right_vals = [], [], []
    lang_vals = []

    _encode_jpeg = tf.image.encode_jpeg

    for step in ep_steps:
        obs = step["observation"]
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
        is_first_vals.append(int(step["is_first"]))
        is_last_vals.append(int(step["is_last"]))
        variant_idx_vals.append(step["variant_idx"])
        segment_idx_vals.append(step["segment_idx"])
        img_head_vals.append(_encode_jpeg(obs["image_camera_head"]).numpy())
        img_wrist_left_vals.append(_encode_jpeg(obs["image_camera_wrist_left"]).numpy())
        img_wrist_right_vals.append(_encode_jpeg(obs["image_camera_wrist_right"]).numpy())
        lang_vals.append(step["language_instruction"].encode("utf-8"))

    feature_dict = {
        "steps/observation/joint_position_arm_left": _float_feature(joint_pos_left_vals),
        "steps/observation/joint_position_arm_right": _float_feature(joint_pos_right_vals),
        "steps/observation/joint_position_torso": _float_feature(joint_pos_torso_vals),
        "steps/observation/joint_velocity_arm_left": _float_feature(joint_vel_left_vals),
        "steps/observation/joint_velocity_arm_right": _float_feature(joint_vel_right_vals),
        "steps/observation/gripper_state_left": _float_feature(gripper_left_vals),
        "steps/observation/gripper_state_right": _float_feature(gripper_right_vals),
        "steps/observation/base_velocity": _float_feature(base_velocity_vals),
        "steps/observation/last_action": _float_feature(last_action_vals),
        "steps/observation/image_camera_head": _bytes_feature(img_head_vals),
        "steps/observation/image_camera_wrist_left": _bytes_feature(img_wrist_left_vals),
        "steps/observation/image_camera_wrist_right": _bytes_feature(img_wrist_right_vals),
        "steps/action": _float_feature(action_vals),
        "steps/is_first": _int64_feature(is_first_vals),
        "steps/is_last": _int64_feature(is_last_vals),
        "steps/variant_idx": _int64_feature(variant_idx_vals),
        "steps/segment_idx": _int64_feature(segment_idx_vals),
        "steps/language_instruction": _bytes_feature(lang_vals),
        "episode_metadata/file_path": _bytes_feature(file_path.encode("utf-8")),
    }

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

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


# ============================================================
# STEP 1: Re-run worker_002
# ============================================================
def rerun_worker_002():
    worker_dst = os.path.join(TMP_DIR, f"worker_{WORKER_ID:03d}")

    # Clean up old incomplete data
    if os.path.exists(worker_dst):
        print(f"Cleaning up old worker_002 data: {worker_dst}")
        shutil.rmtree(worker_dst)
    os.makedirs(worker_dst, exist_ok=True)

    split_str = f"train[{PCT_START}%:{PCT_END}%]"
    print(f"\n{'='*60}")
    print(f"Re-running worker_002: split={split_str}  ep_offset={EP_OFFSET}")
    print(f"{'='*60}\n")

    src_builder = tfds.builder_from_directory(SRC_DATA_DIR)
    src_dataset = src_builder.as_dataset(split=split_str)

    t0 = time.time()
    ep_stats_local = []
    wp_map_local = []

    shard_idx = 0
    ep_in_shard = 0
    shard_lengths = []
    writer = None

    def _open_new_shard():
        nonlocal writer, shard_idx, ep_in_shard
        if writer is not None:
            writer.close()
            shard_lengths.append(ep_in_shard)
        shard_path = os.path.join(
            worker_dst,
            f"worker_{WORKER_ID:03d}-train.tfrecord-{shard_idx:05d}")
        writer = tf.io.TFRecordWriter(shard_path)
        shard_idx += 1
        ep_in_shard = 0

    _open_new_shard()

    for local_idx, episode in enumerate(src_dataset):
        global_ep_idx = EP_OFFSET + local_idx
        steps = list(episode["steps"])
        T = len(steps)

        if T == 0:
            print(f"  [W{WORKER_ID}] Episode {global_ep_idx}: empty, skipping", flush=True)
            del steps
            continue

        jp_left = np.array([s["observation"]["joint_position_arm_left"].numpy() for s in steps])
        jp_right = np.array([s["observation"]["joint_position_arm_right"].numpy() for s in steps])
        actions_all = np.array([s["action"].numpy() for s in steps])

        wp_left = set(dp_waypoint_selection_fast(jp_left, ERR_THRESHOLD))
        wp_right = set(dp_waypoint_selection_fast(jp_right, ERR_THRESHOLD))
        wp_gripper = set(detect_gripper_changes(actions_all))

        waypoints = sorted(wp_left | wp_right | wp_gripper)
        wp_indices = sorted(set([0] + waypoints))
        n_wp = len(wp_indices)

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

        del steps, episode, wp_indices

        serialized = _serialize_episode(ep_steps, file_path)
        writer.write(serialized)
        ep_in_shard += 1
        del ep_steps, serialized

        if ep_in_shard >= EPISODES_PER_SHARD:
            _open_new_shard()

        if local_idx % 10 == 0:
            gc.collect()
        if local_idx % 50 == 0 or local_idx == 0:
            elapsed_so_far = time.time() - t0
            eps_per_sec = (local_idx + 1) / elapsed_so_far if elapsed_so_far > 0 else 0
            print(f"  [W{WORKER_ID}] Episode {global_ep_idx}: "
                  f"{T} -> {n_wp} waypoints  "
                  f"({local_idx+1} done, {elapsed_so_far:.0f}s, "
                  f"{eps_per_sec:.2f} ep/s)",
                  flush=True)

    if writer is not None:
        writer.close()
        if ep_in_shard > 0:
            shard_lengths.append(ep_in_shard)

    total_local = local_idx + 1 if 'local_idx' in dir() else 0
    elapsed = time.time() - t0

    stats_path = os.path.join(worker_dst, "_worker_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "ep_stats": ep_stats_local,
            "wp_map": wp_map_local,
            "shard_lengths": shard_lengths,
        }, f)

    print(f"\n[Worker {WORKER_ID}] Done: {total_local} episodes, "
          f"{len(shard_lengths)} shards, {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return total_local


# ============================================================
# STEP 2: Merge all 8 workers
# ============================================================
def merge_all_workers():
    NUM_WORKERS = 8
    dataset_name = "waypoint_filtered_rlds"
    final_output_dir = os.path.join(DST_DATA_DIR, dataset_name, "1.0.0")

    print(f"\n{'='*60}")
    print("Merging all 8 workers...")
    print(f"{'='*60}")

    if os.path.exists(final_output_dir):
        shutil.rmtree(final_output_dir)
    os.makedirs(final_output_dir, exist_ok=True)

    all_shard_lengths = []
    all_shard_files = []

    for wid in range(NUM_WORKERS):
        worker_dir = os.path.join(TMP_DIR, f"worker_{wid:03d}")
        stats_path = os.path.join(worker_dir, "_worker_stats.json")
        if not os.path.exists(stats_path):
            print(f"  ERROR: Worker {wid} has no _worker_stats.json!")
            return False
        with open(stats_path, "r") as f:
            stats = json.load(f)
        worker_shard_lengths = stats.get("shard_lengths", [])
        shard_files = sorted(glob.glob(
            os.path.join(worker_dir, f"worker_{wid:03d}-train.tfrecord-*")))

        ep_count = sum(worker_shard_lengths)
        print(f"  Worker {wid}: {len(shard_files)} shards, {ep_count} episodes")

        if len(shard_files) != len(worker_shard_lengths):
            print(f"    WARNING: {len(shard_files)} files vs "
                  f"{len(worker_shard_lengths)} shard_lengths")

        for sf in shard_files:
            all_shard_files.append(sf)
        all_shard_lengths.extend(worker_shard_lengths)

    total_shards = len(all_shard_files)
    total_episodes = sum(all_shard_lengths)
    print(f"\n  Total: {total_shards} shards, {total_episodes} episodes")

    # Rename and link/copy
    for idx, src_path in enumerate(all_shard_files):
        dst_name = (f"{dataset_name}-train.tfrecord-"
                    f"{idx:05d}-of-{total_shards:05d}")
        dst_path = os.path.join(final_output_dir, dst_name)
        try:
            os.link(src_path, dst_path)
        except OSError:
            shutil.copy2(src_path, dst_path)

    # Write dataset_info.json
    dataset_info = {
        "fileFormat": "tfrecord",
        "moduleName": f"{dataset_name}.{dataset_name}",
        "name": dataset_name,
        "releaseNotes": {
            "1.0.0": f"Waypoint-filtered with err_threshold={ERR_THRESHOLD} "
                     f"on both arms + gripper changes."
        },
        "splits": [{
            "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
            "name": "train",
            "numBytes": "0",
            "shardLengths": [str(x) for x in all_shard_lengths],
        }],
        "version": "1.0.0",
    }
    with open(os.path.join(final_output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)

    # Write features.json
    features = {
        "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
        "featuresDict": {"features": {
            "steps": {
                "pythonClassName": "tensorflow_datasets.core.features.dataset_feature.Dataset",
                "sequence": {"feature": {
                    "pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict",
                    "featuresDict": {"features": {
                        "is_last": {"pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar", "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"}, "description": "True on last step of the episode."},
                        "language_instruction": {"pythonClassName": "tensorflow_datasets.core.features.text_feature.Text", "text": {}, "description": "Language Instruction."},
                        "variant_idx": {"pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar", "tensor": {"shape": {}, "dtype": "int32", "encoding": "none"}, "description": "Bowl order index."},
                        "observation": {"pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict", "featuresDict": {"features": {
                            "last_action": {"pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor", "tensor": {"shape": {"dimensions": ["26"]}, "dtype": "float32", "encoding": "none"}, "description": "Last robot action."},
                            "joint_position_arm_right": {"pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor", "tensor": {"shape": {"dimensions": ["6"]}, "dtype": "float32", "encoding": "none"}, "description": "Right arm joint positions."},
                            "image_camera_wrist_right": {"pythonClassName": "tensorflow_datasets.core.features.image_feature.Image", "image": {"shape": {"dimensions": ["224", "224", "3"]}, "dtype": "uint8", "encodingFormat": "jpeg"}, "description": "RGB image from right wrist camera."},
                            "gripper_state_right": {"pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor", "tensor": {"shape": {"dimensions": ["1"]}, "dtype": "float32", "encoding": "none"}, "description": "Right gripper state."},
                            "joint_velocity_arm_right": {"pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor", "tensor": {"shape": {"dimensions": ["6"]}, "dtype": "float32", "encoding": "none"}, "description": "Right arm joint velocities."},
                            "image_camera_wrist_left": {"pythonClassName": "tensorflow_datasets.core.features.image_feature.Image", "image": {"shape": {"dimensions": ["224", "224", "3"]}, "dtype": "uint8", "encodingFormat": "jpeg"}, "description": "RGB image from left wrist camera."},
                            "image_camera_head": {"pythonClassName": "tensorflow_datasets.core.features.image_feature.Image", "image": {"shape": {"dimensions": ["224", "224", "3"]}, "dtype": "uint8", "encodingFormat": "jpeg"}, "description": "RGB image from head camera."},
                            "joint_position_torso": {"pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor", "tensor": {"shape": {"dimensions": ["4"]}, "dtype": "float32", "encoding": "none"}, "description": "Torso joint positions."},
                            "gripper_state_left": {"pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor", "tensor": {"shape": {"dimensions": ["1"]}, "dtype": "float32", "encoding": "none"}, "description": "Left gripper state."},
                            "joint_velocity_arm_left": {"pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor", "tensor": {"shape": {"dimensions": ["6"]}, "dtype": "float32", "encoding": "none"}, "description": "Left arm joint velocities."},
                            "joint_position_arm_left": {"pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor", "tensor": {"shape": {"dimensions": ["6"]}, "dtype": "float32", "encoding": "none"}, "description": "Left arm joint positions."},
                            "base_velocity": {"pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor", "tensor": {"shape": {"dimensions": ["3"]}, "dtype": "float32", "encoding": "none"}, "description": "Base velocity."},
                        }}},
                        "is_first": {"pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar", "tensor": {"shape": {}, "dtype": "bool", "encoding": "none"}, "description": "True on first step of the episode."},
                        "segment_idx": {"pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar", "tensor": {"shape": {}, "dtype": "int32", "encoding": "none"}, "description": "Segment index."},
                        "action": {"pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor", "tensor": {"shape": {"dimensions": ["26"]}, "dtype": "float32", "encoding": "none"}, "description": "Robot action."},
                    }}
                }, "length": "-1"}
            },
            "episode_metadata": {"pythonClassName": "tensorflow_datasets.core.features.features_dict.FeaturesDict", "featuresDict": {"features": {
                "file_path": {"pythonClassName": "tensorflow_datasets.core.features.text_feature.Text", "text": {}, "description": "Path to the original data file."}
            }}}
        }}
    }
    with open(os.path.join(final_output_dir, "features.json"), "w") as f:
        json.dump(features, f, indent=4)

    # Merge waypoint_indices.json
    all_wp_map = []
    for wid in range(NUM_WORKERS):
        worker_dir = os.path.join(TMP_DIR, f"worker_{wid:03d}")
        stats_path = os.path.join(worker_dir, "_worker_stats.json")
        with open(stats_path, "r") as f:
            data = json.load(f)
        all_wp_map.extend(data.get("wp_map", []))

    all_wp_map.sort(key=lambda x: x["src_ep_idx"])

    wp_index_data = {
        "config": {
            "err_threshold": ERR_THRESHOLD,
            "extraction_key": "joint_position_arm_left + joint_position_arm_right + gripper_changes",
            "src_data_dir": SRC_DATA_DIR,
            "dst_data_dir": DST_DATA_DIR,
            "total_episodes": len(all_wp_map),
            "total_src_steps": sum(ep["original_steps"] for ep in all_wp_map),
            "total_wp_steps": sum(ep["waypoint_steps"] for ep in all_wp_map),
        },
        "episodes": all_wp_map,
    }
    wp_index_path = os.path.join(DST_DATA_DIR, "waypoint_indices.json")
    with open(wp_index_path, "w", encoding="utf-8") as f:
        json.dump(wp_index_data, f, indent=2, ensure_ascii=False)

    total_src = sum(ep["original_steps"] for ep in all_wp_map)
    total_wp = sum(ep["waypoint_steps"] for ep in all_wp_map)
    ratio = total_src / max(total_wp, 1)

    print(f"\n  Merged: {total_shards} shards, {total_episodes} episodes")
    print(f"  Waypoint compression: {total_src} steps -> {total_wp} waypoints "
          f"({ratio:.1f}x)")
    print(f"  Output: {final_output_dir}")
    print(f"  Waypoint indices: {wp_index_path}")
    return True


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("STEP 1: Re-running worker_002 (train[25%:38%])")
    print("="*60)

    t_start = time.time()
    n_episodes = rerun_worker_002()
    t_rerun = time.time() - t_start
    print(f"\nWorker_002 re-run completed in {t_rerun:.1f}s ({t_rerun/60:.1f} min)")

    print("\n" + "="*60)
    print("STEP 2: Merging all 8 workers into final dataset")
    print("="*60)

    t_merge_start = time.time()
    success = merge_all_workers()
    t_merge = time.time() - t_merge_start

    if success:
        t_total = time.time() - t_start
        final_output_dir = os.path.join(DST_DATA_DIR, "waypoint_filtered_rlds", "1.0.0")
        print(f"\n{'='*60}")
        print(f"ALL DONE!")
        print(f"  Re-run:  {t_rerun:.1f}s ({t_rerun/60:.1f} min)")
        print(f"  Merge:   {t_merge:.1f}s")
        print(f"  Total:   {t_total:.1f}s ({t_total/60:.1f} min)")
        print(f"{'='*60}")
        print(f"\nDataset saved at:\n  {final_output_dir}")
        print(f"\nTo load:")
        print(f'  builder = tfds.builder_from_directory(r"{final_output_dir}")')
        print(f'  dataset = builder.as_dataset(split="train")')
    else:
        print("\nERROR: Merge failed! Check worker outputs above.")
