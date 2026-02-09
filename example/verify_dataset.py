"""
verify_dataset.py
==================
Comprehensive verification of the waypoint-filtered RLDS dataset.

5 levels of checks:
  1. JSON metadata: episode count from dataset_info.json
  2. Waypoint indices: completeness from waypoint_indices.json
  3. tfds loadability: iterate all episodes, count them
  4. Sampled content: shape/dtype/flag checks on sampled episodes
  5. Cross-validation: compare waypoint steps against source dataset

Usage (inside awe_venv):
    conda activate awe_venv
    cd /workspace/awe/example
    python verify_dataset.py
"""

import sys
import os
import json
import time
import numpy as np

# Paths
NEW_DATASET_DIR = "rlds_part5/waypoint_filtered_rlds/1.0.0"
WP_INDICES_PATH = "rlds_part5/waypoint_indices.json"
SRC_DATASET_DIR = "/workspace/galaxea_data/rlds/part5_r1_lite/1.0.0"
EXPECTED_EPISODES = 16218

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  -- {detail}")


# ============================================================
# Level 1: JSON metadata check (no TF needed, instant)
# ============================================================
print(f"\n{'='*60}")
print("Level 1: dataset_info.json metadata")
print(f"{'='*60}")

info_path = os.path.join(NEW_DATASET_DIR, "dataset_info.json")
features_path = os.path.join(NEW_DATASET_DIR, "features.json")

check("dataset_info.json exists", os.path.exists(info_path))
check("features.json exists", os.path.exists(features_path))

with open(info_path) as f:
    info = json.load(f)

shard_lengths = [int(x) for x in info["splits"][0]["shardLengths"]]
total_from_shards = sum(shard_lengths)
num_shards = len(shard_lengths)

print(f"  Shards: {num_shards}, Total episodes: {total_from_shards}")
check("Episode count matches expected",
      total_from_shards == EXPECTED_EPISODES,
      f"got {total_from_shards}, expected {EXPECTED_EPISODES}")
check("All shardLengths > 0",
      all(s > 0 for s in shard_lengths),
      f"found zero-length shards")


# ============================================================
# Level 2: waypoint_indices.json completeness
# ============================================================
print(f"\n{'='*60}")
print("Level 2: waypoint_indices.json completeness")
print(f"{'='*60}")

check("waypoint_indices.json exists", os.path.exists(WP_INDICES_PATH))

with open(WP_INDICES_PATH) as f:
    wp_data = json.load(f)

wp_episodes = wp_data["episodes"]
wp_total = wp_data["config"]["total_episodes"]
print(f"  Episodes in waypoint_indices: {wp_total}")
print(f"  Total source steps: {wp_data['config']['total_src_steps']}")
print(f"  Total waypoint steps: {wp_data['config']['total_wp_steps']}")

check("Waypoint episode count matches",
      wp_total == EXPECTED_EPISODES,
      f"got {wp_total}")

# Check episode index continuity
ep_indices = sorted([ep["src_ep_idx"] for ep in wp_episodes])
expected_indices = list(range(EXPECTED_EPISODES))
missing = set(expected_indices) - set(ep_indices)
duplicates = len(ep_indices) - len(set(ep_indices))

check("No missing episode indices",
      len(missing) == 0,
      f"missing {len(missing)}: {list(missing)[:20]}...")
check("No duplicate episode indices",
      duplicates == 0,
      f"{duplicates} duplicates found")

# Check every episode has waypoint_indices with index 0
for ep in wp_episodes[:100]:  # spot check first 100
    if 0 not in ep["waypoint_indices"]:
        check("All episodes include start index 0", False,
              f"ep {ep['src_ep_idx']} missing index 0")
        break
else:
    check("All sampled episodes include start index 0 (first 100)", True)

# Compression stats
ratios = [ep["original_steps"] / max(ep["waypoint_steps"], 1) for ep in wp_episodes]
print(f"  Compression ratio: min={min(ratios):.1f}x, "
      f"max={max(ratios):.1f}x, "
      f"mean={np.mean(ratios):.1f}x, "
      f"median={np.median(ratios):.1f}x")


# ============================================================
# Level 3: tfds loadability (iterate all episodes)
# ============================================================
print(f"\n{'='*60}")
print("Level 3: tfds full iteration")
print(f"{'='*60}")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds

t0 = time.time()
builder = tfds.builder_from_directory(NEW_DATASET_DIR)
dataset = builder.as_dataset(split="train")

ep_count = 0
step_counts = []
errors = []

for ep_idx, episode in enumerate(dataset):
    try:
        steps = list(episode["steps"])
        n = len(steps)
        step_counts.append(n)
        ep_count += 1

        if ep_idx % 2000 == 0:
            print(f"  Iterating... ep {ep_idx} ({n} steps)", flush=True)
    except Exception as e:
        errors.append((ep_idx, str(e)))
        if len(errors) <= 5:
            print(f"  ERROR at ep {ep_idx}: {e}")

t_iter = time.time() - t0
print(f"  Iterated {ep_count} episodes in {t_iter:.1f}s ({t_iter/60:.1f} min)")

check("All episodes loadable (no errors)",
      len(errors) == 0,
      f"{len(errors)} errors")
check("Episode count matches expected",
      ep_count == EXPECTED_EPISODES,
      f"got {ep_count}")
check("No empty episodes",
      all(s > 0 for s in step_counts),
      f"found {sum(1 for s in step_counts if s == 0)} empty episodes")

print(f"  Steps per episode: min={min(step_counts)}, max={max(step_counts)}, "
      f"mean={np.mean(step_counts):.1f}, median={np.median(step_counts):.0f}")


# ============================================================
# Level 4: Sampled content checks (every 2000th episode)
# ============================================================
print(f"\n{'='*60}")
print("Level 4: Sampled content validation")
print(f"{'='*60}")

dataset = builder.as_dataset(split="train")
sample_indices = list(range(0, EXPECTED_EPISODES, 2000))
level4_ok = True

for ep_idx, episode in enumerate(dataset):
    if ep_idx not in sample_indices:
        continue

    steps = list(episode["steps"])
    n = len(steps)
    first = steps[0]
    last = steps[-1]
    obs = first["observation"]

    # is_first / is_last flags
    if not first["is_first"].numpy():
        check(f"Ep {ep_idx}: is_first flag", False)
        level4_ok = False
    if not last["is_last"].numpy():
        check(f"Ep {ep_idx}: is_last flag", False)
        level4_ok = False

    # Image checks
    for img_key in ["image_camera_head", "image_camera_wrist_left", "image_camera_wrist_right"]:
        img = obs[img_key].numpy()
        if img.shape != (224, 224, 3):
            check(f"Ep {ep_idx}: {img_key} shape", False, f"got {img.shape}")
            level4_ok = False
        if img.dtype != np.uint8:
            check(f"Ep {ep_idx}: {img_key} dtype", False, f"got {img.dtype}")
            level4_ok = False

    # Tensor shape checks
    shape_checks = {
        "joint_position_arm_left": (6,),
        "joint_position_arm_right": (6,),
        "joint_position_torso": (4,),
        "joint_velocity_arm_left": (6,),
        "joint_velocity_arm_right": (6,),
        "gripper_state_left": (1,),
        "gripper_state_right": (1,),
        "base_velocity": (3,),
        "last_action": (26,),
    }
    for key, expected_shape in shape_checks.items():
        actual_shape = tuple(obs[key].shape)
        if actual_shape != expected_shape:
            check(f"Ep {ep_idx}: obs/{key} shape", False,
                  f"got {actual_shape}, expected {expected_shape}")
            level4_ok = False

    if tuple(first["action"].shape) != (26,):
        check(f"Ep {ep_idx}: action shape", False)
        level4_ok = False

    # NaN/Inf checks
    action_vals = first["action"].numpy()
    if np.any(np.isnan(action_vals)) or np.any(np.isinf(action_vals)):
        check(f"Ep {ep_idx}: action NaN/Inf", False)
        level4_ok = False

    print(f"  Ep {ep_idx}: {n} steps, img_mean={obs['image_camera_head'].numpy().mean():.1f}, OK")

    if ep_idx >= sample_indices[-1]:
        break

check("All sampled episodes: shapes, dtypes, flags correct", level4_ok)


# ============================================================
# Level 5: Cross-validation against source dataset (3 episodes)
# ============================================================
print(f"\n{'='*60}")
print("Level 5: Cross-validation against source dataset")
print(f"{'='*60}")

cross_check_indices = [0, 5000, 10000]
src_builder = tfds.builder_from_directory(SRC_DATASET_DIR)

level5_ok = True
for target_idx in cross_check_indices:
    print(f"  Checking episode {target_idx}...", flush=True)

    # Load from new dataset
    new_ds = builder.as_dataset(split="train")
    new_ep = None
    for i, ep in enumerate(new_ds):
        if i == target_idx:
            new_ep = ep
            break
    new_steps = list(new_ep["steps"])

    # Load from source dataset
    src_ds = src_builder.as_dataset(split="train")
    src_ep = None
    for i, ep in enumerate(src_ds):
        if i == target_idx:
            src_ep = ep
            break
    src_steps = list(src_ep["steps"])

    # Get waypoint indices
    wp_indices = wp_data["episodes"][target_idx]["waypoint_indices"]

    if len(new_steps) != len(wp_indices):
        check(f"Ep {target_idx}: step count matches wp_indices",
              False, f"{len(new_steps)} vs {len(wp_indices)}")
        level5_ok = False
        continue

    # Compare action values at waypoint positions
    mismatches = 0
    for i, orig_i in enumerate(wp_indices):
        src_action = src_steps[orig_i]["action"].numpy()
        new_action = new_steps[i]["action"].numpy()
        if not np.allclose(src_action, new_action, atol=1e-5):
            mismatches += 1
            if mismatches <= 3:
                diff = np.abs(src_action - new_action).max()
                print(f"    Mismatch at step {i} (orig {orig_i}): max_diff={diff}")

        # Also compare a joint position
        src_jp = src_steps[orig_i]["observation"]["joint_position_arm_left"].numpy()
        new_jp = new_steps[i]["observation"]["joint_position_arm_left"].numpy()
        if not np.allclose(src_jp, new_jp, atol=1e-5):
            mismatches += 1

    if mismatches > 0:
        level5_ok = False
        check(f"Ep {target_idx}: data matches source", False,
              f"{mismatches} mismatches")
    else:
        print(f"    Ep {target_idx}: all {len(wp_indices)} waypoint steps match source")

check("Cross-validation against source", level5_ok)


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"VERIFICATION SUMMARY")
print(f"{'='*60}")
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
if failed == 0:
    print(f"\n  ALL CHECKS PASSED! Dataset is valid.")
else:
    print(f"\n  WARNING: {failed} check(s) failed. Review output above.")
print(f"{'='*60}")