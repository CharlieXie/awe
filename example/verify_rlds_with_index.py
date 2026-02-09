"""
verify_with_indices.py
======================
Verify that the waypoint-filtered RLDS dataset contains exactly the
correct data from the original RLDS dataset, using saved waypoint indices.

For each episode:
  - Match source and destination by file_path (order-independent)
  - Use waypoint_indices[i] to index into the source episode
  - Compare all features: tensors (allclose), images (exact), action, text

Usage:
    python verify_with_indices.py
"""

import tensorflow_datasets as tfds
import numpy as np
import json
import os
import time

# ============================================================
# Configuration
# ============================================================
SRC_DATA_DIR = r"C:\Users\chuanlia\Documents\learning_space\ntu\projects\awe\example\rlds_data"
DST_DATA_DIR = r"C:\Users\chuanlia\Documents\learning_space\ntu\projects\awe\example\rlds_data_waypoint"
WP_INDEX_PATH = os.path.join(DST_DATA_DIR, "waypoint_indices.json")
DST_RLDS_DIR  = os.path.join(DST_DATA_DIR, "waypoint_filtered_rlds", "1.0.0")

# ============================================================
# 1. Load waypoint indices
# ============================================================
print(f"Loading waypoint indices from: {WP_INDEX_PATH}")
with open(WP_INDEX_PATH, "r", encoding="utf-8") as f:
    wp_data = json.load(f)

config = wp_data["config"]
episodes_info = wp_data["episodes"]
print(f"  err_threshold = {config['err_threshold']}")
print(f"  extraction_key = {config['extraction_key']}")
print(f"  {config['total_episodes']} episodes, "
      f"{config['total_src_steps']} src steps -> {config['total_wp_steps']} wp steps\n")

# Build file_path -> wp_info lookup
wp_by_filepath = {}
for ep_info in episodes_info:
    fp = ep_info["file_path"]
    wp_by_filepath[fp] = ep_info

# ============================================================
# 2. Phase 1: Scan source dataset, cache ONLY waypoint steps
#    (memory-efficient: ~3600 steps instead of ~38000)
# ============================================================
print("=" * 70)
print("Phase 1: Building source cache at waypoint positions...")
print("=" * 70)

TENSOR_OBS_KEYS = [
    "joint_position_arm_left",
    "joint_position_arm_right",
    "joint_position_torso",
    "joint_velocity_arm_left",
    "joint_velocity_arm_right",
    "gripper_state_left",
    "gripper_state_right",
    "base_velocity",
    "last_action",
]

IMAGE_OBS_KEYS = [
    "image_camera_head",
    "image_camera_wrist_left",
    "image_camera_wrist_right",
    # Uncomment if depth images are included in destination:
    # "depth_camera_head",
    # "depth_camera_wrist_left",
    # "depth_camera_wrist_right",
]

src_builder = tfds.builder_from_directory(SRC_DATA_DIR)
src_dataset = src_builder.as_dataset(split="train")

# src_cache: file_path -> list of cached step dicts (only at waypoint positions)
src_cache = {}
t0 = time.time()

for ep_idx, episode in enumerate(src_dataset):
    fp = episode["episode_metadata"]["file_path"].numpy().decode("utf-8")

    # Skip episodes not in waypoint index
    if fp not in wp_by_filepath:
        continue

    wp_info = wp_by_filepath[fp]
    wp_indices = wp_info["waypoint_indices"]

    steps = list(episode["steps"])

    cached_steps = []
    for orig_idx in wp_indices:
        if orig_idx >= len(steps):
            # Out of range — will be caught in verification phase
            cached_steps.append(None)
            continue

        step = steps[orig_idx]
        obs = step["observation"]

        cached = {
            "observation": {},
            "action": step["action"].numpy(),
            "language_instruction": step["language_instruction"].numpy().decode("utf-8"),
            "variant_idx": int(step["variant_idx"].numpy()),
            "segment_idx": int(step["segment_idx"].numpy()),
        }
        for key in TENSOR_OBS_KEYS:
            cached["observation"][key] = obs[key].numpy()
        for key in IMAGE_OBS_KEYS:
            cached["observation"][key] = obs[key].numpy()

        cached_steps.append(cached)

    src_cache[fp] = cached_steps
    del steps  # Free memory early

    if (ep_idx + 1) % 50 == 0:
        print(f"  Scanned {ep_idx + 1} source episodes, "
              f"cached {len(src_cache)} with waypoints...")

elapsed = time.time() - t0
total_cached_steps = sum(len(v) for v in src_cache.values())
print(f"  Done: cached {total_cached_steps} waypoint steps "
      f"from {len(src_cache)} episodes in {elapsed:.1f}s\n")

# ============================================================
# 3. Phase 2: Iterate destination dataset, compare with cache
# ============================================================
print("=" * 70)
print("Phase 2: Verifying destination dataset against source cache...")
print("=" * 70)

dst_builder = tfds.builder_from_directory(DST_RLDS_DIR)
dst_dataset = dst_builder.as_dataset(split="train")

n_episodes_checked = 0
n_steps_checked = 0
n_errors = 0
error_details = []

for ep_idx, dst_episode in enumerate(dst_dataset):
    fp = dst_episode["episode_metadata"]["file_path"].numpy().decode("utf-8")
    dst_steps = list(dst_episode["steps"])

    # --- Find waypoint info ---
    if fp not in wp_by_filepath:
        msg = f"Episode {ep_idx}: file_path not found in waypoint index!"
        print(f"  [ERROR] {msg}")
        error_details.append(msg)
        n_errors += 1
        continue

    wp_info = wp_by_filepath[fp]
    wp_indices = wp_info["waypoint_indices"]

    # --- Find cached source steps ---
    if fp not in src_cache:
        msg = f"Episode {ep_idx}: file_path not found in source cache!"
        print(f"  [ERROR] {msg}")
        error_details.append(msg)
        n_errors += 1
        continue

    cached_src_steps = src_cache[fp]

    # --- Verify step count ---
    if len(dst_steps) != len(wp_indices):
        msg = (f"Episode {ep_idx}: step count mismatch! "
               f"dst={len(dst_steps)}, wp_indices={len(wp_indices)}")
        print(f"  [ERROR] {msg}")
        error_details.append(msg)
        n_errors += 1
        continue

    if len(dst_steps) != len(cached_src_steps):
        msg = (f"Episode {ep_idx}: cached step count mismatch! "
               f"dst={len(dst_steps)}, cached={len(cached_src_steps)}")
        print(f"  [ERROR] {msg}")
        error_details.append(msg)
        n_errors += 1
        continue

    # --- Compare each step ---
    ep_errors = 0
    for step_i in range(len(dst_steps)):
        orig_idx = wp_indices[step_i]
        src_step = cached_src_steps[step_i]
        dst_step = dst_steps[step_i]

        if src_step is None:
            msg = (f"Ep {ep_idx} step {step_i}: "
                   f"waypoint index {orig_idx} was out of range in source")
            error_details.append(msg)
            n_errors += 1
            ep_errors += 1
            continue

        dst_obs = dst_step["observation"]
        src_obs = src_step["observation"]

        # ---- Check tensor observation features ----
        for key in TENSOR_OBS_KEYS:
            dst_val = dst_obs[key].numpy()
            src_val = src_obs[key]
            if not np.allclose(dst_val, src_val, atol=1e-7):
                max_diff = np.max(np.abs(dst_val - src_val))
                msg = (f"Ep {ep_idx} step {step_i} (src[{orig_idx}]): "
                       f"obs/{key} MISMATCH (max_diff={max_diff:.2e})")
                error_details.append(msg)
                n_errors += 1
                ep_errors += 1

        # ---- Check image observation features ----
        for key in IMAGE_OBS_KEYS:
            dst_img = dst_obs[key].numpy()
            src_img = src_obs[key]
            if not np.array_equal(dst_img, src_img):
                n_diff_pixels = int(np.sum(dst_img != src_img))
                max_diff = int(np.max(np.abs(
                    dst_img.astype(np.int32) - src_img.astype(np.int32)
                )))
                msg = (f"Ep {ep_idx} step {step_i} (src[{orig_idx}]): "
                       f"obs/{key} IMAGE MISMATCH "
                       f"({n_diff_pixels} pixels differ, max_diff={max_diff})")
                error_details.append(msg)
                n_errors += 1
                ep_errors += 1

        # ---- Check action ----
        dst_action = dst_step["action"].numpy()
        src_action = src_step["action"]
        if not np.allclose(dst_action, src_action, atol=1e-7):
            max_diff = np.max(np.abs(dst_action - src_action))
            msg = (f"Ep {ep_idx} step {step_i} (src[{orig_idx}]): "
                   f"action MISMATCH (max_diff={max_diff:.2e})")
            error_details.append(msg)
            n_errors += 1
            ep_errors += 1

        # ---- Check language_instruction ----
        dst_lang = dst_step["language_instruction"].numpy().decode("utf-8")
        src_lang = src_step["language_instruction"]
        if dst_lang != src_lang:
            msg = (f"Ep {ep_idx} step {step_i} (src[{orig_idx}]): "
                   f"language_instruction MISMATCH "
                   f"'{dst_lang[:30]}...' vs '{src_lang[:30]}...'")
            error_details.append(msg)
            n_errors += 1
            ep_errors += 1

        # ---- Check variant_idx / segment_idx ----
        dst_variant = int(dst_step["variant_idx"].numpy())
        src_variant = src_step["variant_idx"]
        if dst_variant != src_variant:
            msg = (f"Ep {ep_idx} step {step_i} (src[{orig_idx}]): "
                   f"variant_idx MISMATCH {dst_variant} vs {src_variant}")
            error_details.append(msg)
            n_errors += 1
            ep_errors += 1

        dst_segment = int(dst_step["segment_idx"].numpy())
        src_segment = src_step["segment_idx"]
        if dst_segment != src_segment:
            msg = (f"Ep {ep_idx} step {step_i} (src[{orig_idx}]): "
                   f"segment_idx MISMATCH {dst_segment} vs {src_segment}")
            error_details.append(msg)
            n_errors += 1
            ep_errors += 1

        n_steps_checked += 1

    n_episodes_checked += 1
    status = "PASS" if ep_errors == 0 else f"FAIL ({ep_errors} errors)"

    # Print progress: every 20 episodes, or if there are errors
    if n_episodes_checked % 20 == 0 or ep_errors > 0:
        print(f"  Episode {ep_idx} [{fp[-40:]}]: "
              f"{len(dst_steps)} steps, wp_indices[0:3]={wp_indices[:3]}... "
              f"[{status}]")

# ============================================================
# 4. Summary
# ============================================================
print(f"\n{'='*70}")
print(f"Verification Complete")
print(f"{'='*70}")
print(f"  Episodes checked:  {n_episodes_checked}")
print(f"  Steps checked:     {n_steps_checked}")
print(f"  Features per step: {len(TENSOR_OBS_KEYS)} tensors + "
      f"{len(IMAGE_OBS_KEYS)} images + action + text + scalars")
print(f"  Errors found:      {n_errors}")

if n_errors == 0:
    print(f"\n  >>> RESULT: ALL PASSED <<<")
    print(f"  All {n_steps_checked} waypoint steps across "
          f"{n_episodes_checked} episodes match the original data exactly.")
else:
    print(f"\n  >>> RESULT: FAILED ({n_errors} errors) <<<")
    print(f"\n  First {min(len(error_details), 30)} error details:")
    for i, detail in enumerate(error_details[:30]):
        print(f"    {i+1}. {detail}")
    if len(error_details) > 30:
        print(f"    ... and {len(error_details) - 30} more errors")
print(f"{'='*70}")