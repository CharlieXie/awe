"""
verify_with_indices.py
======================
Verify that the waypoint-filtered RLDS dataset contains exactly the
correct data from the original RLDS dataset, using saved waypoint indices.

For each episode:
  - Match source and destination by file_path (order-independent)
  - Use waypoint_indices[i] to index into the source episode
  - Compare all features: tensors (allclose), images (exact), action, text

Multiprocessing version: parallelizes both source caching (Phase 1) and
verification (Phase 2) using TFDS percentage-based subsplits.

Usage:
    python verify_with_indices.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow_datasets as tfds
import numpy as np
import json

import time
import multiprocessing as mp

# ============================================================
# Configuration
# ============================================================
SRC_DATA_DIR  = r"/workspace/awe/example/part1_r1_lite_compressed"
DST_DATA_DIR  = r"/workspace/awe/example/part1_r1_lite_compressed_wp_0008"

WP_INDEX_PATH = os.path.join(DST_DATA_DIR, "waypoint_indices.json")
DST_RLDS_DIR  = os.path.join(DST_DATA_DIR, "waypoint_filtered_rlds", "1.0.0")

NUM_WORKERS = 32

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
]

# ============================================================
# Module-level globals for Phase 2 workers
# On Linux (fork), child processes inherit these via COW —
# no serialization, no extra memory until written.
# ============================================================
_g_src_cache = None
_g_wp_by_filepath = None


# ============================================================
# Helpers
# ============================================================
def _generate_splits(num_workers):
    """Generate TFDS percentage-based split strings for each worker."""
    splits = []
    chunk = 100 // num_workers
    for i in range(num_workers):
        start = i * chunk
        end = (i + 1) * chunk if i < num_workers - 1 else 100
        splits.append(f"train[{start}%:{end}%]")
    return splits


# ============================================================
# Phase 1 worker: build source cache
# ============================================================
def _build_cache_worker(args):
    """Read a split of the source dataset, return cached waypoint steps."""
    worker_id, split_str, src_data_dir, wp_by_filepath = args

    builder = tfds.builder_from_directory(src_data_dir)
    dataset = builder.as_dataset(split=split_str)

    local_cache = {}
    ep_count = 0

    for episode in dataset:
        fp = episode["episode_metadata"]["file_path"].numpy().decode("utf-8")
        ep_count += 1

        if fp not in wp_by_filepath:
            continue

        wp_info = wp_by_filepath[fp]
        wp_indices = wp_info["waypoint_indices"]
        steps = list(episode["steps"])

        cached_steps = []
        for orig_idx in wp_indices:
            if orig_idx >= len(steps):
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

        local_cache[fp] = cached_steps
        del steps

    print(f"  [Worker {worker_id}] {split_str}: scanned {ep_count} eps, "
          f"cached {len(local_cache)} with waypoints", flush=True)
    return local_cache


# ============================================================
# Phase 2 worker: verify destination against source cache
# ============================================================
def _verify_worker(args):
    """Verify a split of the destination dataset against the source cache."""
    worker_id, split_str, dst_rlds_dir = args

    # Inherited from parent via fork (COW)
    src_cache = _g_src_cache
    wp_by_filepath = _g_wp_by_filepath

    builder = tfds.builder_from_directory(dst_rlds_dir)
    dataset = builder.as_dataset(split=split_str)

    episodes_checked = 0
    steps_checked = 0
    errors = []

    for dst_episode in dataset:
        fp = dst_episode["episode_metadata"]["file_path"].numpy().decode("utf-8")
        dst_steps = list(dst_episode["steps"])

        # --- Find waypoint info ---
        if fp not in wp_by_filepath:
            errors.append(f"Episode [{fp}]: file_path not in waypoint index!")
            continue

        wp_info = wp_by_filepath[fp]
        wp_indices = wp_info["waypoint_indices"]

        # --- Find cached source steps ---
        if fp not in src_cache:
            errors.append(f"Episode [{fp}]: file_path not in source cache!")
            continue

        cached_src_steps = src_cache[fp]

        # --- Verify step count ---
        if len(dst_steps) != len(wp_indices):
            errors.append(
                f"Episode [{fp}]: step count mismatch! "
                f"dst={len(dst_steps)}, wp_indices={len(wp_indices)}")
            continue

        if len(dst_steps) != len(cached_src_steps):
            errors.append(
                f"Episode [{fp}]: cached step count mismatch! "
                f"dst={len(dst_steps)}, cached={len(cached_src_steps)}")
            continue

        # --- Compare each step ---
        ep_errors = 0
        for step_i in range(len(dst_steps)):
            orig_idx = wp_indices[step_i]
            src_step = cached_src_steps[step_i]
            dst_step = dst_steps[step_i]

            if src_step is None:
                errors.append(
                    f"Ep [{fp}] step {step_i}: "
                    f"waypoint index {orig_idx} was out of range in source")
                ep_errors += 1
                continue

            dst_obs = dst_step["observation"]
            src_obs = src_step["observation"]

            # ---- Tensor observations ----
            for key in TENSOR_OBS_KEYS:
                dst_val = dst_obs[key].numpy()
                src_val = src_obs[key]
                if not np.allclose(dst_val, src_val, atol=1e-7):
                    max_diff = np.max(np.abs(dst_val - src_val))
                    errors.append(
                        f"Ep [{fp}] step {step_i} (src[{orig_idx}]): "
                        f"obs/{key} MISMATCH (max_diff={max_diff:.2e})")
                    ep_errors += 1

            # ---- Image observations ----
            for key in IMAGE_OBS_KEYS:
                dst_img = dst_obs[key].numpy()
                src_img = src_obs[key]
                if not np.array_equal(dst_img, src_img):
                    n_diff_pixels = int(np.sum(dst_img != src_img))
                    max_diff = int(np.max(np.abs(
                        dst_img.astype(np.int32) - src_img.astype(np.int32))))
                    errors.append(
                        f"Ep [{fp}] step {step_i} (src[{orig_idx}]): "
                        f"obs/{key} IMAGE MISMATCH "
                        f"({n_diff_pixels} pixels differ, max_diff={max_diff})")
                    ep_errors += 1

            # ---- Action ----
            dst_action = dst_step["action"].numpy()
            src_action = src_step["action"]
            if not np.allclose(dst_action, src_action, atol=1e-7):
                max_diff = np.max(np.abs(dst_action - src_action))
                errors.append(
                    f"Ep [{fp}] step {step_i} (src[{orig_idx}]): "
                    f"action MISMATCH (max_diff={max_diff:.2e})")
                ep_errors += 1

            # ---- Language instruction ----
            dst_lang = dst_step["language_instruction"].numpy().decode("utf-8")
            src_lang = src_step["language_instruction"]
            if dst_lang != src_lang:
                errors.append(
                    f"Ep [{fp}] step {step_i} (src[{orig_idx}]): "
                    f"language_instruction MISMATCH "
                    f"'{dst_lang[:30]}...' vs '{src_lang[:30]}...'")
                ep_errors += 1

            # ---- variant_idx / segment_idx ----
            dst_variant = int(dst_step["variant_idx"].numpy())
            src_variant = src_step["variant_idx"]
            if dst_variant != src_variant:
                errors.append(
                    f"Ep [{fp}] step {step_i} (src[{orig_idx}]): "
                    f"variant_idx MISMATCH {dst_variant} vs {src_variant}")
                ep_errors += 1

            dst_segment = int(dst_step["segment_idx"].numpy())
            src_segment = src_step["segment_idx"]
            if dst_segment != src_segment:
                errors.append(
                    f"Ep [{fp}] step {step_i} (src[{orig_idx}]): "
                    f"segment_idx MISMATCH {dst_segment} vs {src_segment}")
                ep_errors += 1

            steps_checked += 1

        episodes_checked += 1
        status = "PASS" if ep_errors == 0 else f"FAIL ({ep_errors} errors)"
        if episodes_checked % 20 == 0 or ep_errors > 0:
            print(f"  [Worker {worker_id}] Episode [{fp[-40:]}]: "
                  f"{len(dst_steps)} steps [{status}]", flush=True)

    print(f"  [Worker {worker_id}] Done: {episodes_checked} eps, "
          f"{steps_checked} steps, {len(errors)} errors", flush=True)

    return {
        "episodes_checked": episodes_checked,
        "steps_checked": steps_checked,
        "errors": errors,
    }


# ============================================================
# Main
# ============================================================
def main():
    global _g_src_cache, _g_wp_by_filepath

    # --------------------------------------------------------
    # 1. Load waypoint indices
    # --------------------------------------------------------
    print(f"Loading waypoint indices from: {WP_INDEX_PATH}")
    with open(WP_INDEX_PATH, "r", encoding="utf-8") as f:
        wp_data = json.load(f)

    config = wp_data["config"]
    episodes_info = wp_data["episodes"]
    print(f"  err_threshold = {config['err_threshold']}")
    print(f"  extraction_key = {config['extraction_key']}")
    print(f"  {config['total_episodes']} episodes, "
          f"{config['total_src_steps']} src steps -> {config['total_wp_steps']} wp steps")
    print(f"  Using {NUM_WORKERS} workers\n")

    wp_by_filepath = {ep["file_path"]: ep for ep in episodes_info}

    # --------------------------------------------------------
    # 2. Phase 1: Build source cache (multiprocess)
    # --------------------------------------------------------
    print("=" * 70)
    print("Phase 1: Building source cache at waypoint positions...")
    print("=" * 70)

    splits = _generate_splits(NUM_WORKERS)
    t0 = time.time()

    phase1_args = [(i, splits[i], SRC_DATA_DIR, wp_by_filepath)
                   for i in range(NUM_WORKERS)]

    with mp.Pool(NUM_WORKERS) as pool:
        cache_results = pool.map(_build_cache_worker, phase1_args)

    # Merge caches from all workers
    src_cache = {}
    for local_cache in cache_results:
        src_cache.update(local_cache)
    del cache_results

    elapsed = time.time() - t0
    total_cached_steps = sum(len(v) for v in src_cache.values())
    print(f"\n  Phase 1 done: cached {total_cached_steps} waypoint steps "
          f"from {len(src_cache)} episodes in {elapsed:.1f}s\n")

    # --------------------------------------------------------
    # 3. Phase 2: Verify destination dataset (multiprocess)
    #    Set globals BEFORE creating Pool so forked workers
    #    inherit src_cache via COW (no pickle, no extra memory).
    # --------------------------------------------------------
    print("=" * 70)
    print("Phase 2: Verifying destination dataset against source cache...")
    print("=" * 70)

    _g_src_cache = src_cache
    _g_wp_by_filepath = wp_by_filepath

    t0 = time.time()

    phase2_args = [(i, splits[i], DST_RLDS_DIR)
                   for i in range(NUM_WORKERS)]

    with mp.Pool(NUM_WORKERS) as pool:
        verify_results = pool.map(_verify_worker, phase2_args)

    elapsed = time.time() - t0

    # --------------------------------------------------------
    # 4. Aggregate results and print summary
    # --------------------------------------------------------
    n_episodes_checked = sum(r["episodes_checked"] for r in verify_results)
    n_steps_checked = sum(r["steps_checked"] for r in verify_results)
    all_errors = []
    for r in verify_results:
        all_errors.extend(r["errors"])
    n_errors = len(all_errors)

    print(f"\n{'='*70}")
    print(f"Verification Complete ({elapsed:.1f}s for Phase 2)")
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
        print(f"\n  First {min(len(all_errors), 30)} error details:")
        for i, detail in enumerate(all_errors[:30]):
            print(f"    {i+1}. {detail}")
        if len(all_errors) > 30:
            print(f"    ... and {len(all_errors) - 30} more errors")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()