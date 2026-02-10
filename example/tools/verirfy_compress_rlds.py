"""
Verify converted RLDS dataset:
  1. Shard file completeness
  2. Episode count per shard matches
  3. Feature key correctness (depth removed, others preserved)
  4. Non-image data bit-for-bit identical
  5. RGB images: decode and compare PSNR/SSIM (JPEG is lossy, can't be identical)
  6. New images are valid JPEG
"""

import tensorflow as tf
import numpy as np
import os
import glob
import io
import sys
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==================== Configuration ====================
ORIG_DIR = r"/workspace/galaxea_data/rlds/part1_r1_lite/1.0.0"
NEW_DIR  = r"/workspace/awe/example/part1_r1_lite_compressed"
NUM_WORKERS = 20

RGB_IMAGE_KEYS = [
    "steps/observation/image_camera_head",
    "steps/observation/image_camera_wrist_left",
    "steps/observation/image_camera_wrist_right",
]

DEPTH_IMAGE_KEYS = [
    "steps/observation/depth_camera_head",
    "steps/observation/depth_camera_wrist_left",
    "steps/observation/depth_camera_wrist_right",
]


# ==================== Helpers ====================

def decode_image_bytes(img_bytes):
    """Decode image bytes (PNG or JPEG) to numpy array."""
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img)


def compute_psnr(img1, img2):
    """Compute PSNR between two images (numpy arrays, uint8)."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def is_jpeg(data: bytes) -> bool:
    """Check if bytes start with JPEG magic number."""
    return data[:2] == b'\xff\xd8'


# ==================== Per-shard verification ====================

def verify_shard(orig_path: str, new_path: str, shard_idx: int) -> dict:
    """
    Compare one shard between original and new dataset.
    Returns a dict with verification results.
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    filename = os.path.basename(orig_path)
    errors = []
    warnings = []
    stats = {
        "shard_idx": shard_idx,
        "filename": filename,
        "episodes_orig": 0,
        "episodes_new": 0,
        "non_image_features_checked": 0,
        "non_image_mismatches": 0,
        "rgb_images_checked": 0,
        "rgb_all_jpeg": True,
        "rgb_psnr_min": float("inf"),
        "rgb_psnr_mean_sum": 0.0,
        "depth_correctly_removed": True,
    }

    # Read all records from both shards
    orig_records = list(tf.data.TFRecordDataset(orig_path))
    new_records  = list(tf.data.TFRecordDataset(new_path))

    stats["episodes_orig"] = len(orig_records)
    stats["episodes_new"]  = len(new_records)

    if len(orig_records) != len(new_records):
        errors.append(
            f"Episode count mismatch: orig={len(orig_records)}, new={len(new_records)}"
        )
        stats["errors"] = errors
        stats["warnings"] = warnings
        return stats

    for ep_idx, (orig_raw, new_raw) in enumerate(zip(orig_records, new_records)):
        orig_ex = tf.train.Example()
        orig_ex.ParseFromString(orig_raw.numpy())
        orig_feat = orig_ex.features.feature

        new_ex = tf.train.Example()
        new_ex.ParseFromString(new_raw.numpy())
        new_feat = new_ex.features.feature

        orig_keys = set(orig_feat.keys())
        new_keys  = set(new_feat.keys())

        # --- Check 1: depth features should be removed ---
        for dk in DEPTH_IMAGE_KEYS:
            if dk in new_keys:
                errors.append(f"ep{ep_idx}: depth key '{dk}' still present in new data")
                stats["depth_correctly_removed"] = False

        # --- Check 2: all non-depth keys should be present ---
        expected_keys = orig_keys - set(DEPTH_IMAGE_KEYS)
        missing_keys = expected_keys - new_keys
        extra_keys   = new_keys - expected_keys
        if missing_keys:
            errors.append(f"ep{ep_idx}: missing keys in new: {missing_keys}")
        if extra_keys:
            warnings.append(f"ep{ep_idx}: extra keys in new: {extra_keys}")

        # --- Check 3: non-image features should be bit-identical ---
        skip_keys = set(RGB_IMAGE_KEYS) | set(DEPTH_IMAGE_KEYS)
        for key in sorted(expected_keys - set(RGB_IMAGE_KEYS)):
            if key in skip_keys:
                continue
            if key not in new_feat:
                continue

            stats["non_image_features_checked"] += 1
            orig_f = orig_feat[key]
            new_f  = new_feat[key]

            # Compare serialized bytes for exact match
            if orig_f.SerializeToString() != new_f.SerializeToString():
                stats["non_image_mismatches"] += 1
                errors.append(f"ep{ep_idx}: feature '{key}' data mismatch!")

        # --- Check 4: RGB images - valid JPEG + PSNR ---
        for img_key in RGB_IMAGE_KEYS:
            if img_key not in orig_feat or img_key not in new_feat:
                continue

            orig_imgs = orig_feat[img_key].bytes_list.value
            new_imgs  = new_feat[img_key].bytes_list.value

            if len(orig_imgs) != len(new_imgs):
                errors.append(
                    f"ep{ep_idx}: '{img_key}' step count mismatch: "
                    f"orig={len(orig_imgs)}, new={len(new_imgs)}"
                )
                continue

            for step_idx, (orig_b, new_b) in enumerate(zip(orig_imgs, new_imgs)):
                stats["rgb_images_checked"] += 1

                # Check JPEG format
                if not is_jpeg(new_b):
                    stats["rgb_all_jpeg"] = False
                    errors.append(
                        f"ep{ep_idx}: '{img_key}' step{step_idx} is NOT JPEG"
                    )

                # Decode both and compare pixel values
                try:
                    orig_arr = decode_image_bytes(orig_b)
                    new_arr  = decode_image_bytes(new_b)

                    if orig_arr.shape != new_arr.shape:
                        errors.append(
                            f"ep{ep_idx}: '{img_key}' step{step_idx} shape mismatch: "
                            f"{orig_arr.shape} vs {new_arr.shape}"
                        )
                        continue

                    psnr = compute_psnr(orig_arr, new_arr)
                    stats["rgb_psnr_min"] = min(stats["rgb_psnr_min"], psnr)
                    stats["rgb_psnr_mean_sum"] += psnr

                except Exception as e:
                    errors.append(
                        f"ep{ep_idx}: '{img_key}' step{step_idx} decode error: {e}"
                    )

    stats["errors"]   = errors
    stats["warnings"] = warnings
    return stats


# ==================== Main ====================

def main():
    print(f"Original: {ORIG_DIR}")
    print(f"New:      {NEW_DIR}")
    print(f"Workers:  {NUM_WORKERS}")
    print()

    # --- Step 1: Check shard file completeness ---
    orig_shards = sorted(glob.glob(os.path.join(ORIG_DIR, "*.tfrecord*")))
    new_shards  = sorted(glob.glob(os.path.join(NEW_DIR,  "*.tfrecord*")))

    orig_names = {os.path.basename(f) for f in orig_shards}
    new_names  = {os.path.basename(f) for f in new_shards}

    missing_shards = orig_names - new_names
    extra_shards   = new_names - orig_names

    print(f"[Shard count] orig={len(orig_shards)}, new={len(new_shards)}")
    if missing_shards:
        print(f"  [FAIL] Missing shards in new: {sorted(missing_shards)}")
    if extra_shards:
        print(f"  [WARN] Extra shards in new: {sorted(extra_shards)}")
    if not missing_shards and not extra_shards:
        print(f"  [PASS] All shard files present")
    print()

    # --- Step 2: Build tasks for common shards ---
    common_names = sorted(orig_names & new_names)
    tasks = []
    for idx, name in enumerate(common_names):
        orig_path = os.path.join(ORIG_DIR, name)
        new_path  = os.path.join(NEW_DIR, name)
        tasks.append((orig_path, new_path, idx))

    print(f"Verifying {len(tasks)} common shards...")
    print("=" * 70)

    # --- Step 3: Parallel verification ---
    all_results = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(verify_shard, o, n, i): i
            for o, n, i in tasks
        }

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                idx = futures[future]
                print(f"  [ERROR] shard {idx} raised: {exc}")
                continue

            all_results.append(result)
            done = len(all_results)
            err_count = len(result["errors"])
            status = "PASS" if err_count == 0 else f"FAIL ({err_count} errors)"
            print(
                f"  [{done:>3}/{len(tasks)}] {result['filename']}  "
                f"ep={result['episodes_new']}  "
                f"imgs={result['rgb_images_checked']}  "
                f"psnr_min={result['rgb_psnr_min']:.1f}dB  "
                f"[{status}]"
            )

    # --- Step 4: Aggregate results ---
    all_results.sort(key=lambda r: r["shard_idx"])

    total_errors    = sum(len(r["errors"]) for r in all_results)
    total_warnings  = sum(len(r["warnings"]) for r in all_results)
    total_episodes  = sum(r["episodes_new"] for r in all_results)
    total_imgs      = sum(r["rgb_images_checked"] for r in all_results)
    total_non_img   = sum(r["non_image_features_checked"] for r in all_results)
    total_mismatch  = sum(r["non_image_mismatches"] for r in all_results)
    all_jpeg        = all(r["rgb_all_jpeg"] for r in all_results)
    all_depth_gone  = all(r["depth_correctly_removed"] for r in all_results)

    global_psnr_min = min(
        (r["rgb_psnr_min"] for r in all_results if r["rgb_images_checked"] > 0),
        default=float("inf"),
    )
    global_psnr_sum = sum(r["rgb_psnr_mean_sum"] for r in all_results)
    global_psnr_avg = global_psnr_sum / total_imgs if total_imgs > 0 else 0

    print()
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"  Shards verified:          {len(all_results)}/{len(orig_shards)}")
    print(f"  Missing shards:           {len(missing_shards)}")
    print(f"  Episodes verified:        {total_episodes}")
    print()
    print(f"  --- Non-image features ---")
    print(f"  Features compared:        {total_non_img}")
    print(f"  Mismatches:               {total_mismatch}")
    print(f"  Result:                   {'PASS' if total_mismatch == 0 else 'FAIL'}")
    print()
    print(f"  --- Depth removal ---")
    print(f"  All depth removed:        {'PASS' if all_depth_gone else 'FAIL'}")
    print()
    print(f"  --- RGB images ---")
    print(f"  Images compared:          {total_imgs}")
    print(f"  All valid JPEG:           {'PASS' if all_jpeg else 'FAIL'}")
    print(f"  PSNR (min):               {global_psnr_min:.2f} dB")
    print(f"  PSNR (avg):               {global_psnr_avg:.2f} dB")
    print(f"  Quality assessment:       ", end="")
    if global_psnr_min > 40:
        print("EXCELLENT (>40dB, visually lossless)")
    elif global_psnr_min > 35:
        print("GOOD (>35dB, minor artifacts)")
    elif global_psnr_min > 30:
        print("ACCEPTABLE (>30dB)")
    else:
        print(f"LOW (<30dB, check quality)")
    print()
    print(f"  --- Overall ---")
    print(f"  Errors:                   {total_errors}")
    print(f"  Warnings:                 {total_warnings}")
    overall = "PASS" if (total_errors == 0 and len(missing_shards) == 0) else "FAIL"
    print(f"  OVERALL:                  {overall}")
    print("=" * 70)

    # Print detailed errors if any
    if total_errors > 0:
        print("\nDETAILED ERRORS:")
        for r in all_results:
            if r["errors"]:
                print(f"\n  {r['filename']}:")
                for e in r["errors"]:
                    print(f"    - {e}")

    if total_warnings > 0:
        print("\nWARNINGS:")
        for r in all_results:
            if r["warnings"]:
                print(f"\n  {r['filename']}:")
                for w in r["warnings"]:
                    print(f"    - {w}")

    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()