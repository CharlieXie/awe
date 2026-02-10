"""
Multi-process RLDS dataset converter:
  - Remove depth images
  - Convert RGB images PNG -> JPEG (quality=95)
  - Record episode metadata to JSON
  - Update features.json & dataset_info.json

Usage:
    python convert_rlds_multiprocess.py
"""

import json
import os
import glob
import io
import time
import sys
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed


# ==================== Configuration ====================
INPUT_DIR  = r"/workspace/galaxea_data/rlds/part1_r1_lite/1.0.0"
OUTPUT_DIR = r"/workspace/awe/example/part1_r1_lite_compressed"
JPEG_QUALITY = 95
# NUM_WORKERS  = min(8, os.cpu_count() or 4)   # auto-detect, cap at 8
NUM_WORKERS  = 20

# RGB image keys (PNG -> JPEG)
RGB_IMAGE_KEYS = [
    "steps/observation/image_camera_head",
    "steps/observation/image_camera_wrist_left",
    "steps/observation/image_camera_wrist_right",
]

# Depth image keys (to remove)
DEPTH_IMAGE_KEYS = [
    "steps/observation/depth_camera_head",
    "steps/observation/depth_camera_wrist_left",
    "steps/observation/depth_camera_wrist_right",
]


# ==================== Worker Functions ====================
# These run in separate processes (spawned on Windows).
# TensorFlow is imported lazily inside the worker to avoid
# pickling issues and to limit TF init to worker processes only.

def png_to_jpeg(png_bytes: bytes, quality: int = 95) -> bytes:
    """Convert PNG/raw image bytes to JPEG bytes."""
    img = Image.open(io.BytesIO(png_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def extract_metadata(features) -> dict:
    """Extract important metadata from a tf.train.Example."""
    metadata = {}

    key = "episode_metadata/file_path"
    if key in features:
        vals = features[key].bytes_list.value
        if vals:
            metadata["file_path"] = vals[0].decode("utf-8")

    key = "steps/language_instruction"
    if key in features:
        vals = features[key].bytes_list.value
        if vals:
            unique = list(set(v.decode("utf-8") for v in vals))
            metadata["language_instruction"] = unique[0] if len(unique) == 1 else unique

    key = "steps/variant_idx"
    if key in features:
        vals = features[key].int64_list.value
        if vals:
            metadata["variant_idx"] = sorted(set(int(v) for v in vals))

    key = "steps/segment_idx"
    if key in features:
        vals = features[key].int64_list.value
        if vals:
            metadata["segment_idx"] = sorted(set(int(v) for v in vals))

    for key in RGB_IMAGE_KEYS:
        if key in features:
            metadata["num_steps"] = len(features[key].bytes_list.value)
            break

    return metadata


def process_shard(input_path: str, output_path: str, shard_idx: int) -> dict:
    """
    Process one TFRecord shard:
      1. Read all examples
      2. Extract metadata
      3. Convert RGB -> JPEG
      4. Remove depth features
      5. Write new TFRecord
    Returns dict with results for this shard.
    """
    # Lazy import TF in worker process (avoids pickle issues on Windows)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    filename = os.path.basename(input_path)
    orig_size = os.path.getsize(input_path)
    episodes_metadata = []
    episode_count = 0

    writer = tf.io.TFRecordWriter(output_path)
    dataset = tf.data.TFRecordDataset(input_path)

    for raw_record in dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature

        # --- metadata ---
        meta = extract_metadata(features)
        meta["source_file"] = filename
        meta["shard_idx"] = shard_idx
        meta["local_episode_idx"] = episode_count

        # --- RGB: PNG -> JPEG ---
        for key in RGB_IMAGE_KEYS:
            if key in features:
                png_list = features[key].bytes_list.value
                jpeg_list = [png_to_jpeg(b, JPEG_QUALITY) for b in png_list]
                del features[key].bytes_list.value[:]
                features[key].bytes_list.value.extend(jpeg_list)

        # --- Remove depth ---
        for key in DEPTH_IMAGE_KEYS:
            if key in features:
                del features[key]

        writer.write(example.SerializeToString())
        episodes_metadata.append(meta)
        episode_count += 1

    writer.close()
    new_size = os.path.getsize(output_path)

    return {
        "shard_idx": shard_idx,
        "filename": filename,
        "episodes": episodes_metadata,
        "episode_count": episode_count,
        "orig_bytes": orig_size,
        "new_bytes": new_size,
    }


# ==================== JSON helpers ====================

def update_features_json(input_dir: str, output_dir: str):
    """Copy features.json, remove depth entries, mark JPEG encoding."""
    src = os.path.join(input_dir, "features.json")
    if not os.path.exists(src):
        print("  [WARN] features.json not found, skipping.")
        return

    with open(src, "r", encoding="utf-8") as f:
        feat = json.load(f)

    obs = (
        feat["featuresDict"]["features"]["steps"]["sequence"]["feature"]
        ["featuresDict"]["features"]["observation"]["featuresDict"]["features"]
    )

    # Remove depth
    for dk in ["depth_camera_head", "depth_camera_wrist_left", "depth_camera_wrist_right"]:
        if obs.pop(dk, None):
            print(f"    Removed: observation/{dk}")

    # Mark JPEG
    for ik in ["image_camera_head", "image_camera_wrist_left", "image_camera_wrist_right"]:
        if ik in obs:
            obs[ik]["image"]["encodingFormat"] = "jpeg"
            print(f"    Updated: observation/{ik} -> JPEG")

    dst = os.path.join(output_dir, "features.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(feat, f, indent=4, ensure_ascii=False)
    print(f"    Saved: {dst}")


def update_dataset_info(input_dir: str, output_dir: str, total_new_bytes: int):
    """Copy dataset_info.json, update numBytes."""
    src = os.path.join(input_dir, "dataset_info.json")
    if not os.path.exists(src):
        print("  [WARN] dataset_info.json not found, skipping.")
        return

    with open(src, "r", encoding="utf-8") as f:
        info = json.load(f)

    for split in info.get("splits", []):
        split["numBytes"] = str(total_new_bytes)

    dst = os.path.join(output_dir, "dataset_info.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"    Saved: {dst}")


# ==================== Main ====================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tfrecord_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tfrecord*")))
    # tfrecord_files = tfrecord_files[:800]  # 只处理前 800 个
    if not tfrecord_files:
        print(f"No TFRecord files found in {INPUT_DIR}")
        sys.exit(1)

    print(f"Input:   {INPUT_DIR}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Shards:  {len(tfrecord_files)}")
    print(f"JPEG Q:  {JPEG_QUALITY}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"{'=' * 60}")

    # ---- Build task list ----
    tasks = []
    for idx, tfpath in enumerate(tfrecord_files):
        fname = os.path.basename(tfpath)
        out_path = os.path.join(OUTPUT_DIR, fname)
        tasks.append((tfpath, out_path, idx))

    # ---- Process shards in parallel ----
    all_results = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_shard, inp, out, idx): idx
            for inp, out, idx in tasks
        }

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            try:
                result = future.result()
            except Exception as exc:
                shard_idx = future_to_idx[future]
                print(f"  [ERROR] shard {shard_idx} raised: {exc}")
                continue

            all_results.append(result)
            done = len(all_results)
            elapsed = time.time() - t0
            ratio = result["orig_bytes"] / result["new_bytes"] if result["new_bytes"] > 0 else 0
            print(
                f"  [{done:>3}/{len(tasks)}] shard {result['shard_idx']:03d}  "
                f"{result['episode_count']} ep  "
                f"{result['orig_bytes']/1e6:>7.1f}MB -> {result['new_bytes']/1e6:>7.1f}MB  "
                f"({ratio:.1f}x)  [{elapsed:.0f}s]"
            )

    # ---- Sort by shard index for consistent ordering ----
    all_results.sort(key=lambda r: r["shard_idx"])

    # ---- Aggregate ----
    total_episodes = sum(r["episode_count"] for r in all_results)
    total_orig = sum(r["orig_bytes"] for r in all_results)
    total_new  = sum(r["new_bytes"]  for r in all_results)

    # Assign global episode indices (ordered by shard)
    all_episodes = []
    global_idx = 0
    for result in all_results:
        for ep in result["episodes"]:
            ep["episode_global_idx"] = global_idx
            all_episodes.append(ep)
            global_idx += 1

    # ---- Update JSON config files ----
    print(f"\n{'=' * 60}")
    print("Updating config files...")
    update_features_json(INPUT_DIR, OUTPUT_DIR)
    update_dataset_info(INPUT_DIR, OUTPUT_DIR, total_new)

    # ---- Save episode metadata ----
    metadata_path = os.path.join(OUTPUT_DIR, "episode_metadata.json")
    output_info = {
        "config": {
            "input_dir": INPUT_DIR,
            "output_dir": OUTPUT_DIR,
            "jpeg_quality": JPEG_QUALITY,
            "num_workers": NUM_WORKERS,
            "removed_features": DEPTH_IMAGE_KEYS,
            "converted_features": RGB_IMAGE_KEYS,
        },
        "summary": {
            "total_episodes": total_episodes,
            "total_shards": len(tfrecord_files),
            "original_size_MB": round(total_orig / 1e6, 2),
            "new_size_MB": round(total_new / 1e6, 2),
            "compression_ratio": round(total_orig / total_new, 2) if total_new > 0 else None,
            "space_saved_MB": round((total_orig - total_new) / 1e6, 2),
            "processing_time_s": round(time.time() - t0, 2),
        },
        "episodes": all_episodes,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(output_info, f, indent=2, ensure_ascii=False)

    # ---- Summary ----
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"{'=' * 60}")
    print(f"  Episodes:       {total_episodes}")
    print(f"  Shards:         {len(tfrecord_files)}")
    print(f"  Original size:  {total_orig / 1e9:.2f} GB")
    print(f"  New size:       {total_new  / 1e9:.2f} GB")
    if total_new > 0:
        print(f"  Compression:    {total_orig / total_new:.2f}x")
        print(f"  Space saved:    {(total_orig - total_new) / 1e9:.2f} GB")
    print(f"  Metadata JSON:  {metadata_path}")
    print(f"  Features JSON:  {os.path.join(OUTPUT_DIR, 'features.json')}")
    print(f"  Dataset info:   {os.path.join(OUTPUT_DIR, 'dataset_info.json')}")


if __name__ == "__main__":
    main()