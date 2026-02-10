import json
import os
import glob

OUTPUT_DIR = "/workspace/awe/example/part1_r1_lite_compressed"

# 1. 合并 episode_metadata
with open(os.path.join(OUTPUT_DIR, "episode_metadata_batch1.json")) as f:
    batch1 = json.load(f)
with open(os.path.join(OUTPUT_DIR, "episode_metadata_batch2.json")) as f:
    batch2 = json.load(f)

# 合并 episodes，修正 batch2 的 global_idx
all_episodes = batch1["episodes"]
offset = len(all_episodes)
for ep in batch2["episodes"]:
    ep["episode_global_idx"] += offset
    all_episodes.append(ep)

# 合并 summary
merged = batch1
merged["episodes"] = all_episodes
merged["summary"]["total_episodes"] = len(all_episodes)
merged["summary"]["total_shards"] = batch1["summary"]["total_shards"] + batch2["summary"]["total_shards"]
merged["summary"]["original_size_MB"] = round(
    batch1["summary"]["original_size_MB"] + batch2["summary"]["original_size_MB"], 2)
merged["summary"]["new_size_MB"] = round(
    batch1["summary"]["new_size_MB"] + batch2["summary"]["new_size_MB"], 2)

total_orig = merged["summary"]["original_size_MB"]
total_new = merged["summary"]["new_size_MB"]
merged["summary"]["compression_ratio"] = round(total_orig / total_new, 2) if total_new > 0 else None
merged["summary"]["space_saved_MB"] = round(total_orig - total_new, 2)

with open(os.path.join(OUTPUT_DIR, "episode_metadata.json"), "w") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

# 2. 修正 dataset_info.json 的 numBytes
# 用实际文件大小计算
tfrecords = glob.glob(os.path.join(OUTPUT_DIR, "*.tfrecord*"))
total_bytes = sum(os.path.getsize(f) for f in tfrecords)

info_path = os.path.join(OUTPUT_DIR, "dataset_info.json")
with open(info_path) as f:
    info = json.load(f)
for split in info.get("splits", []):
    split["numBytes"] = str(total_bytes)
with open(info_path, "w") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print(f"Merged episodes: {len(all_episodes)}")
print(f"Total compressed size: {total_bytes / 1e9:.2f} GB")
print("Done!")