import tensorflow_datasets as tfds
import cv2
import numpy as np
import os

# 配置
data_dir = r'test'  # 你的 RLDS 数据目录（包含 features.json 等文件的那个目录）
output_video = 'test/output.mp4'
IMAGE_KEY = 'image_camera_head'
fps = 15

def save_video():
    # 用 tfds 正确加载 RLDS 数据集
    builder = tfds.builder_from_directory(data_dir)
    dataset = builder.as_dataset(split='train')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    print("开始处理...")
    # 取第一个 episode
    for episode in dataset.take(1):
        steps = list(episode["steps"])
        print(f"Episode 共 {len(steps)} 步")

        for i, step in enumerate(steps):
            # 通过 TFDS 的嵌套结构访问图片
            frame = step["observation"][IMAGE_KEY].numpy()  # (224, 224, 3) uint8 RGB

            # RGB -> BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if out is None:
                h, w, _ = frame.shape
                out = cv2.VideoWriter(output_video, fourcc, float(fps), (w, h))

            out.write(frame)

            if i % 100 == 0:
                print(f"已处理 {i} 帧")

    if out:
        out.release()
    print(f"视频已保存为 {output_video}")

if __name__ == "__main__":
    save_video()