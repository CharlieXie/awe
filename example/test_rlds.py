# import tensorflow_datasets as tfds
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt

# data_dir = r"C:\Users\chuanlia\Documents\learning_space\ntu\projects\awe\example\rlds_data"
# builder = tfds.builder_from_directory(data_dir)
# dataset = builder.as_dataset(split="train")

# for episode in dataset.take(1):
#     steps = list(episode["steps"])
#     print(f"Episode 共 {len(steps)} 步\n")
    
#     # ========== 打印前3步的具体数值 ==========
#     for i, step in enumerate(steps[:3]):
#         print(f"--- Step {i} ---")
#         print(f"  is_first: {step['is_first'].numpy()}")
#         print(f"  is_last:  {step['is_last'].numpy()}")
#         print(f"  language_instruction: {step['language_instruction'].numpy().decode('utf-8')}")
#         print(f"  segment_idx: {step['segment_idx'].numpy()}")
#         print(f"  variant_idx: {step['variant_idx'].numpy()}")
        
#         obs = step["observation"]
#         print(f"  observation/joint_position_arm_left:  {obs['joint_position_arm_left'].numpy()}")
#         print(f"  observation/joint_position_arm_right: {obs['joint_position_arm_right'].numpy()}")
#         print(f"  observation/joint_position_torso:     {obs['joint_position_torso'].numpy()}")
#         print(f"  observation/gripper_state_left:       {obs['gripper_state_left'].numpy()}")
#         print(f"  observation/gripper_state_right:      {obs['gripper_state_right'].numpy()}")
#         print(f"  observation/base_velocity:            {obs['base_velocity'].numpy()}")
#         print(f"  action: {step['action'].numpy()}")
#         print()
    
#     # ========== 可视化第一步的 6 张图像 ==========
#     first_step = steps[0]
#     obs = first_step["observation"]
    
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     fig.suptitle("Step 0 - All Camera Views", fontsize=16)
    
#     # RGB 图像 (第一行)
#     rgb_cameras = [
#         ("image_camera_head", "Head RGB"),
#         ("image_camera_wrist_left", "Left Wrist RGB"),
#         ("image_camera_wrist_right", "Right Wrist RGB"),
#     ]
#     for j, (cam_key, cam_name) in enumerate(rgb_cameras):
#         img = obs[cam_key].numpy()
#         axes[0, j].imshow(img)
#         axes[0, j].set_title(cam_name)
#         axes[0, j].axis("off")
    
#     # 深度图 (第二行)
#     depth_cameras = [
#         ("depth_camera_head", "Head Depth"),
#         ("depth_camera_wrist_left", "Left Wrist Depth"),
#         ("depth_camera_wrist_right", "Right Wrist Depth"),
#     ]
#     for j, (cam_key, cam_name) in enumerate(depth_cameras):
#         depth = obs[cam_key].numpy().squeeze()  # 去掉最后一维
#         axes[1, j].imshow(depth, cmap="viridis")
#         axes[1, j].set_title(cam_name)
#         axes[1, j].axis("off")
    
#     plt.tight_layout()
#     plt.savefig("episode_step0_images.png", dpi=150)
#     plt.show()
#     print("图像已保存为 episode_step0_images.png")
    
#     # ========== 绘制整个 episode 的关节轨迹 ==========
#     joint_left_all = []
#     joint_right_all = []
#     gripper_left_all = []
#     gripper_right_all = []
#     actions_all = []
    
#     for step in steps:
#         obs = step["observation"]
#         joint_left_all.append(obs["joint_position_arm_left"].numpy())
#         joint_right_all.append(obs["joint_position_arm_right"].numpy())
#         gripper_left_all.append(obs["gripper_state_left"].numpy())
#         gripper_right_all.append(obs["gripper_state_right"].numpy())
#         actions_all.append(step["action"].numpy())
    
#     joint_left_all = np.array(joint_left_all)    # (T, 6)
#     joint_right_all = np.array(joint_right_all)   # (T, 6)
#     gripper_left_all = np.array(gripper_left_all)  # (T, 1)
#     gripper_right_all = np.array(gripper_right_all) # (T, 1)
#     actions_all = np.array(actions_all)            # (T, 26)
    
#     T = len(steps)
#     time_steps = np.arange(T)
    
#     fig, axes = plt.subplots(3, 1, figsize=(14, 10))
#     fig.suptitle("Episode Trajectory", fontsize=16)
    
#     # 左臂关节轨迹
#     for dim in range(6):
#         axes[0].plot(time_steps, joint_left_all[:, dim], label=f"joint_{dim}")
#     axes[0].set_title("Left Arm Joint Positions")
#     axes[0].set_xlabel("Step")
#     axes[0].legend(loc="upper right", fontsize=8)
    
#     # 右臂关节轨迹
#     for dim in range(6):
#         axes[1].plot(time_steps, joint_right_all[:, dim], label=f"joint_{dim}")
#     axes[1].set_title("Right Arm Joint Positions")
#     axes[1].set_xlabel("Step")
#     axes[1].legend(loc="upper right", fontsize=8)
    
#     # 夹爪状态
#     axes[2].plot(time_steps, gripper_left_all[:, 0], label="Left Gripper", linewidth=2)
#     axes[2].plot(time_steps, gripper_right_all[:, 0], label="Right Gripper", linewidth=2)
#     axes[2].set_title("Gripper States")
#     axes[2].set_xlabel("Step")
#     axes[2].legend()
    
#     plt.tight_layout()
#     plt.savefig("episode_trajectory.png", dpi=150)
#     plt.show()
#     print("轨迹图已保存为 episode_trajectory.png")


# # print("\n===== 验证 2: last_action[t] == action[t-1] =====\n")
# # for ep_idx, episode in enumerate(dataset.take(50)):
# #     steps = list(episode["steps"])
# #     T = len(steps)
    
# #     mismatches = 0
# #     max_diff = 0.0
# #     for t in range(1, T):
# #         prev_action = steps[t - 1]["action"].numpy()
# #         curr_last_action = steps[t]["observation"]["last_action"].numpy()
        
# #         diff = np.abs(prev_action - curr_last_action).max()
# #         max_diff = max(max_diff, diff)
# #         if diff > 1e-6:
# #             mismatches += 1
# #             if mismatches <= 2:  # 只打印前几个不匹配的
# #                 print(f"  Episode {ep_idx}, Step {t}: max_diff={diff:.8f}")
    
# #     status = "PASS" if mismatches == 0 else f"FAIL ({mismatches} mismatches)"
# #     print(f"Episode {ep_idx} ({T} steps): [{status}]  max_diff={max_diff:.2e}")




# # episode images to video


import tensorflow_datasets as tfds
import numpy as np
import cv2
import os

# 定义多组数据源和对应的输出目录
datasets_config = [
    {
        "data_dir": r"C:\Users\chuanlia\Documents\learning_space\ntu\projects\awe\example\rlds_data",
        "output_dir": r"C:\Users\chuanlia\Documents\learning_space\ntu\projects\awe\example\videos",
    },
    {
        "data_dir": r"C:\Users\chuanlia\Documents\learning_space\ntu\projects\awe\example\rlds_data_waypoint\waypoint_filtered_rlds\1.0.0",
        "output_dir": r"C:\Users\chuanlia\Documents\learning_space\ntu\projects\awe\example\videos_wp",
    },
]

rgb_cameras = [
    "image_camera_head",
    "image_camera_wrist_left",
    "image_camera_wrist_right",
]

fps = 15
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

for config in datasets_config:
    data_dir = config["data_dir"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"处理数据集: {data_dir}")
    print(f"输出目录:   {output_dir}")
    print(f"{'='*60}")

    # 加载数据集
    builder = tfds.builder_from_directory(data_dir)
    dataset = builder.as_dataset(split="train")

    # 取第一个 episode
    for episode in dataset.skip(111).take(1):
    # for episode in dataset.take(1):
        steps = list(episode["steps"])
        print(f"Episode 共 {len(steps)} 步")

        # ========== 保存 RGB 视频 ==========
        for cam_name in rgb_cameras:
            video_path = os.path.join(output_dir, f"{cam_name}.mp4")
            writer = cv2.VideoWriter(video_path, fourcc, fps, (224, 224))

            for step in steps:
                img = step["observation"][cam_name].numpy()  # (224, 224, 3) uint8, RGB
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                writer.write(img_bgr)

            writer.release()
            print(f"已保存: {video_path}")

        print(f"所有视频已保存到: {output_dir}")



# import tensorflow_datasets as tfds
# import numpy as np
# import matplotlib.pyplot as plt

# data_dir = r"C:\Users\chuanlia\Documents\learning_space\ntu\projects\awe\example\rlds_data"
# builder = tfds.builder_from_directory(data_dir)
# dataset = builder.as_dataset(split="train")

# for episode in dataset.take(1):
#     steps = list(episode["steps"])
#     T = len(steps)
#     print(f"Episode 共 {T} 步")

#     # 收集所有 left arm 相关数据
#     joint_pos = []      # 关节位置 (观测)
#     joint_vel = []      # 关节速度 (观测)
#     gripper = []        # 夹爪状态 (观测)
#     action_pos = []     # 关节位置 (动作指令)
#     action_gripper = [] # 夹爪指令

#     for step in steps:
#         obs = step["observation"]
#         act = step["action"].numpy()

#         joint_pos.append(obs["joint_position_arm_left"].numpy())
#         joint_vel.append(obs["joint_velocity_arm_left"].numpy())
#         gripper.append(obs["gripper_state_left"].numpy())
#         action_pos.append(act[0:6])       # action 前6维 = 左臂关节目标
#         action_gripper.append(act[6])     # action 第7维 = 左夹爪目标

#     joint_pos = np.array(joint_pos)           # (T, 6)
#     joint_vel = np.array(joint_vel)           # (T, 6)
#     gripper = np.array(gripper)               # (T, 1)
#     action_pos = np.array(action_pos)         # (T, 6)
#     action_gripper = np.array(action_gripper) # (T,)
#     time_steps = np.arange(T)

#     joint_names = [f"Joint {i}" for i in range(6)]

#     # ========== 图1: 关节位置 (观测 vs 动作指令) ==========
#     fig, axes = plt.subplots(3, 2, figsize=(16, 12))
#     fig.suptitle("Left Arm - Joint Position: Observation vs Action Command", fontsize=14)

#     for i in range(6):
#         ax = axes[i // 2, i % 2]
#         ax.plot(time_steps, joint_pos[:, i], label="obs (当前位置)", linewidth=1.5)
#         ax.plot(time_steps, action_pos[:, i], label="action (目标位置)", linewidth=1.5, linestyle="--")
#         ax.set_title(joint_names[i])
#         ax.set_xlabel("Step")
#         ax.set_ylabel("Position (rad)")
#         ax.legend(fontsize=8)
#         ax.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig("left_arm_pos_obs_vs_action.png", dpi=150)
#     plt.show()

#     # ========== 图2: 关节速度 ==========
#     fig, axes = plt.subplots(3, 2, figsize=(16, 12))
#     fig.suptitle("Left Arm - Joint Velocity", fontsize=14)

#     for i in range(6):
#         ax = axes[i // 2, i % 2]
#         ax.plot(time_steps, joint_vel[:, i], linewidth=1.5, color="tab:green")
#         ax.set_title(joint_names[i])
#         ax.set_xlabel("Step")
#         ax.set_ylabel("Velocity (rad/s)")
#         ax.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig("left_arm_velocity.png", dpi=150)
#     plt.show()

#     # ========== 图3: 夹爪 (观测 vs 动作指令) ==========
#     fig, ax = plt.subplots(figsize=(12, 4))
#     ax.plot(time_steps, gripper[:, 0], label="obs (当前状态)", linewidth=2)
#     ax.plot(time_steps, action_gripper, label="action (目标值)", linewidth=2, linestyle="--")
#     ax.set_title("Left Gripper: Observation vs Action Command")
#     ax.set_xlabel("Step")
#     ax.set_ylabel("Gripper State")
#     ax.legend()
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig("left_arm_gripper.png", dpi=150)
#     plt.show()

#     # ========== 导出为 CSV 方便后续分析 ==========
#     header = ",".join(
#         [f"pos_joint{i}" for i in range(6)] +
#         [f"vel_joint{i}" for i in range(6)] +
#         ["gripper_state"] +
#         [f"action_joint{i}" for i in range(6)] +
#         ["action_gripper"]
#     )
#     data = np.hstack([joint_pos, joint_vel, gripper, action_pos, action_gripper[:, None]])
#     np.savetxt("left_arm_trajectory.csv", data, delimiter=",", header=header, comments="")
#     print("轨迹已导出为 left_arm_trajectory.csv")