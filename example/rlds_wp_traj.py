
import sys
import os
import types
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pinocchio as pin
import tensorflow_datasets as tfds


# ============================================================
# 0. Mock（必须在 import AWE 之前）
# ============================================================

# 第一步：清除已经加载的 mujoco_py（你环境里装了 mujoco_py 但没装 MuJoCo）
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

# 第二步：只 mock mujoco_py 和 glfw，不要 mock robosuite！
for _mod_name in [
    'mujoco_py', 'mujoco_py.builder', 'mujoco_py.utils',
    'mujoco_py.generated', 'mujoco_py.generated.const', 'mujoco_py.cymj',
]:
    sys.modules[_mod_name] = _MockModule(_mod_name)

try:
    import glfw
except (ImportError, ModuleNotFoundError):
    sys.modules['glfw'] = _MockModule('glfw')

# 添加 AWE 项目路径
awe_root = r"C:\Users\chuanlia\Documents\learning_space\ntu\projects\awe"
sys.path.insert(0, awe_root)

from waypoint_extraction import dp_waypoint_selection, greedy_waypoint_selection
from utils.utils import plot_3d_trajectory

# ============================================================
# 1. 加载 URDF 建立运动学模型
# ============================================================
urdf_path = r"C:\Users\chuanlia\Documents\learning_space\ntu\projects\URDF\R1\urdf\r1_v2_1_0.urdf"

model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

def get_joint_idx(model, name):
    joint_id = model.getJointId(name)
    return model.joints[joint_id].idx_q

torso_indices = [get_joint_idx(model, f"torso_joint{i}") for i in range(1, 5)]
left_arm_indices = [get_joint_idx(model, f"left_arm_joint{i}") for i in range(1, 7)]
left_ee_frame_id = model.getFrameId("left_gripper_link")

def compute_left_ee_pos(torso_q, left_arm_q):
    """正运动学: torso(4) + left_arm(6) -> EE xyz(3)"""
    q = pin.neutral(model)
    for i, idx in enumerate(torso_indices):
        q[idx] = torso_q[i]
    for i, idx in enumerate(left_arm_indices):
        q[idx] = left_arm_q[i]
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[left_ee_frame_id].translation.copy()

# ============================================================
# 2. 从 RLDS 读取数据，计算 EE 3D 轨迹
# ============================================================
data_dir = r"C:\Users\chuanlia\Documents\learning_space\ntu\projects\awe\example\rlds_data"
builder = tfds.builder_from_directory(data_dir)
dataset = builder.as_dataset(split="train")
output_dir = r"C:\Users\chuanlia\Documents\learning_space\ntu\projects\awe\example\videos"
os.makedirs(output_dir, exist_ok=True)
import cv2

for episode in dataset.skip(106).take(1):
    steps = list(episode["steps"])
    T = len(steps)
    print(f"Episode 共 {T} 步，开始计算 FK...")
    # 定义要保存的摄像头
    rgb_cameras = [
        "image_camera_head",
        "image_camera_wrist_left",
        "image_camera_wrist_right",
    ]
    depth_cameras = [
        "depth_camera_head",
        "depth_camera_wrist_left",
        "depth_camera_wrist_right",
    ]

    fps = 15  # 根据实际采集频率调整，常见 10~50 Hz
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # ========== 保存 RGB 视频 ==========
    for cam_name in rgb_cameras:
        video_path = os.path.join(output_dir, f"{cam_name}.mp4")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (224, 224))

        for step in steps:
            img = step["observation"][cam_name].numpy()  # (224, 224, 3) uint8, RGB
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # OpenCV 用 BGR
            writer.write(img_bgr)

        writer.release()
        print(f"已保存: {video_path}")


    ee_positions = []
    joint_positions_left = []   # <-- 新增：收集 joint_position_arm_left
    gripper_states = []

    for i, step in enumerate(steps):
        obs = step["observation"]
        torso_q = obs["joint_position_torso"].numpy()
        left_arm_q = obs["joint_position_arm_left"].numpy()
        gripper = obs["gripper_state_left"].numpy()[0]

        pos = compute_left_ee_pos(torso_q, left_arm_q)
        ee_positions.append(pos)
        joint_positions_left.append(left_arm_q)   # <-- 新增
        gripper_states.append(gripper)

        if i % 100 == 0:
            print(f"  Step {i}/{T}: EE=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    ee_positions = np.array(ee_positions)                  # (T, 3)
    joint_positions_left = np.array(joint_positions_left)  # (T, 6)  <-- 新增
    gripper_states = np.array(gripper_states)              # (T,)
    print(f"FK 计算完成！轨迹形状: {ee_positions.shape}")

    # ============================================================
    # 3. 用 AWE 提取 Waypoints（在 joint space 上提取）
    # ============================================================
    err_threshold = 0.0001  # 单位: rad (joint space)。越小 waypoints 越多越精确  # <-- 修改注释

    print(f"\n提取 Waypoints (joint space, err_threshold={err_threshold})...")  # <-- 修改
    waypoints = dp_waypoint_selection(
        env=None,
        actions=joint_positions_left,       # (T, 6) 的 joint position  # <-- 修改
        gt_states=joint_positions_left,     # (T, 6) 的 joint position  # <-- 修改
        err_threshold=err_threshold,
        pos_only=True,                      # 只用位置，不用姿态
    )
    print(f"结果: {T} 帧 -> {len(waypoints)} waypoints (压缩比 {T/len(waypoints):.1f}x)")

    # 确保包含起点
    wp_with_start = [0] + waypoints if 0 not in waypoints else waypoints

    # ============================================================
    # 4. 可视化（仍使用 EE 位置可视化，waypoint 索引来自 joint space）
    # ============================================================
    output_dir = r"C:\Users\chuanlia\Documents\learning_space\ntu\projects\awe\example"

    # # --- 图1: AWE 官方风格 3D 轨迹 (ground truth + waypoints) ---
    # fig = plt.figure(figsize=(14, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel('X (m)')
    # ax.set_ylabel('Y (m)')
    # ax.set_zlabel('Z (m)')

    # # 官方 API: 蓝色渐变画完整轨迹
    # plot_3d_trajectory(ax, ee_positions, label="ground truth",
    #                    gripper=gripper_states, legend=False)

    # # 官方 API: 红色 + 箭头画 waypoints
    # plot_3d_trajectory(ax, ee_positions[wp_with_start], label="waypoints",
    #                    gripper=gripper_states[wp_with_start], legend=False)

    # # 合并图例
    # handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys(), fontsize=11)

    # ax.set_title(
    #     f"Left Arm EE Trajectory (AWE Official Style)\n"
    #     f"{T} frames → {len(waypoints)} waypoints "
    #     f"(compression {T/len(waypoints):.1f}x, threshold={err_threshold}m)",
    #     fontsize=13
    # )

    # --- 图1: 精简版 3D 轨迹 ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 1) 原始轨迹：用一条线 + 颜色渐变表示时间
    #    用 LineCollection 太复杂，这里用分段画线实现渐变
    for i in range(T - 1):
        c = mpl.cm.Blues(0.3 + 0.6 * i / T)
        ax.plot(ee_positions[i:i+2, 0], ee_positions[i:i+2, 1], ee_positions[i:i+2, 2],
                color=c, linewidth=1.5, alpha=0.7)

    # 2) Waypoints：菱形标记 + 红色连线
    wp_pos = ee_positions[wp_with_start]
    ax.plot(wp_pos[:, 0], wp_pos[:, 1], wp_pos[:, 2],
            color='red', linewidth=1.5, linestyle='--', alpha=0.6)
    ax.scatter(wp_pos[:, 0], wp_pos[:, 1], wp_pos[:, 2],
               color='red', marker='D', s=50, zorder=5, label='Waypoints')

    # 3) 起点和终点
    ax.scatter(*ee_positions[0], color='lime', s=120, marker='o',
              edgecolors='black', linewidths=1, zorder=6, label='Start')
    ax.scatter(*ee_positions[-1], color='darkred', s=120, marker='X',
              edgecolors='black', linewidths=1, zorder=6, label='End')

    # 4) 夹爪变化：只在状态切换处标注（二值化后）
    gripper_binary = (gripper_states > 50).astype(float)
    for i in range(1, T):
        if gripper_binary[i] != gripper_binary[i - 1]:
            action = "Open" if gripper_binary[i] == 1 else "Close"
            ax.scatter(*ee_positions[i], color='orange', s=80, marker='s',
                       edgecolors='black', linewidths=0.8, zorder=5)
            ax.text(ee_positions[i, 0], ee_positions[i, 1], ee_positions[i, 2],
                    f' {action}', fontsize=8, color='darkorange')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(fontsize=10)
    ax.set_title(
        f"Left Arm EE Trajectory (Joint-space Waypoints)\n"          # <-- 修改标题
        f"{T} frames → {len(waypoints)} waypoints "
        f"(threshold={err_threshold} rad)",                           # <-- m -> rad
        fontsize=13
    )
    save_path = os.path.join(output_dir, "left_ee_awe_official.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"已保存: {save_path}")

    # --- 图2: 双视角 (默认3D + 俯视) ---
    fig = plt.figure(figsize=(22, 9))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title('3D View', fontsize=14)
    plot_3d_trajectory(ax1, ee_positions, label="ground truth",
                       gripper=gripper_states, legend=False)
    plot_3d_trajectory(ax1, ee_positions[wp_with_start], label="waypoints",
                       gripper=gripper_states[wp_with_start], legend=False)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.set_title('Top-down View', fontsize=14)
    ax2.view_init(elev=80, azim=-90)
    plot_3d_trajectory(ax2, ee_positions, label="ground truth",
                       gripper=gripper_states, legend=False)
    plot_3d_trajectory(ax2, ee_positions[wp_with_start], label="waypoints",
                       gripper=gripper_states[wp_with_start], legend=False)

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center',
               ncol=2, fontsize=12)
    fig.suptitle(
        f"Left Arm EE (Joint-space Waypoints): {T} frames → {len(waypoints)} waypoints "  # <-- 修改
        f"(threshold={err_threshold} rad)", fontsize=15, fontweight='bold')                 # <-- m -> rad
    plt.tight_layout()
    save_path = os.path.join(output_dir, "left_ee_dual_view.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"已保存: {save_path}")

    # --- 图3: XYZ 分量 + Waypoint 重建对比 ---
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle("Left Arm EE: Original vs Joint-space Waypoint Reconstruction", fontsize=15)  # <-- 修改

    t = np.arange(T)
    wp_t = np.array(wp_with_start)
    axis_labels = ["X (m)", "Y (m)", "Z (m)"]
    axis_colors = ["tab:red", "tab:green", "tab:blue"]

    for i in range(3):
        ax = axes[i]
        # 原始轨迹
        ax.plot(t, ee_positions[:, i], color=axis_colors[i], linewidth=1,
                alpha=0.7, label="Original")
        # Waypoints 标记
        ax.scatter(wp_t, ee_positions[wp_t, i], color="red", s=25,
                   zorder=5, label="Waypoints")
        # 线性插值重建
        reconstructed = np.interp(t, wp_t, ee_positions[wp_t, i])
        ax.plot(t, reconstructed, color="orange", linewidth=1.2,
                linestyle="--", alpha=0.8, label="Reconstructed")
        # 重建误差
        ax.fill_between(t, ee_positions[:, i], reconstructed,
                        alpha=0.1, color="red")
        ax.set_ylabel(axis_labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9, loc="upper right")
        # Waypoint 竖线
        for wi in wp_t:
            ax.axvline(x=wi, color="red", alpha=0.08, linewidth=0.8)

    # 夹爪状态
    axes[3].plot(t, gripper_states, color="purple", linewidth=2, label="Gripper State")
    for wi in wp_t:
        axes[3].axvline(x=wi, color="red", alpha=0.08, linewidth=0.8)
    axes[3].scatter(wp_t, gripper_states[wp_t], color="red", s=25, zorder=5)
    axes[3].set_ylabel("Gripper")
    axes[3].set_xlabel("Step")
    axes[3].legend(fontsize=9)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "left_ee_xyz_waypoints.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"已保存: {save_path}")

    # # --- 图4: 不同 threshold 对比 ---
    # thresholds = [0.005, 0.01, 0.02, 0.05, 0.1]
    # fig = plt.figure(figsize=(25, 5))
    # fig.suptitle("Effect of Error Threshold on Waypoint Selection", fontsize=14)

    # for idx, th in enumerate(thresholds):
    #     wps = dp_waypoint_selection(
    #         env=None, actions=ee_positions, gt_states=ee_positions,
    #         err_threshold=th, pos_only=True)
    #     wp_idx = [0] + wps if 0 not in wps else wps

    #     ax = fig.add_subplot(1, len(thresholds), idx + 1, projection='3d')
    #     plot_3d_trajectory(ax, ee_positions, label="ground truth", legend=False)
    #     plot_3d_trajectory(ax, ee_positions[wp_idx], label="waypoints", legend=False)
    #     ax.set_title(f"η={th}\n{len(wps)} wps ({T/len(wps):.0f}x)", fontsize=11)
    #     ax.set_xlabel('X', fontsize=8)
    #     ax.set_ylabel('Y', fontsize=8)
    #     ax.set_zlabel('Z', fontsize=8)

    # plt.tight_layout()
    # save_path = os.path.join(output_dir, "left_ee_threshold_compare.png")
    # fig.savefig(save_path, dpi=150, bbox_inches='tight')
    # plt.show()
    # print(f"已保存: {save_path}")

    # print(f"\n完成！所有图表保存在: {output_dir}")

    # --- 图5: 不同 threshold 下的 XYZ 分量重建对比 ---
    thresholds = [0.008, 0.007, 0.006, 0.005, 0.001]
    n_th = len(thresholds)
    t = np.arange(T)

    # 预先计算所有 threshold 的 waypoints（在 joint space 上提取）
    all_wp_results = {}
    for th in thresholds:
        wps = dp_waypoint_selection(
            env=None, actions=joint_positions_left, gt_states=joint_positions_left,  # <-- 修改
            err_threshold=th, pos_only=True)
        wp_idx = [0] + wps if 0 not in wps else wps
        all_wp_results[th] = (wps, np.array(wp_idx))

    # 布局: 3行(XYZ) x n_th列(每列一个threshold)
    fig, axes = plt.subplots(3, n_th, figsize=(5 * n_th, 10), sharex=True, sharey='row')
    fig.suptitle("EE XYZ Reconstruction (Joint-space Waypoints) across Thresholds",  # <-- 修改
                 fontsize=16, y=1.02)

    axis_labels = ["X (m)", "Y (m)", "Z (m)"]
    axis_colors = ["tab:red", "tab:green", "tab:blue"]

    for col, th in enumerate(thresholds):
        wps, wp_t = all_wp_results[th]

        for row in range(3):
            ax = axes[row, col]

            # 原始轨迹
            ax.plot(t, ee_positions[:, row], color=axis_colors[row],
                    linewidth=1, alpha=0.6, label="Original")

            # 线性插值重建
            reconstructed = np.interp(t, wp_t, ee_positions[wp_t, row])
            ax.plot(t, reconstructed, color="orange", linewidth=1.2,
                    linestyle="--", alpha=0.9, label="Reconstructed")

            # 重建误差阴影
            ax.fill_between(t, ee_positions[:, row], reconstructed,
                            alpha=0.12, color="red")

            # Waypoints 标记
            ax.scatter(wp_t, ee_positions[wp_t, row], color="red",
                       s=15, zorder=5, label="Waypoints")

            ax.grid(True, alpha=0.3)

            # 行标签 (最左列)
            if col == 0:
                ax.set_ylabel(axis_labels[row], fontsize=12)

            # 列标题 (第一行)
            if row == 0:
                ax.set_title(f"η = {th} rad\n{len(wps)} wps ({T/len(wps):.0f}x)",  # <-- 加 rad
                             fontsize=12, fontweight='bold')

            # 图例 (只在左上角显示一次)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper right")

            # X轴标签 (最后一行)
            if row == 2:
                ax.set_xlabel("Step", fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "left_ee_xyz_threshold_compare.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"已保存: {save_path}")




