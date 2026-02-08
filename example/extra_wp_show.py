"""..."""

import sys
import os
import types
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 1. 先设置 PROJECT_ROOT
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, PROJECT_ROOT)

# 2. 然后 mock mujoco_py（必须在 import waypoint_extraction 之前！）
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

# 3. 现在可以安全导入官方 API 了
from waypoint_extraction import dp_waypoint_selection, greedy_waypoint_selection
from utils.utils import plot_3d_trajectory

# 4. 后面的代码保持不变...
# ============================================================
# 导入 AWE 官方 API
# ============================================================

# # 官方 waypoint 提取 API
# from waypoint_extraction import dp_waypoint_selection, greedy_waypoint_selection

# # 官方 3D 轨迹可视化 API
# from utils.utils import plot_3d_trajectory


# ============================================================
# Part 1: 模拟 RLDS 数据生成
# ============================================================

def generate_arc(start, end, height, n_points, noise_std=0.002):
    """生成带弧度的 3D 路径（抛物线形状）"""
    t = np.linspace(0, 1, n_points)
    path = np.outer(1 - t, start) + np.outer(t, end)
    arc_height = height * 4 * t * (1 - t)
    path[:, 2] += arc_height
    path += np.random.randn(*path.shape) * noise_std
    return path


def generate_line(start, end, n_points, noise_std=0.001):
    """生成直线路径"""
    t = np.linspace(0, 1, n_points).reshape(-1, 1)
    path = (1 - t) * start + t * end
    path += np.random.randn(*path.shape) * noise_std
    return path


def generate_rlds_pick_and_place_episode():
    """
    生成一条模拟 RLDS 格式的 pick-and-place 轨迹。
    
    返回 dict，结构模仿 RLDS 标准格式：
    {
        'steps': {
            'observation': {
                'state': np.ndarray (T, 7),          # [x, y, z, rx, ry, rz, gripper]
                'cartesian_position': np.ndarray (T, 3),  # [x, y, z]
            },
            'action': np.ndarray (T, 7),
            'is_terminal': np.ndarray (T,),
        },
        'metadata': { ... }
    }
    """
    np.random.seed(42)
    
    home_pos      = np.array([0.3, 0.0, 0.4])
    obj_above     = np.array([0.5, -0.1, 0.3])
    obj_pos       = np.array([0.5, -0.1, 0.08])
    lift_pos      = np.array([0.5, -0.1, 0.35])
    target_above  = np.array([0.3, 0.25, 0.35])
    target_pos    = np.array([0.3, 0.25, 0.08])
    retreat_pos   = np.array([0.3, 0.25, 0.35])
    
    phase1 = generate_line(home_pos, obj_above, n_points=50, noise_std=0.001)
    phase2 = generate_line(obj_above, obj_pos, n_points=30, noise_std=0.0005)
    phase3 = generate_line(obj_pos, obj_pos + np.array([0, 0, 0.005]), n_points=10, noise_std=0.0002)
    phase4 = generate_line(obj_pos + np.array([0, 0, 0.005]), lift_pos, n_points=30, noise_std=0.001)
    phase5 = generate_arc(lift_pos, target_above, height=0.15, n_points=80, noise_std=0.002)
    phase6 = generate_line(target_above, target_pos, n_points=30, noise_std=0.0005)
    phase7 = generate_line(target_pos, target_pos + np.array([0, 0, 0.005]), n_points=10, noise_std=0.0002)
    phase8 = generate_line(target_pos + np.array([0, 0, 0.005]), retreat_pos, n_points=30, noise_std=0.001)
    
    xyz = np.vstack([phase1, phase2, phase3, phase4, phase5, phase6, phase7, phase8])
    T = len(xyz)
    
    boundaries = {
        'phase1_reach':     (0, 50),
        'phase2_descend':   (50, 80),
        'phase3_grasp':     (80, 90),
        'phase4_lift':      (90, 120),
        'phase5_transport': (120, 200),
        'phase6_descend':   (200, 230),
        'phase7_release':   (230, 240),
        'phase8_retreat':   (240, 270),
    }
    
    rz = np.linspace(0, 0.3, T) + np.random.randn(T) * 0.005
    rx = np.zeros(T) + np.random.randn(T) * 0.002
    ry = np.zeros(T) + np.random.randn(T) * 0.002
    
    gripper = np.ones(T)
    gripper[80:230] = 0.0
    
    state = np.column_stack([xyz, rx, ry, rz, gripper])
    action = np.diff(state, axis=0, prepend=state[:1])
    action[:, -1] = gripper
    
    episode = {
        'steps': {
            'observation': {
                'state': state.astype(np.float32),
                'cartesian_position': xyz.astype(np.float32),
            },
            'action': action.astype(np.float32),
            'is_terminal': np.zeros(T, dtype=bool),
        },
        'metadata': {
            'task': 'pick_and_place',
            'num_steps': T,
            'phase_boundaries': boundaries,
        }
    }
    episode['steps']['is_terminal'][-1] = True
    return episode


# ============================================================
# Part 2: 使用官方 API 的可视化
# ============================================================

def visualize_with_official_api(xyz, waypoints, gripper, title="", save_path=None):
    """
    使用 AWE 官方 plot_3d_trajectory 绘制 ground truth + waypoints。
    
    这就是官方 act_waypoint.py 里的可视化方式：
      - "ground truth" → 蓝色渐变点
      - "waypoints"    → 红色渐变点 + 红色箭头
      - 夹爪变化        → 特殊标记和颜色
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # --- 官方 API 调用 1: 画完整轨迹 (蓝色) ---
    plot_3d_trajectory(
        ax, 
        xyz,                     # (T, 3) 的 numpy 数组
        label="ground truth",    # 触发蓝色渐变
        gripper=gripper,         # 夹爪状态，变化时改颜色和标记
        legend=False,
    )
    
    # --- 官方 API 调用 2: 画 waypoint 子集 (红色 + 箭头) ---
    wp_with_start = [0] + waypoints
    plot_3d_trajectory(
        ax, 
        xyz[wp_with_start],      # 只取 waypoint 帧的 xyz
        label="waypoints",       # 触发红色渐变 + 箭头
        gripper=gripper[wp_with_start],
        legend=False,
    )
    
    # 合并图例（避免重复）
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10)
    
    T = len(xyz)
    ax.set_title(
        f'{title}\n{T} frames → {len(waypoints)} waypoints '
        f'(compression {T/len(waypoints):.1f}x)',
        fontsize=13
    )
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存到 {save_path}")
    plt.show()


def visualize_dual_arm_style(xyz, waypoints, gripper, title="", save_path=None):
    """
    模仿官方 act_waypoint.py 的双面板风格（左/右分割），
    这里我们用 "俯视 / 侧视" 两个角度展示同一条轨迹。
    """
    fig = plt.figure(figsize=(20, 8))
    
    # ---- 左面板：默认 3D 视角 ----
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title('Default View', fontsize=14)
    
    plot_3d_trajectory(ax1, xyz, label="ground truth", gripper=gripper, legend=False)
    wp_with_start = [0] + waypoints
    plot_3d_trajectory(ax1, xyz[wp_with_start], label="waypoints", 
                       gripper=gripper[wp_with_start], legend=False)
    
    # ---- 右面板：俯视角度 ----
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.set_title('Top-down View', fontsize=14)
    ax2.view_init(elev=80, azim=-90)  # 俯视
    
    plot_3d_trajectory(ax2, xyz, label="ground truth", gripper=gripper, legend=False)
    plot_3d_trajectory(ax2, xyz[wp_with_start], label="waypoints",
                       gripper=gripper[wp_with_start], legend=False)
    
    # 统一图例
    T = len(xyz)
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=2, fontsize=12)
    fig.suptitle(
        f'{title}  |  {T} frames → {len(waypoints)} waypoints '
        f'(compression {T/len(waypoints):.1f}x)',
        fontsize=15, fontweight='bold'
    )
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存到 {save_path}")
    plt.show()


def visualize_density_and_timeline(episode, waypoints, err_threshold, save_path=None):
    """补充可视化：waypoint 密度分析 + 时间线（官方未提供此功能）"""
    
    xyz = episode['steps']['observation']['cartesian_position']
    gripper = episode['steps']['observation']['state'][:, -1]
    boundaries = episode['metadata']['phase_boundaries']
    T = len(xyz)
    
    phase_colors = {
        'phase1_reach': '#2196F3',     'phase2_descend': '#4CAF50',
        'phase3_grasp': '#FF9800',     'phase4_lift': '#9C27B0',
        'phase5_transport': '#F44336', 'phase6_descend': '#00BCD4',
        'phase7_release': '#FF9800',   'phase8_retreat': '#607D8B',
    }
    phase_labels = {
        'phase1_reach': '1.Reach',        'phase2_descend': '2.Descend',
        'phase3_grasp': '3.Grasp',        'phase4_lift': '4.Lift',
        'phase5_transport': '5.Transport', 'phase6_descend': '6.Descend',
        'phase7_release': '7.Release',    'phase8_retreat': '8.Retreat',
    }
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(
        f'Waypoint Density & Timeline Analysis  (eta={err_threshold})',
        fontsize=14, fontweight='bold'
    )
    
    # ---- 上: 每阶段 waypoint 密度柱状图 ----
    phase_names_short = []
    wp_counts = []
    frame_counts = []
    
    for phase_name, (start, end) in boundaries.items():
        n_frames = end - start
        n_wps = sum(1 for w in waypoints if start <= w < end)
        phase_names_short.append(phase_labels[phase_name])
        wp_counts.append(n_wps)
        frame_counts.append(n_frames)
    
    x_pos = np.arange(len(phase_names_short))
    ax1.bar(x_pos - 0.2, frame_counts, 0.35, label='Total Frames', 
            color='#90CAF9', edgecolor='#1565C0')
    ax1.bar(x_pos + 0.2, wp_counts, 0.35, label='Waypoints',
            color='#EF9A9A', edgecolor='#C62828')
    
    for i, (fc, wc) in enumerate(zip(frame_counts, wp_counts)):
        if wc > 0:
            dens = wc / fc * 100
            ax1.text(i + 0.2, wc + 0.5, f'{dens:.0f}%', ha='center', fontsize=9, color='#C62828')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(phase_names_short, fontsize=10)
    ax1.set_ylabel('Count')
    ax1.set_title('Waypoint Density per Phase (% = waypoints/frames)')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ---- 下: XYZ + Gripper 时间线 + waypoint 位置 ----
    time = np.arange(T)
    ax2.plot(time, xyz[:, 0], label='X', color='#F44336', alpha=0.7, linewidth=1)
    ax2.plot(time, xyz[:, 1], label='Y', color='#4CAF50', alpha=0.7, linewidth=1)
    ax2.plot(time, xyz[:, 2], label='Z', color='#2196F3', alpha=0.7, linewidth=1)
    ax2.plot(time, gripper * 0.3 + 0.1, label='Gripper', color='#FF9800',
             linewidth=2, linestyle='--')
    
    for w in waypoints:
        ax2.axvline(x=w, color='red', alpha=0.15, linewidth=1)
    ax2.axvline(x=-10, color='red', alpha=0.4, linewidth=2, label='Waypoints')
    
    for phase_name, (start, end) in boundaries.items():
        ax2.axvspan(start, end, alpha=0.06, color=phase_colors[phase_name])
        mid = (start + end) / 2
        ax2.text(mid, ax2.get_ylim()[1] if hasattr(ax2, '_ylim') else 0.45, 
                 phase_labels[phase_name], ha='center', fontsize=7, alpha=0.6)
    
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Value')
    ax2.set_title('State Timeline + Waypoint Positions (red vertical lines)')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.set_xlim(0, T)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存到 {save_path}")
    plt.show()


def visualize_eta_comparison(xyz, save_path=None):
    """对比不同 eta 下的效果（使用官方 greedy API）"""
    
    etas = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('Effect of Error Threshold eta on Waypoint Selection\n'
                 '(using official greedy_waypoint_selection API)',
                 fontsize=14, fontweight='bold')
    
    for idx, eta in enumerate(etas):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        
        # --- 官方 API: greedy_waypoint_selection ---
        wps = greedy_waypoint_selection(
            actions=xyz,
            gt_states=xyz,
            err_threshold=eta,
            pos_only=True,
        )
        
        wp_indices = [0] + wps
        
        # --- 官方 API: plot_3d_trajectory ---
        plot_3d_trajectory(ax, xyz, label="ground truth", legend=False)
        plot_3d_trajectory(ax, xyz[wp_indices], label="waypoints", legend=False)
        
        ratio = len(xyz) / len(wps)
        ax.set_title(f'eta={eta}\n{len(wps)} wps (1:{ratio:.0f})', fontsize=11)
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  保存到 {save_path}")
    plt.show()


# ============================================================
# Part 3: 主程序
# ============================================================

def main():
    print("=" * 60)
    print("RLDS Waypoint Demo (using AWE Official API)")
    print("=" * 60)
    
    # 确保输出目录存在
    os.makedirs(os.path.join(PROJECT_ROOT, 'plot', 'rlds'), exist_ok=True)
    plot_dir = os.path.join(PROJECT_ROOT, 'plot', 'rlds')
    
    # ---- Step 1: 生成 RLDS 测试数据 ----
    print("\n[Step 1] 生成模拟 RLDS pick-and-place episode...")
    episode = generate_rlds_pick_and_place_episode()
    T = episode['metadata']['num_steps']
    print(f"  轨迹长度: {T} 帧")
    print(f"  状态维度: {episode['steps']['observation']['state'].shape}")
    print(f"  阶段划分:")
    for name, (s, e) in episode['metadata']['phase_boundaries'].items():
        print(f"    {name}: frame {s}-{e} ({e-s} frames)")
    
    # 提取 xyz 和 gripper
    xyz = episode['steps']['observation']['cartesian_position']  # (T, 3)
    gripper = episode['steps']['observation']['state'][:, -1]    # (T,)
    
    # ---- Step 2: 用官方 DP API 提取 waypoints ----
    err_threshold = 0.008
    print(f"\n[Step 2] 调用官方 dp_waypoint_selection API...")
    print(f"  输入 shape: {xyz.shape}, eta={err_threshold}")
    
    waypoints = dp_waypoint_selection(
        actions=xyz,
        gt_states=xyz,
        err_threshold=err_threshold,
        pos_only=True,
    )
    
    print(f"  结果: {len(waypoints)} waypoints, 索引={waypoints}")
    
    # 分析各阶段 waypoint 分布
    print(f"\n[Step 3] 各阶段 waypoint 分布:")
    for name, (s, e) in episode['metadata']['phase_boundaries'].items():
        n_wps = sum(1 for w in waypoints if s <= w < e)
        density = n_wps / (e - s) * 100
        bar = '#' * n_wps + '.' * max(0, 15 - n_wps)
        print(f"  {name:25s} [{s:3d}-{e:3d}] {n_wps:2d} wps ({density:5.1f}%) {bar[:15]}")
    
    # ---- Step 3: 可视化 (使用官方 plot_3d_trajectory) ----
    print(f"\n[Step 4] 生成可视化...")
    
    # 图 1: 官方风格 3D 轨迹（单面板）
    print("  图1: 官方风格 3D 轨迹...")
    visualize_with_official_api(
        xyz, waypoints, gripper,
        title="AWE Official API: Pick-and-Place",
        save_path=os.path.join(plot_dir, 'rlds_official_3d.png'),
    )
    
    # 图 2: 双面板（模仿官方 act_waypoint.py 的左右分割风格）
    print("  图2: 双视角面板...")
    visualize_dual_arm_style(
        xyz, waypoints, gripper,
        title="RLDS Pick-and-Place",
        save_path=os.path.join(plot_dir, 'rlds_official_dual_view.png'),
    )
    
    # 图 3: 密度分析 + 时间线（官方未提供，自定义补充）
    print("  图3: 密度分析 + 时间线...")
    visualize_density_and_timeline(
        episode, waypoints, err_threshold,
        save_path=os.path.join(plot_dir, 'rlds_density_timeline.png'),
    )
    
    # 图 4: 不同 eta 对比（使用官方 greedy API）
    print("  图4: eta 参数对比...")
    visualize_eta_comparison(
        xyz,
        save_path=os.path.join(plot_dir, 'rlds_eta_comparison.png'),
    )
    
    print("\n" + "=" * 60)
    print(f"完成! 图表保存在 {plot_dir}/")
    print("  1. rlds_official_3d.png        - 官方风格 3D 轨迹")
    print("  2. rlds_official_dual_view.png  - 双视角面板")
    print("  3. rlds_density_timeline.png    - 密度分析+时间线")
    print("  4. rlds_eta_comparison.png      - eta 参数对比")
    print("=" * 60)


if __name__ == "__main__":
    main()