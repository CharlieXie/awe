import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

data_dir = r"C:\Users\chuanlia\Documents\learning_space\ntu\projects\awe\example\rlds_data"
builder = tfds.builder_from_directory(data_dir)
dataset = builder.as_dataset(split="train")

fig, axes = plt.subplots(5, 2, figsize=(18, 20))
fig.suptitle("Gripper State - First 10 Episodes", fontsize=18, fontweight='bold')

for ep_idx, episode in enumerate(dataset.take(10)):
    steps = list(episode["steps"])
    T = len(steps)

    gripper_left = np.array([s["observation"]["gripper_state_left"].numpy()[0] for s in steps])
    gripper_right = np.array([s["observation"]["gripper_state_right"].numpy()[0] for s in steps])

    # 同时提取 action 中的夹爪指令
    action_gripper_left = np.array([s["action"].numpy()[6] for s in steps])
    action_gripper_right = np.array([s["action"].numpy()[13] for s in steps])

    time_steps = np.arange(T)
    ax = axes[ep_idx // 2, ep_idx % 2]

    # 画观测值 (实线)
    ax.plot(time_steps, gripper_left, label="Left Gripper (obs)", linewidth=2, color='tab:blue')
    ax.plot(time_steps, gripper_right, label="Right Gripper (obs)", linewidth=2, color='tab:red')

    # 画 action 指令 (虚线)
    ax.plot(time_steps, action_gripper_left, label="Left Gripper (action)", 
            linewidth=1.5, linestyle='--', color='tab:blue', alpha=0.5)
    ax.plot(time_steps, action_gripper_right, label="Right Gripper (action)", 
            linewidth=1.5, linestyle='--', color='tab:red', alpha=0.5)

    ax.set_title(f"Episode {ep_idx} ({T} steps)", fontsize=12)
    ax.set_xlabel("Step")
    ax.set_ylabel("Gripper State (0=close, 100=open)")
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gripper_first_10_episodes.png", dpi=150)
plt.show()
print("图表已保存为 gripper_first_10_episodes.png")