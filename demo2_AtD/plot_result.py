import numpy as np
import matplotlib.pyplot as plt

# --- 配置参数 (需与 main.c 一致) ---
nx, ny = 101, 101
dx, dy = 10.0, 10.0

# --- 读取数据的函数 ---
def read_bin(filename):
    try:
        data = np.fromfile(filename, dtype=np.float64)
        return data.reshape((ny, nx))
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return np.zeros((ny, nx))

def read_cost(filename):
    try:
        data = np.loadtxt(filename)
        if data.ndim == 1: return np.array([data[0]]), np.array([data[1]])
        return data[:, 0], data[:, 1]
    except:
        return [], []

# --- 1. 加载数据 ---
z_true = read_bin("z_true_init.bin")          # 真实值
z_guess = read_bin("z_guess_init.bin")        # 初始猜测 (未同化前)
z_final = read_bin("z_final_analysis.bin")    # 最终分析值 (同化后)
iters, costs = read_cost("cost_history.txt")  # 误差下降曲线

# 计算差值图 (Error)
diff_initial = z_guess - z_true
diff_final = z_final - z_true

# --- 2. 绘图 ---
fig = plt.figure(figsize=(18, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.2)

# === 第一行：2D 空间场对比 ===
# 用来展示波的位置
vmin, vmax = -0.5, 1.0 # 固定色标范围方便对比

ax1 = plt.subplot(2, 3, 1)
im1 = ax1.imshow(z_true, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
ax1.set_title("(A) True State (Target)", fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

ax2 = plt.subplot(2, 3, 2)
im2 = ax2.imshow(z_guess, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
ax2.set_title("(B) Initial Guess (Before 4DVar)", fontsize=12)
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

ax3 = plt.subplot(2, 3, 3)
im3 = ax3.imshow(z_final, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
ax3.set_title("(C) Final Analysis (After 4DVar)", fontsize=12, fontweight='bold')
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

# === 第二行：定量分析 ===

# 子图 4：1D 截面波形对比 (最直观的判断方式)
# 取中间一行 (y=50) 切片
mid_y = ny // 2
ax4 = plt.subplot(2, 2, 3)
ax4.plot(z_true[mid_y, :], 'k-', linewidth=2.5, label='True State')
ax4.plot(z_guess[mid_y, :], 'b--', linewidth=1.5, label='Initial Guess')
ax4.plot(z_final[mid_y, :], 'r.-', linewidth=1.5, markersize=8, label='Final Analysis')
ax4.set_title("(D) Cross-section at Y=50 (1D Profile)", fontsize=12, fontweight='bold')
ax4.set_xlabel("Grid X")
ax4.set_ylabel("Water Level (z)")
ax4.legend()
ax4.grid(True, linestyle=':', alpha=0.6)

# 子图 5：Cost Function 下降曲线 (判断收敛)
ax5 = plt.subplot(2, 2, 4)
if len(iters) > 0:
    ax5.semilogy(iters, costs, 'o-', color='purple', linewidth=2)
    ax5.set_title("(E) Cost Function J vs Iteration", fontsize=12, fontweight='bold')
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("Cost J (Log Scale)")
    ax5.grid(True, which="both", ls="-", alpha=0.5)
    # 标注起始和结束值
    ax5.text(iters[0], costs[0], f"{costs[0]:.1e}", verticalalignment='bottom')
    ax5.text(iters[-1], costs[-1], f"{costs[-1]:.1e}", verticalalignment='top', color='red')
else:
    ax5.text(0.5, 0.5, "No Cost History Found", ha='center')

plt.suptitle(f"4DVar Assimilation Results Summary", fontsize=16)
plt.savefig("4dvar_report.png", dpi=150)
plt.show()
