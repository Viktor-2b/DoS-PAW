import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import matplotlib.patches as m_patches

# 设置学术风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12


def solve_alpha_crit(omega_b, gamma, r1, r2, eta=0.1, beta=0.25):
    """
    通过数值求根寻找临界攻击算力 alpha_crit
    """

    def utility_diff(alpha):
        delta = 1.0 - alpha - beta - eta
        if delta <= 0: return -1.0

        # S=Mine
        term1_mine_stop = ((1 - r1) * alpha) / (1 - r2 * alpha)
        term2_mine = (1+r1 * alpha * (eta + delta)) / (1 - alpha)
        pi_0_mine = 1.0 / (term1_mine_stop + term2_mine)

        pi_1_mine = pi_0_mine * ((1 - r1) * alpha) / (1 - alpha)
        pi_2_mine = pi_0_mine * (r1 * alpha) / (1 - alpha)

        p_a4_mine = alpha + gamma * (beta + eta + delta)
        p_a5_mine = alpha + beta + gamma * (eta + delta)
        p_4_mine = 1 - p_a4_mine
        p_5_mine = 1 - p_a5_mine
        p_eff_mine = pi_0_mine + pi_1_mine * p_4_mine + pi_2_mine * p_5_mine

        # S=Stop
        term2_stop = (1-eta+r1 * alpha * delta) / (beta+delta)
        pi_0_stop = 1.0 / (term1_mine_stop + term2_stop)

        return omega_b * (p_eff_mine - pi_0_stop) - (1 - pi_0_stop)

    # 边界判定: 极低算力就致死
    if utility_diff(0.0001) <= 0:
        return 0.0
        # 极高算力都无法致死
    if utility_diff(0.499) >= 0:
        return np.nan

    try:
        return brentq(utility_diff, 0.0001, 0.499)
    except ValueError:
        return np.nan


def plot_figure_2_shaded():
    print("正在求解稳态方程并生成带有阴影的 2x2 子图...")
    omega_bs = np.linspace(0, 12.0, 500)  # 提高分辨率让阴影更平滑

    gamma_list = [0.2, 0.4, 0.6, 0.8]
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=300, sharex=True, sharey=True)
    axes = axes.flatten()

    for i, gamma in enumerate(gamma_list):
        ax = axes[i]

        # 计算曲线
        raw_alpha_b_dos = [solve_alpha_crit(w, gamma=gamma, r1=0.0, r2=0.0) for w in omega_bs]
        raw_alpha_paw = [solve_alpha_crit(w, gamma=gamma, r1=0.5, r2=0.99) for w in omega_bs]

        # 处理 NaN 用于阴影填充 (将无穷大的门槛设为图表上限 0.5)
        fill_b_dos = np.array([0.5 if np.isnan(x) else x for x in raw_alpha_b_dos])
        fill_paw = np.array([0.5 if np.isnan(x) else x for x in raw_alpha_paw])

        # 1. 绘制绿色安全区 (在 PAW 下方)
        ax.fill_between(omega_bs, 0, fill_paw, color='#2ca02c', alpha=0.25)

        # 2. 绘制本论文的核心贡献区: 红色斜线区 (被 DoS PAW 攻破，但在 BDoS 下安全)
        ax.fill_between(omega_bs, fill_paw, fill_b_dos, facecolor='#ff7f0e', alpha=0.35, hatch='//', edgecolor='white')

        # 3. 绘制深红绝对危险区 (在 BDoS 上方)
        ax.fill_between(omega_bs, fill_b_dos, 0.5, color='#d62728', alpha=0.35)

        # 绘制边界实线
        ax.plot(omega_bs, raw_alpha_b_dos, color='black', linestyle='-', linewidth=2.0, zorder=4)
        ax.plot(omega_bs, raw_alpha_paw, color='black', linestyle='--', linewidth=2.0, zorder=4)

        ax.set_title(rf'{subplot_labels[i]} Propagation Advantage $\gamma={gamma}$', fontsize=14, pad=10)
        ax.set_ylim(0, 0.5)
        ax.set_xlim(0, 12.0)
        ax.grid(True, linestyle=':', alpha=0.5, zorder=0)

        if i % 2 == 0:
            ax.set_ylabel(r'Critical Attack Threshold ($\alpha_{crit}$)', fontsize=13)
        if i >= 2:
            ax.set_xlabel(r'Profitability Factor ($\omega_b$)', fontsize=13)

    # 统一图例设计 (Legend)
    # 创建自定义图例 proxy artists
    green_patch = m_patches.Patch(color='#2ca02c', alpha=0.25, label='Secure Region for both')
    hatch_patch = m_patches.Patch(facecolor='#ff7f0e', alpha=0.35, hatch='//', edgecolor='white',
                                  label='Vulnerable to DoS PAW (Ours)')
    red_patch = m_patches.Patch(color='#d62728', alpha=0.35, label='Vulnerable to both')
    line_b_dos = plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2.0, label='BDoS Boundary')
    line_paw = plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2.0, label='DoS PAW Boundary')

    # 将全局图例放在顶部右侧
    axes[1].legend(handles=[line_b_dos, line_paw, green_patch, hatch_patch, red_patch],
               loc='upper right', fontsize=10, framealpha=0.95)

    plt.tight_layout()  # 留出顶部大标题和图例的空间
    plt.savefig('Figure_2.pdf', format='pdf')
    print("图表已保存为 Figure_2.pdf")
    plt.show()


if __name__ == '__main__':
    plot_figure_2_shaded()