import numpy as np
import matplotlib.pyplot as plt

# 设置学术风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12


def calc_net_cost(omega_b, alpha, gamma, r1, r2, eta=0.1, beta=0.25, is_b_dos=False):
    """
    计算攻击者的归一化净成本率 (C_atk / c)
    """
    delta = 1.0 - alpha - beta - eta
    if delta <= 0: return np.nan

    # 计算 S=Mine 下的稳态概率
    term1_mine_stop = ((1 - r1) * alpha) / (1 - r2 * alpha)
    term2_mine = (1 + r1 * alpha * (eta + delta)) / (1 - alpha)
    pi_0 = 1.0 / (term1_mine_stop + term2_mine)

    pi_1 = pi_0 * ((1 - r1) * alpha) / (1 - alpha)
    pi_2 = pi_0 * (r1 * alpha) / (1 - alpha)

    if is_b_dos: return alpha * pi_0 # BDoS零收益
    # 竞争获胜概率
    p_a3 = (1 - r2) * alpha + gamma * (eta + delta)
    p_a4 = alpha + gamma * (beta + eta + delta)
    p_a5 = alpha + beta + gamma * (eta + delta)
    p_beta3 = 1 - p_a3

    # 3. 计算攻击者的收益
    r_block = beta * pi_1 * p_a3 + (eta + delta) * pi_1 * p_a4

    if beta + r2 * alpha > 0:
        pool_total_revenue = beta * pi_0 + beta * pi_1 * p_beta3 + (eta + delta) * pi_2 * p_a5
        r_share = (r2 * alpha) / (beta + r2 * alpha) * pool_total_revenue
    else:
        r_share = 0.0

    # 4. 净成本 = 成本 - 收益
    total_revenue = omega_b * (r_block + r_share)
    cost = alpha * pi_0

    return cost - total_revenue


def plot_figure_3():
    print("正在计算攻击者净成本率并生成合并图表...")
    omega_bs = np.linspace(1.0, 6.0, 200)

    gamma_fixed = 0.5

    # === 计算 BDoS 基线 (以最大的 alpha=0.25 为例展示) ===
    cost_b_dos = [calc_net_cost(w, 0.25, gamma_fixed, 0.0, 0.0, is_b_dos=True) for w in omega_bs]

    # === 计算 DoS PAW 的多条曲线 ===
    alphas = [0.05, 0.15, 0.25]
    colors_alpha = ['#ff7f0e', '#2ca02c', '#1f77b4']
    costs_paw_alphas = [
        [calc_net_cost(w, a, gamma_fixed, r1=0.5, r2=0.99) for w in omega_bs]
        for a in alphas
    ]

    # === 开始绘图 (单图结构) ===
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 1. 绘制背景阴影与 0 轴分界线
    ax.axhline(0, color='black', linestyle='-', linewidth=1.2, zorder=3)
    ax.axhspan(-0.25, 0, facecolor='#2ca02c', alpha=0.15, zorder=0)
    ax.axhspan(0, 0.25, facecolor='#d62728', alpha=0.1, zorder=0)

    # 2. 绘制 BDoS 曲线
    ax.plot(omega_bs, cost_b_dos, label=r'BDoS Baseline ($\alpha=0.25$)', color='#d62728', linestyle='--', linewidth=2.5,
            zorder=4)

    # 3. 绘制 DoS PAW 曲线
    for i, a in enumerate(alphas):
        ax.plot(omega_bs, costs_paw_alphas[i], label=rf'DoS PAW ($\alpha={a}$)',
                color=colors_alpha[i], linestyle='-', linewidth=2.5, zorder=4)

    # 4. 在右侧靠近 0 轴的位置添加文本
    # 使用 ha='right' 使其靠右对齐，va='bottom'/'top' 使其紧贴 0 轴上下
    ax.text(5.9, 0.015, 'Cost-Incurring Zone (Capital Burn)',
            color='darkred', fontsize=12, fontweight='bold', ha='right', va='bottom', zorder=5)
    ax.text(5.9, -0.015, 'Self-Sustaining Zone (Profitable)',
            color='darkgreen', fontsize=12, fontweight='bold', ha='right', va='top', zorder=5)

    # 5. 格式化坐标轴
    ax.set_xlabel(r'Profitability Factor ($\omega_b$)', fontsize=13)
    ax.set_ylabel(r'Normalized Net Cost Rate ($\mathcal{C}_{atk} / c$)', fontsize=13)
    ax.set_xlim(1.0, 6.0)
    ax.set_ylim(-0.25, 0.25)
    ax.grid(True, linestyle=':', alpha=0.6, zorder=1)

    # 6. 设置图例
    ax.legend(loc='lower right', bbox_to_anchor=(0.99, 0.15), fontsize=11, framealpha=0.95)

    plt.tight_layout()
    plt.savefig('../figures/Figure_3.pdf', format='pdf')
    print("图表已保存至 ../figures 文件夹")
    plt.show()


if __name__ == '__main__':
    plot_figure_3()