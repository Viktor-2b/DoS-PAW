import numpy as np
import matplotlib.pyplot as plt

# 设置学术风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12


def calc_net_cost(omega_b, alpha, gamma, r1, r2, eta=0.1, beta=0.25, is_b_dos=False):
    """
    计算攻击者的归一化净成本率 (C_atk / c)
    (已修正：严格的时间加权份额与真实的物理电费成本)
    """
    delta = 1.0 - alpha - beta - eta
    if delta <= 0: return np.nan

    # 1. 计算稳态概率
    term1_mine_stop = ((1 - r1) * alpha) / (1 - r2 * alpha)
    term2_mine = (1 + r1 * alpha * (eta + delta)) / (1 - alpha)
    pi_0 = 1.0 / (term1_mine_stop + term2_mine)

    pi_1 = pi_0 * ((1 - r1) * alpha) / (1 - alpha)
    pi_2 = pi_0 * (r1 * alpha) / (1 - alpha)

    # === BDoS: 纯烧钱，无任何收益 ===
    if is_b_dos:
        return alpha * pi_0

    # 2. 竞争获胜概率
    p_a3 = (1 - r2) * alpha + gamma * (eta + delta)
    p_a4 = alpha + gamma * (beta + eta + delta)
    p_a5 = alpha + beta + gamma * (eta + delta)
    p_beta3 = 1 - p_a3

    # 3. 计算攻击者的独占区块收益 (仅限私有块竞争)
    r_block = beta * pi_1 * p_a3 + (eta + delta) * pi_1 * p_a4

    # === 修正的核心: 真实的时间加权分红比例 ===
    # 攻击者在所有状态下提交给矿池的 Share 总期望
    expected_attacker_shares = (r1 * alpha) * pi_0 + (r2 * alpha) * (pi_1 + pi_2)
    # 诚实矿工提交的 Share 总期望 (因为 pi_0+pi_1+pi_2=1，所以等于 beta)
    expected_honest_shares = beta

    if expected_honest_shares + expected_attacker_shares > 0:
        pool_total_revenue = beta * pi_0 + beta * pi_1 * p_beta3 + (eta + delta) * pi_2 * p_a5
        fraction = expected_attacker_shares / (expected_honest_shares + expected_attacker_shares)
        r_share = fraction * pool_total_revenue
    else:
        r_share = 0.0

    total_revenue = omega_b * (r_block + r_share)

    # === 修正的核心: 真实的电费支出 ===
    # State 0 满负荷运转，State 1&2 只有渗透算力在运转
    cost = alpha * pi_0 + (r2 * alpha) * (pi_1 + pi_2)

    return cost - total_revenue


def get_t_max_proportional(cost_array, alpha, w_unit, cap=1e6):
    """
    预算与算力规模成正比，即 Total_Budget = alpha * W_unit
    """
    t_max = []
    for c in cost_array:
        if c <= 0:
            t_max.append(cap)
        else:
            t_max.append(min((alpha * w_unit) / c, cap))
    return t_max


def plot_figure_4():
    print("正在计算按比例预算下的最大持续时间并生成单图...")

    # 因为转折点推迟了，我们将 X 轴范围调整为 1.0 到 3.0
    omega_bs = np.linspace(1.0, 3.0, 400)

    gamma_fixed = 0.5
    w_unit = 10.0  # 单位算力的预算

    alphas = [0.05, 0.10, 0.15, 0.20]
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#8c564b']

    # BDoS 基线
    cost_b_dos = [calc_net_cost(w, 0.20, gamma_fixed, 0.0, 0.0, is_b_dos=True) for w in omega_bs]
    t_b_dos = get_t_max_proportional(cost_b_dos, 0.20, w_unit)

    # DoS PAW 曲线
    t_paw_list = []
    asymptotes = []

    for a in alphas:
        costs = [calc_net_cost(w, a, gamma_fixed, r1=0.5, r2=0.99) for w in omega_bs]
        t_paw_list.append(get_t_max_proportional(costs, a, w_unit))

        # 寻找渐近线
        cross_idx = np.where(np.array(costs) <= 0)[0]
        if len(cross_idx) > 0:
            asymptotes.append(omega_bs[cross_idx[0]])
        else:
            asymptotes.append(None)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ax.plot(omega_bs, t_b_dos, label=r'BDoS Baseline ($\alpha=0.20$, Pure Burn)',
            color='#d62728', linestyle='--', linewidth=2.5, zorder=3)

    for i, a in enumerate(alphas):
        ax.plot(omega_bs, t_paw_list[i], label=rf'DoS PAW ($\alpha={a}$)',
                color=colors[i], linestyle='-', linewidth=2.5, zorder=4)

        if asymptotes[i] is not None:
            ax.axvline(asymptotes[i], color=colors[i], linestyle=':', linewidth=1.5, alpha=0.8, zorder=2)

    ax.text(2.95, 2e5, 'Vertical Asymptotes ($\infty$)\nAttack becomes self-sustaining',
            fontsize=12, color='dimgray', style='italic', ha='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    ax.set_title(r'Max Sustainable Time $T_{max}$ vs. Profitability Factor $\omega_b$ (Proportional Budget)',
                 fontsize=15, pad=12)
    ax.set_xlabel(r'Profitability Factor ($\omega_b$)', fontsize=13)
    ax.set_ylabel(r'Max Sustainable Time $T_{max}$ (Log Scale)', fontsize=13)
    ax.set_xlim(1.0, 3.0)

    # 对数坐标
    ax.set_yscale('log')
    ax.set_ylim(8, 1e6)
    ax.grid(True, which="both", linestyle=':', alpha=0.5, zorder=1)

    # 避开曲线，放至右下角
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)

    plt.tight_layout()
    plt.savefig('../figures/Figure_4.pdf', format='pdf')
    print("图表已保存至 ../figures 文件夹")
    plt.show()


if __name__ == '__main__':
    plot_figure_4()