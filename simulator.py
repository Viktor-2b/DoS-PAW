import numpy as np
import matplotlib.pyplot as plt
import random

# 设置学术风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12


# =====================================================================
# 1. 理论公式部分 (Theoretical CTMC) - 作为 Ground Truth 线条
# =====================================================================
def calc_net_cost_theoretical(omega_b, alpha, gamma, r1, r2, eta=0.1, beta=0.25, is_b_dos=False):
    delta = 1.0 - alpha - beta - eta
    if delta <= 0: return np.nan

    term1_mine = ((1 - r1) * alpha) / (1 - r2 * alpha)
    term2_mine = (1 + r1 * alpha * (eta + delta)) / (1 - alpha)
    pi_0 = 1.0 / (term1_mine + term2_mine)
    pi_1 = pi_0 * ((1 - r1) * alpha) / (1 - alpha)
    pi_2 = pi_0 * (r1 * alpha) / (1 - alpha)

    if is_b_dos: return alpha * pi_0

    p_a3 = (1 - r2) * alpha + gamma * (eta + delta)
    p_a4 = alpha + gamma * (beta + eta + delta)
    p_a5 = alpha + beta + gamma * (eta + delta)
    p_beta3 = 1 - p_a3

    r_block = beta * pi_1 * p_a3 + (eta + delta) * pi_1 * p_a4

    expected_attacker_shares = (r1 * alpha) * pi_0 + (r2 * alpha) * (pi_1 + pi_2)
    expected_honest_shares = beta

    if expected_honest_shares + expected_attacker_shares > 0:
        pool_total_revenue = beta * pi_0 + beta * pi_1 * p_beta3 + (eta + delta) * pi_2 * p_a5
        fraction = expected_attacker_shares / (expected_honest_shares + expected_attacker_shares)
        r_share = fraction * pool_total_revenue
    else:
        r_share = 0.0

    total_revenue = omega_b * (r_block + r_share)
    cost = alpha * pi_0 + (r2 * alpha) * (pi_1 + pi_2)

    return cost - total_revenue


# =====================================================================
# 2. 蒙特卡洛离散事件模拟器 (Monte Carlo DES) - 作为散点验证
# =====================================================================
class Simulator:
    def __init__(self, alpha, beta=0.25, eta=0.1, gamma=0.5, r1=0.5, r2=0.99, is_bdos=False):
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.delta = 1.0 - alpha - beta - eta
        self.gamma = gamma
        self.r1 = 0.0 if is_bdos else r1
        self.r2 = 0.0 if is_bdos else r2
        self.is_bdos = is_bdos

        # S=Mine 时的竞争胜率 (对应全网算力为 1.0)
        self.pa3 = (1.0 - self.r2) * alpha + gamma * (eta + self.delta)
        self.pa4 = alpha + gamma * (beta + eta + self.delta)
        self.pa5 = alpha + beta + gamma * (eta + self.delta)

    def run(self, omega_b, num_events=100000):
        state = 0
        time_elapsed = 0.0

        attacker_blocks = 0
        pool_blocks = 0
        attacker_shares = 0.0
        honest_shares = 0.0
        attacker_electricity = 0.0

        for _ in range(num_events):
            if state == 0:
                rate_pri = (1.0 - self.r1) * self.alpha
                rate_inf = self.r1 * self.alpha
                rate_pool = self.beta
                rate_out = self.eta + self.delta
                total_rate = rate_pri + rate_inf + rate_pool + rate_out  # = 1.0

                # 采样物理时间
                dt = random.expovariate(total_rate)
                time_elapsed += dt

                # 累加份额与耗电
                attacker_shares += self.r1 * self.alpha * dt
                honest_shares += self.beta * dt
                attacker_electricity += self.alpha * dt

                # 事件轮盘赌
                rand = random.random() * total_rate
                if rand < rate_pri:
                    state = 1
                elif rand < rate_pri + rate_inf:
                    state = 2
                elif rand < rate_pri + rate_inf + rate_pool:
                    pool_blocks += 1
                else:
                    pass

            elif state == 1:
                rate_pool = self.beta
                rate_out = self.eta + self.delta
                total_rate = rate_pool + rate_out

                dt = random.expovariate(total_rate)
                time_elapsed += dt

                # BDoS 在扣块时完全关机 (r2=0)，PAW 只有渗透算力开机 (r2=0.99)
                attacker_shares += self.r2 * self.alpha * dt
                honest_shares += self.beta * dt
                attacker_electricity += self.r2 * self.alpha * dt

                rand = random.random() * total_rate
                if rand < rate_pool:
                    # PvV Race
                    if random.random() < self.pa3:
                        attacker_blocks += 1
                    else:
                        pool_blocks += 1
                else:
                    # PvE Race
                    if random.random() < self.pa4: attacker_blocks += 1

                state = 0

            elif state == 2:
                rate_pool = self.beta
                rate_out = self.eta + self.delta
                total_rate = rate_pool + rate_out

                dt = random.expovariate(total_rate)
                time_elapsed += dt

                attacker_shares += self.r2 * self.alpha * dt
                honest_shares += self.beta * dt
                attacker_electricity += self.r2 * self.alpha * dt

                rand = random.random() * total_rate
                if rand < rate_pool:
                    # 触发双重陷阱，区块全部作废
                    pass
                else:
                    # IvE Race: 如果攻击者用渗透块打赢了外部矿工，奖励算矿池的！
                    if random.random() < self.pa5: pool_blocks += 1

                state = 0

        # 结算 BDoS
        if self.is_bdos:
            attacker_blocks = 0
            attacker_shares = 0

        # 计算 PPoW 分红比例
        share_ratio = attacker_shares / (attacker_shares + honest_shares) if (
                                                                                         attacker_shares + honest_shares) > 0 else 0
        pool_payout_to_attacker = pool_blocks * share_ratio

        # 结算单位时间的期望净成本率
        total_revenue_rate = (attacker_blocks + pool_payout_to_attacker) / time_elapsed
        cost_rate = attacker_electricity / time_elapsed

        return cost_rate - omega_b * total_revenue_rate


# =====================================================================
# 3. 绘图：理论曲线 + 蒙特卡洛散点
# =====================================================================
def plot_figure_3_with_MC():
    print("正在计算理论曲线...")
    omega_bs_line = np.linspace(1.0, 6.0, 200)
    gamma_fixed = 0.5

    cost_b_dos_line = [calc_net_cost_theoretical(w, 0.25, gamma_fixed, 0.0, 0.0, is_b_dos=True) for w in omega_bs_line]

    alphas = [0.05, 0.15, 0.25]
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
    costs_paw_lines = [[calc_net_cost_theoretical(w, a, gamma_fixed, 0.5, 0.99) for w in omega_bs_line] for a in alphas]

    print("正在运行蒙特卡洛模拟器进行验证 (每个点 10万 步)...")
    # 选择几个特定的 omega_b 锚点进行蒙特卡洛打点
    omega_bs_scatter = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    # 实例化模拟器
    sim_bdos = Simulator(alpha=0.25, gamma=gamma_fixed, is_bdos=True)
    cost_b_dos_mc = [sim_bdos.run(w) for w in omega_bs_scatter]

    costs_paw_mcs = []
    for a in alphas:
        sim_paw = Simulator(alpha=a, gamma=gamma_fixed, r1=0.5, r2=0.99, is_bdos=False)
        costs_paw_mcs.append([sim_paw.run(w) for w in omega_bs_scatter])

    # === 开始绘图 ===
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 背景与分界线
    ax.axhline(0, color='black', linestyle='-', linewidth=1.2, zorder=3)
    ax.axhspan(-0.25, 0, facecolor='#2ca02c', alpha=0.15, zorder=0)
    ax.axhspan(0, 0.25, facecolor='#d62728', alpha=0.1, zorder=0)

    # 1. 绘制理论曲线 (Lines)
    ax.plot(omega_bs_line, cost_b_dos_line, color='#d62728', linestyle='--', linewidth=2.5, zorder=4,
            label='_nolegend_')
    for i, a in enumerate(alphas):
        ax.plot(omega_bs_line, costs_paw_lines[i], color=colors[i], linestyle='-', linewidth=2.5, zorder=4,
                label='_nolegend_')

    # 2. 绘制蒙特卡洛散点 (Scatters)
    ax.scatter(omega_bs_scatter, cost_b_dos_mc, color='#d62728', marker='X', s=100, zorder=5,
               label=r'BDoS Baseline ($\alpha=0.25$)')
    for i, a in enumerate(alphas):
        ax.scatter(omega_bs_scatter, costs_paw_mcs[i], color=colors[i], marker='o', s=80, edgecolors='black',
                   linewidths=0.8, zorder=5, label=rf'DoS PAW ($\alpha={a}$)')

    # 文本与格式化
    ax.text(5.9, 0.015, 'Cost-Incurring Zone (Capital Burn)', color='darkred', fontsize=12, fontweight='bold',
            ha='right', va='bottom', zorder=5)
    ax.text(5.9, -0.015, 'Self-Sustaining Zone (Profitable)', color='darkgreen', fontsize=12, fontweight='bold',
            ha='right', va='top', zorder=5)

    ax.set_title("Theoretical Analysis vs. Monte Carlo Simulation", fontsize=15, pad=12)
    ax.set_xlabel(r'Profitability Factor ($\omega_b$)', fontsize=13)
    ax.set_ylabel(r'Normalized Net Cost Rate ($\mathcal{C}_{atk} / c$)', fontsize=13)
    ax.set_xlim(1.0, 6.0)
    ax.set_ylim(-0.25, 0.25)
    ax.grid(True, linestyle=':', alpha=0.6, zorder=1)

    # 补充图例说明 (Lines = Theory, Markers = Simulation)
    ax.plot([], [], color='gray', linestyle='-', linewidth=2.5, label='Lines: Theoretical Model')
    ax.scatter([], [], color='gray', marker='o', s=80, edgecolors='black', label='Markers: Monte Carlo Sim')

    ax.legend(loc='lower left', fontsize=11, framealpha=0.95)

    plt.tight_layout()
    plt.savefig('./figures/Figure_3_with_MC.pdf', format='pdf')
    print("图表已保存至 ./figures 文件夹")
    plt.show()


if __name__ == '__main__':
    plot_figure_3_with_MC()