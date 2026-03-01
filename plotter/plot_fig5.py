import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 设置学术风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12


def calc_fitness_mine(x, omega_b, alpha, beta, eta, delta, gamma, r1, r2, is_b_dos=False):
    """
    计算选择 Mine 的适应度
    """
    # 计算当前存活的目标矿工算力
    eta = x * eta

    honest = beta + eta + delta
    active = alpha + honest
    if honest <= 0:
        return -1.0  # 诚实矿工全退
    # 根据全 CTMC 计算稳态概率，注意带有退网的不能用1减推算
    term1 = ((1 - r1) * alpha) / (active - r2 * alpha)
    term2 = (active + r1 * alpha * (eta + delta)) / honest
    pi_0 = 1.0 / (term1 + term2)
    pi_1 = pi_0 * ((1 - r1) * alpha) / honest
    pi_2 = pi_0 * (r1 * alpha) / honest

    # 计算 Race 获胜概率 p_race(x)
    p_a4 = (alpha + gamma * (beta + eta + delta)) / active
    if is_b_dos:
        p_a5 = p_a4  # BDoS 没有渗透，不存在 IvE
    else:
        p_a5 = (alpha + beta + gamma * (eta + delta)) / active

    # 外部矿工获胜的概率
    p_4 = 1.0 - p_a4
    p_5 = 1.0 - p_a5

    p_eff_mine = pi_0 + pi_1 * p_4 + pi_2 * p_5

    # f_Mine(x) = c * [omega_b * p_race(x) - 1] (假设 c=1 进行归一化)
    return omega_b * p_eff_mine - 1.0


def replicator_dynamics(x, t, omega_b, alpha, beta, eta, delta, gamma, r1, r2, kappa, is_b_dos):
    """
    使用 np.clip 解决底层 Runge-Kutta 求解器的边界震荡问题
    """
    # 保证求解器内部的 x 永远处于合法概率区间，防止分母除以 0 或负数
    x_safe = np.clip(x, 1e-5, 0.99999)

    # 计算适应度 (效用差)
    f_mine = calc_fitness_mine(x_safe, omega_b, alpha, beta, eta, delta, gamma, r1, r2, is_b_dos)

    # 复制子动态方程
    dx_dt = kappa * x_safe * (1.0 - x_safe) * f_mine

    return dx_dt


def plot_figure_5():
    print("正在求解系统演化常微分方程 (ODE)...")

    # 时间轴 (模拟 100 个演化周期)
    t = np.linspace(0, 100, 1000)

    # --- 修正的核心实验参数 ---
    omega_b = 1.25
    alpha = 0.2
    beta = 0.2
    eta = 0.4
    delta = 0.2
    gamma = 0.5

    # 将演化速率常数调低，让博弈过程舒展在整个时间轴上
    kappa = 1.0

    x0 = 0.99  # 初始状态，必须给一个扰动

    # 1. 求解 BDoS 下的演化轨迹
    x_b_dos = odeint(replicator_dynamics, x0, t, args=(omega_b, alpha, beta, eta, delta, gamma, 0.0, 0.0, kappa, True))

    # 2. 求解 DoS PAW 下的演化轨迹
    x_paw = odeint(replicator_dynamics, x0, t,
                   args=(omega_b, alpha, beta, eta, delta, gamma, 0.5, 0.99, kappa, False))

    # === 开始绘图 ===
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    # 绘制阈值线
    collapse_threshold = 0.20
    ax.axhspan(0, collapse_threshold, facecolor='#d62728', alpha=0.1, zorder=0)
    ax.axhline(collapse_threshold, color='dimgray', linestyle=':', linewidth=1.5, zorder=1)
    ax.text(80, collapse_threshold + 0.02, 'Collapse Threshold ($x=0.2$)', color='dimgray', fontsize=11, ha='center')
    ax.text(80, 0.1, 'System Collapse Zone ($x \leq 0.2$)', color='darkred', fontsize=12, fontweight='bold',
            ha='center', va='center')
    # 绘制演化曲线
    ax.plot(t, x_b_dos.flatten(), label=r'BDoS', color='#d62728', linestyle='--', linewidth=2.5,
            zorder=4)
    ax.plot(t, x_paw.flatten(), label=r'DoS PAW', color='#1f77b4', linestyle='-',
            linewidth=2.5, zorder=4)

    # 填充差值区域
    ax.fill_between(t, x_paw.flatten(), x_b_dos.flatten(), facecolor='#ff7f0e', alpha=0.3, hatch='//',
                    edgecolor='white', label='Accelerated Liveness Loss', zorder=2)

    ax.set_xlabel(r'Evolutionary Time Step $t$', fontsize=13)
    ax.set_ylabel(r'Proportion of Active Victim Miners $x(t)$', fontsize=13)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=':', alpha=0.6, zorder=1)

    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

    plt.tight_layout()
    plt.savefig('../figures/Figure_5.pdf', format='pdf')
    print("图表已保存至 ../figures 文件夹")
    plt.show()


if __name__ == '__main__':
    plot_figure_5()
