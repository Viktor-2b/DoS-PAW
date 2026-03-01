import numpy as np
import matplotlib.pyplot as plt

# 设置学术风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12


def calc_strategy_utilities(alpha, omega_b, gamma, r1, r2, beta, eta):
    """
    计算目标矿工 eta 在三种策略下的归一化效用。
    公式已与论文 Eq. 13, Eq. 15, Eq. 18 达到字面上的绝对对齐！
    """
    delta = 1.0 - alpha - beta - eta
    if delta < 0:
        return np.nan, np.nan, np.nan

    # ================= 1. Mine 策略 (对齐 Eq. 13 & Table 2) =================
    inv_pi_0_mine = ((1 - r1) * alpha) / (1 - r2 * alpha) + (1.0 + r1 * alpha * (eta + delta)) / (1 - alpha)
    pi_0_mine = 1.0 / inv_pi_0_mine
    pi_1_mine = pi_0_mine * ((1 - r1) * alpha) / (1 - alpha)
    pi_2_mine = pi_0_mine * (r1 * alpha) / (1 - alpha)

    p_a4_mine = alpha + gamma * (beta + eta + delta)
    p_a5_mine = alpha + beta + gamma * (eta + delta)
    # 防止概率溢出 1.0
    p_4_mine = max(0.0, 1.0 - p_a4_mine)
    p_5_mine = max(0.0, 1.0 - p_a5_mine)

    p_eff_mine = pi_0_mine + pi_1_mine * p_4_mine + pi_2_mine * p_5_mine
    u_mine = omega_b * p_eff_mine - 1.0

    # ================= 2. Stop 策略 (对齐 Eq. 15) =================
    inv_pi_0_stop = ((1 - r1) * alpha) / (1 - r2 * alpha) + (1.0 - eta + r1 * alpha * delta) / (beta + delta)
    pi_0_stop = 1.0 / inv_pi_0_stop
    u_stop = pi_0_stop * (omega_b - 1.0)

    # ================= 3. SPV 策略 (对齐 Eq. 18) =================
    inv_pi_0_spv = (((1 - r1) * alpha) / (1 - r2 * alpha)) * ((beta + delta) / (1 - alpha)) + (
                1.0 + r1 * alpha * delta) / (1 - alpha)
    pi_0_spv = 1.0 / inv_pi_0_spv
    u_spv = omega_b * pi_0_spv - 1.0

    return u_mine, u_stop, u_spv


def plot_figure_6():
    print("正在计算响应策略效用并生成 Figure 6...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    # 固定基准参数
    beta_fixed = 0.20
    eta_fixed = 0.10
    r1_fixed = 0.50
    r2_fixed = 0.99

    # --- (a) Utility vs. Attack Power ---
    alphas = np.linspace(0.05, 0.45, 100)
    u_m_a, u_s_a, u_spv_a = [], [], []
    for a in alphas:
        um, us, uspv = calc_strategy_utilities(a, 1.2, 0.5, r1_fixed, r2_fixed, beta_fixed, eta_fixed)
        u_m_a.append(um);
        u_s_a.append(us);
        u_spv_a.append(uspv)

    axes[0].plot(alphas, u_m_a, label=r'Mine ($S=\text{Mine}$)', color='#1f77b4', lw=2.5)
    axes[0].plot(alphas, u_s_a, label=r'Stop ($S=\text{Stop}$)', color='#2ca02c', lw=2.5)
    axes[0].plot(alphas, u_spv_a, label=r'SPV ($S=\text{SPV}$)', color='#d62728', lw=2.5)

    idx_cross = np.argwhere(np.array(u_s_a) > np.array(u_m_a))[0][0]
    alpha_crit = alphas[idx_cross]
    axes[0].axvline(alpha_crit, color='gray', linestyle=':', lw=1.5)
    axes[0].text(alpha_crit + 0.01, max(u_m_a), rf'$\alpha_{{crit}} \approx {alpha_crit:.2f}$', color='dimgray')
    axes[0].set_xlabel(r'Attack Power ($\alpha$)')
    axes[0].set_ylabel(r'Expected Utility ($U_{\eta}$)')
    axes[0].set_title(r'(a) Utility vs. Attack Power $\alpha$')
    axes[0].legend()

    # --- (b) Utility vs. Profitability Factor ---
    omega_bs = np.linspace(1.0, 2.5, 100)
    u_m_b, u_s_b, u_spv_b = [], [], []
    for w in omega_bs:
        um, us, uspv = calc_strategy_utilities(0.25, w, 0.5, r1_fixed, r2_fixed, beta_fixed, eta_fixed)
        u_m_b.append(um);
        u_s_b.append(us);
        u_spv_b.append(uspv)

    axes[1].plot(omega_bs, u_m_b, color='#1f77b4', lw=2.5)
    axes[1].plot(omega_bs, u_s_b, color='#2ca02c', lw=2.5)
    axes[1].plot(omega_bs, u_spv_b, color='#d62728', lw=2.5)
    axes[1].set_xlabel(r'Profitability Factor ($\omega_b$)')
    axes[1].set_title(r'(b) Utility vs. Profitability Factor $\omega_b$')

    # --- (c) Utility vs. Propagation Advantage ---
    gammas = np.linspace(0.0, 1.0, 100)
    u_m_c, u_s_c, u_spv_c = [], [], []
    for g in gammas:
        um, us, uspv = calc_strategy_utilities(0.25, 1.2, g, r1_fixed, r2_fixed, beta_fixed, eta_fixed)
        u_m_c.append(um);
        u_s_c.append(us);
        u_spv_c.append(uspv)

    axes[2].plot(gammas, u_m_c, color='#1f77b4', lw=2.5)
    axes[2].plot(gammas, u_s_c, color='#2ca02c', lw=2.5)
    axes[2].plot(gammas, u_spv_c, color='#d62728', lw=2.5)
    axes[2].set_xlabel(r'Propagation Advantage ($\gamma$)')
    axes[2].set_title(r'(c) Utility vs. Propagation Advantage $\gamma$')

    for ax in axes:
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('../figures/Figure_6.pdf', format='pdf')
    print("图表已保存至 ../figures 文件夹")
    plt.show()


if __name__ == '__main__':
    plot_figure_6()