import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= 设置学术风格 =================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12


def find_closest(array, target):
    """辅助函数：在 CSV 离散网格中寻找最接近目标的参数值"""
    return array.iloc[np.argmin(np.abs(array - target))]


def plot_figure_3(df):
    """绘制 Figure 3: 攻击者净成本率 (Net Cost) 曲线"""
    print("正在绘制 Figure 3...")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 过滤基准环境：目标矿工坚持 Mine，传播优势 gamma 约等于 0.5
    gamma_val = find_closest(pd.Series(df['gamma'].unique()), 0.5)
    df_mine = df[(df['Target_Strategy'] == 'Mine') & (df['gamma'] == gamma_val)]

    # 我们挑三根 alpha 线来画 (由于linspace生成，找最接近 0.05, 0.15, 0.25 的值)
    alphas_to_plot = [
        find_closest(pd.Series(df_mine['alpha'].unique()), 0.05),
        find_closest(pd.Series(df_mine['alpha'].unique()), 0.15),
        find_closest(pd.Series(df_mine['alpha'].unique()), 0.25)
    ]
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

    # 画背景区
    ax.axhline(0, color='black', linestyle='-', linewidth=1.2, zorder=3)
    ax.axhspan(-0.2, 0, facecolor='#2ca02c', alpha=0.15, zorder=0)
    ax.axhspan(0, 0.2, facecolor='#d62728', alpha=0.1, zorder=0)

    for i, a in enumerate(alphas_to_plot):
        df_a = df_mine[df_mine['alpha'] == a].sort_values('omega_b')
        # 【核心逻辑】：论文中的 Net Cost = -U_A (效用的相反数就是成本)
        net_cost = -df_a['Attacker_Utility'].values

        ax.plot(df_a['omega_b'], net_cost, color=colors[i], marker='o', markersize=6,
                linestyle='-', linewidth=2, label=rf'DoS PAW ($\alpha \approx {a:.2f}$)', zorder=4)

    ax.text(2.4, 0.01, 'Cost-Incurring Zone (Capital Burn)', color='darkred', fontweight='bold', ha='right',
            va='bottom')
    ax.text(2.4, -0.01, 'Self-Sustaining Zone (Profitable)', color='darkgreen', fontweight='bold', ha='right', va='top')

    ax.set_xlabel(r'Profitability Factor ($\omega_b$)', fontsize=13)
    ax.set_ylabel(r'Normalized Net Cost Rate ($\mathcal{C}_{atk} / c$)', fontsize=13)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig('Figure_3_from_CSV.pdf')
    print("Figure 3 已保存！")


def plot_figure_6(df):
    """绘制 Figure 6: 目标矿工的响应策略效用对比"""
    print("正在绘制 Figure 6...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    # 策略颜色与标记映射
    styles = {
        'Mine': {'color': '#1f77b4', 'label': r'Mine ($S=\text{Mine}$)'},
        'Stop': {'color': '#2ca02c', 'label': r'Stop ($S=\text{Stop}$)'},
        'SPV': {'color': '#d62728', 'label': r'SPV ($S=\text{SPV}$)'}
    }

    # -------------- (a) U_target vs alpha --------------
    omega_val_a = find_closest(pd.Series(df['omega_b'].unique()), 1.2)
    gamma_val_a = find_closest(pd.Series(df['gamma'].unique()), 0.5)
    df_a = df[(df['omega_b'] == omega_val_a) & (df['gamma'] == gamma_val_a)]

    for strategy, style in styles.items():
        sub_df = df_a[df_a['Target_Strategy'] == strategy].sort_values('alpha')
        axes[0].plot(sub_df['alpha'], sub_df['Target_Utility'], color=style['color'],
                     marker='^' if strategy == 'SPV' else 'o', linewidth=2.5, label=style['label'])
    axes[0].set_xlabel(r'Attack Power ($\alpha$)')
    axes[0].set_ylabel(r'Target Expected Utility ($U_{\eta}$)')
    axes[0].set_title(r'(a) Utility vs. Attack Power $\alpha$')
    axes[0].legend()

    # -------------- (b) U_target vs omega_b --------------
    alpha_val_b = find_closest(pd.Series(df['alpha'].unique()), 0.25)
    gamma_val_b = find_closest(pd.Series(df['gamma'].unique()), 0.5)
    df_b = df[(df['alpha'] == alpha_val_b) & (df['gamma'] == gamma_val_b)]

    for strategy, style in styles.items():
        sub_df = df_b[df_b['Target_Strategy'] == strategy].sort_values('omega_b')
        axes[1].plot(sub_df['omega_b'], sub_df['Target_Utility'], color=style['color'],
                     marker='^' if strategy == 'SPV' else 'o', linewidth=2.5)
    axes[1].set_xlabel(r'Profitability Factor ($\omega_b$)')
    axes[1].set_title(r'(b) Utility vs. Profitability Factor $\omega_b$')

    # -------------- (c) U_target vs gamma --------------
    alpha_val_c = find_closest(pd.Series(df['alpha'].unique()), 0.25)
    omega_val_c = find_closest(pd.Series(df['omega_b'].unique()), 1.2)
    df_c = df[(df['alpha'] == alpha_val_c) & (df['omega_b'] == omega_val_c)]

    for strategy, style in styles.items():
        sub_df = df_c[df_c['Target_Strategy'] == strategy].sort_values('gamma')
        axes[2].plot(sub_df['gamma'], sub_df['Target_Utility'], color=style['color'],
                     marker='^' if strategy == 'SPV' else 'o', linewidth=2.5)
    axes[2].set_xlabel(r'Propagation Advantage ($\gamma$)')
    axes[2].set_title(r'(c) Utility vs. Propagation Advantage $\gamma$')

    for ax in axes:
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig('Figure_6_from_CSV.pdf')
    print("Figure 6 已保存！")


if __name__ == '__main__':
    # 读取你生成的完整仿真数据
    df = pd.read_csv('../simulation_results.csv')

    # 执行绘图
    plot_figure_3(df)
    plot_figure_6(df)