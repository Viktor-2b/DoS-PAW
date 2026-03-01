import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as m_dates

# 设置学术风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12


def calculate_cost_btc(difficulty, btc_price_usd, efficiency_j_th, electricity_price):
    """
    计算开采一个区块的预期物理成本，以 BTC 计价
    """
    hashes_per_diff = 2 ** 32  # 难度1对应的哈希次数
    j_per_kwh = 3.6e6          # 1度电的焦耳数
    price_cooling = 1.15       # 15% 冷却开销

    energy_j = difficulty * hashes_per_diff * (efficiency_j_th * 1e-12)
    cost_usd = (energy_j / j_per_kwh) * electricity_price * price_cooling
    return cost_usd / btc_price_usd


def plot_figure_1(input_block, input_daily, input_price):
    print("正在绘图...")
    # 加载区块级数据
    df_block = pd.read_csv(input_block)
    df_block['timestamp'] = pd.to_datetime(df_block['timestamp'], utc=True)
    df_block['date'] = df_block['timestamp'].dt.normalize()

    # 加载价格数据
    df_price = pd.read_csv(input_price)
    df_price['Date'] = pd.to_datetime(df_price['Date'], utc=True)
    df_price['btc_price_usd'] = df_price['Price'].astype(str).str.replace(',', '').astype(float)

    # 加载日级数据
    df_daily = pd.read_csv(input_daily)
    df_daily['date'] = pd.to_datetime(df_daily['date'], utc=True)

    # 三表联查
    df_block = df_block.merge(df_price[['Date', 'btc_price_usd']], left_on='date', right_on='Date', how='left')
    df_block['btc_price_usd'] = df_block['btc_price_usd'].ffill().bfill()
    df_block = df_block.merge(df_daily[['date', 'whale_threshold']], on='date', how='inner')

    # 定义主流机型与电价
    miners = {
        'Bitmain Antminer S21 XP+ Hyd (Advanced, 11 J/TH)': {'e': 11, 'color': '#2ca02c'},
        'Bitmain Antminer S21+ Hyd (Sub-Adv, 15 J/TH)': {'e': 15, 'color': '#98df8a'},
        'Bitmain Antminer T19 Pro Hyd (Mainstream, 22 J/TH)': {'e': 22, 'color': '#1f77b4'},
        'Bitmain Antminer S19 Pro++ (Marginal, 26 J/TH)': {'e': 26, 'color': '#ff7f0e'},
        'Bitmain Antminer S19 Pro (Obsolete, 29.5 J/TH)': {'e': 29.5, 'color': '#d62728'}
    }

    prices = {
        'Cheap ($0.04/kWh)': {'p': 0.04, 'color': '#2ca02c', 'style': '-'},
        'Average ($0.05/kWh)': {'p': 0.05, 'color': '#1f77b4', 'style': '-'},
        'Expensive ($0.07/kWh)': {'p': 0.07, 'color': '#d62728', 'style': '-'}
    }

    # 计算日级平滑参数
    daily_df = df_block.groupby('date').agg({
        'difficulty': 'mean',
        'revenue': 'mean',
        'btc_price_usd': 'mean'
    }).reset_index()

    # 捕捉基于 30 天滑动窗口阈值的真正“局部鲸鱼”
    print("捕捉动态鲸鱼区块...")
    whales = df_block[df_block['fee'] > df_block['whale_threshold']].copy()

    # ================= 绘图部分 =================
    print("正在生成子图...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), dpi=300, sharex=True)

    # 定义减半日期
    halving_date = pd.Timestamp('2024-04-20', tz='UTC')

    # --- 子图 1: 交易费趋势与鲸鱼标注 ---
    ax1.plot(df_block['timestamp'], df_block['fee'], color='gray', linewidth=0.8, alpha=0.75, label='Transaction Fee',
             zorder=1)
    ax1.plot(df_daily['date'], df_daily['whale_threshold'], label='Dynamic Whale Threshold (P99.9)',
             color='purple', linestyle='--', linewidth=1.0, zorder=2)
    ax1.scatter(whales['timestamp'], whales['fee'], color='darkred', marker='*', s=60, alpha=0.8, edgecolors='black',
                linewidth=0.5, label='Whale Events', zorder=3)

    ax1.set_title('(a) Transaction Fees Trend and Whale Events', fontsize=13, pad=10)
    ax1.set_ylabel('Transaction Fee (BTC)', fontsize=11)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # 公共格式设置函数
    def format_omega_b_ax(omega_b_ax, title):
        omega_b_ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, zorder=2, label=r'Shutdown Threshold ($\omega_b = 1$)')
        omega_b_ax.set_title(title, fontsize=13, pad=10)
        omega_b_ax.set_ylabel(r'Profitability Factor ($\omega_b$)', fontsize=11)
        omega_b_ax.set_ylim(bottom=0.5)
        omega_b_ax.grid(True, linestyle=':', alpha=0.6)

    # --- 子图 2: 固定电价，5 种矿机演化 ---
    fixed_price = 0.05
    for name, params in miners.items():
        daily_c = calculate_cost_btc(daily_df['difficulty'], daily_df['btc_price_usd'], params['e'], fixed_price)
        daily_omega = daily_df['revenue'] / daily_c
        ax2.plot(daily_df['date'], daily_omega, label=name, color=params['color'], linewidth=1.5, alpha=0.8, zorder=3)

    format_omega_b_ax(ax2, f'(b) Impact of Hardware Efficiency (Fixed Electricity Price: ${fixed_price}/kWh)')
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=2)

    # --- 子图 3: 固定矿机，3 种电价演化 ---
    fixed_efficiency = 22
    for name, params in prices.items():
        daily_c = calculate_cost_btc(daily_df['difficulty'], daily_df['btc_price_usd'], fixed_efficiency, params['p'])
        daily_omega = daily_df['revenue'] / daily_c
        ax3.plot(daily_df['date'], daily_omega, label=name, color=params['color'], linestyle=params['style'],
                 linewidth=1.5, alpha=0.8, zorder=3)

    format_omega_b_ax(ax3, f'(c) Impact of Electricity Costs (Fixed Hardware: Bitmain Antminer T19 Pro Hyd, {fixed_efficiency} J/TH)')
    ax3.legend(loc='upper right', fontsize=10, framealpha=0.9)
    # 减半垂直线
    for ax in [ax1, ax2, ax3]:
        ax.axvline(x=halving_date, color='#8c564b', linestyle='-.', linewidth=1.5, zorder=10)
        ax.text(halving_date + pd.Timedelta(days=8), ax.get_ylim()[1] * 0.8, 'Halving',
                 color='#8c564b', fontsize=11, fontweight='bold', ha='left')
    # X轴时间格式化
    ax3.set_xlabel('Date from 2024-01 to 2025-12', fontsize=12, labelpad=10)
    ax3.xaxis.set_major_locator(m_dates.MonthLocator(interval=2))
    ax3.xaxis.set_major_formatter(m_dates.DateFormatter('%Y-%m'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha='center')

    plt.tight_layout()
    plt.savefig('../figures/Figure_1.pdf', format='pdf')
    print("图表已保存为 Figure_1.pdf")
    plt.show()


if __name__ == "__main__":
    input_block_filename = '../dataset/block_metrics.csv'
    input_daily_filename = '../dataset/daily_params.csv'
    input_price_filename = '../dataset/btc-usd.csv'
    plot_figure_1(input_block_filename, input_daily_filename, input_price_filename)