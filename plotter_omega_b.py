import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates

# 设置学术风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12


def plot_figure1(input_file, output_file):
    print("正在绘图...")
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # 识别鲸鱼
    df['whale_threshold_dynamic'] = df['omega_b'].rolling(window=144 * 30, min_periods=144).quantile(0.999)
    whales = df[df['omega_b'] > df['whale_threshold_dynamic']]

    # 计算移动平均
    df['omega_b_ma'] = df['omega_b'].rolling(window=144 * 7).mean()

    # 创建画布
    fig, ax = plt.subplots(figsize=(15, 7))  # 稍微加宽画布

    # 1. 原始数据 (半透明)
    ax.plot(df['timestamp'], df['omega_b'],
            color='silver', linewidth=0.4, alpha=0.5, label='Raw Profitability', zorder=1)

    # 2. 移动平均线
    ax.plot(df['timestamp'], df['omega_b_ma'],
            color='#1f77b4', linewidth=2.0, label='7-Day Moving Average', zorder=2)

    # 3. 鲸鱼事件
    # 只显示 Y 轴范围内可见的鲸鱼
    visible_whales = whales[whales['omega_b'] < 6.0]
    ax.scatter(visible_whales['timestamp'], visible_whales['omega_b'],
               color='#d62728', marker='x', s=40, label='Whale Events (P99.9)', zorder=3)

    # 4. 盈亏平衡线
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Break-even Level (Cost)', zorder=2)

    # 5. 减半事件垂直线
    halving_date = pd.Timestamp('2024-04-20', tz='UTC')
    ax.axvline(x=halving_date, color='purple', linestyle='--', linewidth=1.5,
               label='Bitcoin Halving Event')

    # 在减半线旁边增加文字标注
    ax.text(halving_date + pd.Timedelta(days=10), 5.5, 'Halving & Runes Launch',
            color='purple', rotation=0, fontsize=10)

    # 格式化
    ax.set_ylabel(r'Profitability Factor $\omega_b$', fontsize=14)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_title('Figure 1: Bitcoin Mining Profitability Factor ($\omega_b$) from 2024 to 2025', fontsize=16)

    # X 轴日期格式化
    ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(dates.MonthLocator(interval=2))

    # 图例
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=10)

    # 网格
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()

    # 保存为 PDF
    plt.savefig(output_file, dpi=300)
    print(f"图表已保存为 {output_file}")
    plt.show()


if __name__ == "__main__":
    plot_figure1('btc.csv','figure1_profitability.pdf')