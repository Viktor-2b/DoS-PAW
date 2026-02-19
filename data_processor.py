import pandas as pd
import numpy as np

def parse_bits_to_difficulty(bits):
    """
    bits字段转换为难度数值。
    :param bits: 比特币链上bits字段
    :return: 难度数值
    """
    bits_int = int(bits, 16)
    exponent = bits_int >> 24
    coefficient = bits_int & 0xffffff
    if coefficient == 0: return np.nan
    target = coefficient * (2 ** (8 * (exponent - 3)))
    difficulty_1_target = 0xffff * (2 ** 208)

    if target == 0: return np.nan
    return difficulty_1_target / target

def get_dynamic_efficiency(ts):
    """
    根据时间戳返回全网加权平均能效 (J/TH)
    :param ts: 时间戳
    :return: 能效
    """
    if ts < pd.Timestamp('2024-04-21', tz='UTC'): # 减半前
        return 28.0
    elif ts < pd.Timestamp('2025-01-01', tz='UTC'): # 减半后 shakeout 期
        return 22.0
    elif ts < pd.Timestamp('2025-07-01', tz='UTC'): # 25年上半年
        return 18.0
    else: # 25年下半年
        return 13.0

def calculate_cost(df, input_price_dataset):
    """
    计算成本
    :param df:
    :param input_price_dataset:
    :return:
    """
    # 读取价格
    pdf = pd.read_csv(input_price_dataset)
    # 处理 Price 列的逗号
    pdf['btc_price_usd'] = pdf['Price'].str.replace(',', '').astype(float)
    pdf['date'] = pd.to_datetime(pdf['Date']).dt.date

    # 合并价格到主表
    df['date'] = df['timestamp'].dt.date
    df = df.merge(pdf[['date', 'btc_price_usd']], on='date', how='left')

    # 物理常数与环境参数
    hashes_per_diff = 2 ** 32  # 难度1对应的哈希次数
    j_per_kwh = 3.6e6  # 1度电等于多少焦耳
    price_electricity = 0.05  # 电价 $/kWh
    price_cooling = 1.15  # 15% 冷却开销

    # 动态应用能效模型
    df['efficiency'] = df['timestamp'].apply(get_dynamic_efficiency)

    df['cost_usd'] = (df['difficulty'] * hashes_per_diff) * \
                     (df['efficiency'] * 1e-12) * \
                     (price_electricity / j_per_kwh) * \
                     price_cooling
    # 转换为 BTC 成本
    df['c'] = df['cost_usd'] / df['btc_price_usd']
    return df



def process_data(input_chain_dataset, input_price_dataset, output_file):
    print(f"正在读取 {input_chain_dataset} 和 {input_price_dataset}...")

    # 加载数据
    df = pd.read_csv(input_chain_dataset)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 选定时间段
    start_date = pd.Timestamp('2024-01-01', tz='UTC')
    end_date = pd.Timestamp('2026-01-01', tz='UTC')
    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] < end_date)].copy()
    df = df.sort_values('height').reset_index(drop=True)

    # 字段处理
    df['difficulty'] = df['bits'].apply(parse_bits_to_difficulty)
    df['fee'] = df['total_fee_satoshi'] / 1e8
    df['revenue'] = df['miner_revenue_satoshi'] / 1e8
    df['block_reward'] = df['revenue']-df['fee']

    # 成本计算
    df = calculate_cost(df, input_price_dataset)

    # 计算 omega_b
    df['omega_b'] = df['revenue'] / df['c']

    # 估算 lambda_t
    lambda_t = df['fee'].sum() / df['delta_t'].sum()

    # 5. 识别鲸鱼
    whale_threshold = df['omega_b'].quantile(0.999)
    whales_count = len(df[df['omega_b'] > whale_threshold])

    # 输出统计信息
    print("\n" + "=" * 40)
    print("      Table 1: Dataset Statistics")
    print("=" * 40)
    print(f"Time Range:         {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total Block Count:  {len(df)}")
    print(f"Avg Block Interval: {df['delta_t'].mean():.2f} s")
    print(f"Median Interval:    {df['delta_t'].median():.2f} s")
    print(f"95% Interval:       {df['delta_t'].quantile(0.95):.2f} s")
    print(f"Avg Fee per Block:  {df['fee'].mean():.4f} BTC")
    print(f"Extreme Fee (P99.9):{df['fee'].quantile(0.999):.4f} BTC")
    print(f"Avg Profitability (Omega_b): {df['omega_b'].mean():.4f}")
    print(f"Whale Events (n):   {whales_count}")
    print(f"Fee Accumulation Rate (lambda_t): {lambda_t:.8f} BTC/s")
    print("=" * 40 + "\n")

    # 保存结果
    cols = ['height', 'timestamp', 'delta_t', 'difficulty', 'btc_price_usd',
            'block_reward', 'fee', 'c', 'omega_b', 'efficiency']
    df[cols].to_csv(output_file, index=False)
    print(f"已保存清洗数据至 {output_file}")


if __name__ == "__main__":
    process_data('btc_chain.csv','btc-usd.csv', 'btc.csv')