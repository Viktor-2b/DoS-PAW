import pandas as pd
import numpy as np

def parse_bits_to_difficulty(bits):
    """
    bits字段转换为难度数值。
    :param bits: 比特币链上 bits 字段
    :return: 难度数值
    """
    try:
        bits_int = int(bits, 16)
        exponent = bits_int >> 24
        coefficient = bits_int & 0xffffff
        if coefficient == 0: return np.nan
        target = coefficient * (2 ** (8 * (exponent - 3)))
        difficulty_1_target = 0xffff * (2 ** 208)
        if target == 0: return np.nan
        return difficulty_1_target / target
    except Exception as e:
        print(f"Error parsing bits {bits}: {e}")
        return np.nan

def generate_block_metrics(input_chain, output_file):
    """
    清洗区块级数据并输出。
    :param input_chain: 输入的比特币链上原始数据
    :param output_file: 输出文件名
    :return:返回 DataFrame 供下一阶段使用。
    """
    print(f"正在读取 {input_chain} ...")

    # 加载数据，强制将 bits 读为字符串，防止 pandas 误判为科学计数法
    df = pd.read_csv(input_chain, dtype={'bits': str})
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 排序，确保时间差计算正确
    df = df.sort_values('height').reset_index(drop=True)

    # 计算出块时间间隔，填充NaN，并处理负数时间漂移
    df['delta_t'] = df['timestamp'].diff().dt.total_seconds().fillna(600)
    df['delta_t'] = df['delta_t'].apply(lambda x: max(x, 1.0))

    # 字段处理
    df['difficulty'] = df['bits'].apply(parse_bits_to_difficulty)
    df['fee'] = df['total_fee_satoshi'] / 1e8
    df['revenue'] = df['miner_revenue_satoshi'] / 1e8
    df['block_reward'] = df['revenue']-df['fee']

    # 保存结果
    cols = ['height', 'timestamp', 'delta_t', 'difficulty', 'block_reward', 'fee', 'revenue']
    df[cols].to_csv(output_file, index=False)
    print(f"已保存区块级数据至 {output_file}")
    return df

def generate_daily_metrics(df, output_file, window_days=30):
    """
    基于清洗后的区块级数据，使用中心化滑动窗口计算日级宏观参数。
    :param output_file: 输出文件名
    :param df:清洗后的区块级数据
    :param window_days:窗口大小
    :return:
    """
    print(f"正在进行 {window_days} 天中心化滑动窗口参数估算...")
    # 将时间戳设为索引并排序，极大提升滑动窗口切片的速度
    df_indexed = df.set_index('timestamp').sort_index()

    # 实验的正式评估期
    start_date = pd.Timestamp('2024-01-01', tz='UTC')
    end_date = pd.Timestamp('2025-12-31', tz='UTC')
    daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # 中心化窗口：前后各取 window_days 的一半（即前后各 15 天）
    half_window = pd.Timedelta(days=window_days / 2)
    results = []
    for current_date in daily_dates:
        # 获取滑动窗口内的数据切片
        window_start = current_date - half_window
        window_end = current_date + half_window
        df_window = df_indexed[(df_indexed.index >= window_start) & (df_indexed.index < window_end)]
        # 计算时间与基础参数
        total_time = df_window['delta_t'].sum()
        total_fee = df_window['fee'].sum()
        # 交易费用累积速率
        lambda_t = total_fee / total_time if total_time > 0 else 0
        # 鲸鱼阈值，鲸鱼交易，鲸鱼到达频率与超额手续费期望
        whale_threshold = df_window['fee'].quantile(0.999)
        whales = df_window[df_window['fee'] > whale_threshold]
        lambda_w = len(whales) / total_time if total_time > 0 else 0
        e_rw = (whales['fee'] - whale_threshold).mean() if len(whales) > 0 else 0

        results.append({
            'date': current_date.date(),
            'lambda_t': lambda_t,
            'lambda_w': lambda_w,
            'E_Rw': e_rw,
            'mean_K': df_window['block_reward'].mean(),
            'whale_threshold': whale_threshold
        })
    df_daily = pd.DataFrame(results)
    df_daily.to_csv(output_file, index=False)
    print(f"已保存日级宏观参数至: {output_file}")
    return df_daily

if __name__ == "__main__":
    input_chain_filename = 'dataset/btc_chain.csv'
    input_price_filename = 'dataset/btc-usd.csv'
    output_block_filename = 'dataset/block_metrics.csv'
    output_daily_filename = 'dataset/daily_params.csv'
    df_block_metrics = generate_block_metrics(input_chain_filename, output_block_filename)
    generate_daily_metrics(df_block_metrics, output_daily_filename)
    print("数据预处理全部完成，")