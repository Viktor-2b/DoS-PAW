import pandas as pd


def print_tables(input_blocks, input_daily):
    # 读取区块数据
    df_blocks = pd.read_csv(input_blocks)
    df_blocks['timestamp'] = pd.to_datetime(df_blocks['timestamp'])

    # 选定时间段
    start_date = pd.Timestamp('2024-01-01', tz='UTC')
    end_date = pd.Timestamp('2025-12-31', tz='UTC')
    df_blocks = df_blocks[(df_blocks['timestamp'] >= start_date) & (df_blocks['timestamp'] <= end_date)].copy()

    # 读取日级宏观数据
    df_daily = pd.read_csv(input_daily)

    min_time_str = df_blocks['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
    max_time_str = df_blocks['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
    # 打印表1
    print("\n" + "=" * 50)
    print("Table 1: Dataset Statistics")
    print("=" * 50)
    print(f"时间范围: {min_time_str} to {max_time_str}")
    print(f"区块总数: {len(df_blocks):,}")
    print(f"出块间隔平均数: {df_blocks['delta_t'].mean():.2f} s")
    print(f"出块间隔中位数: {df_blocks['delta_t'].median():.2f} s")
    print(f"出块间隔分位数 P95: {df_blocks['delta_t'].quantile(0.95):.2f} s")
    print(f"平均手续费: {df_blocks['fee'].mean():.4f} BTC")
    print(f"极端手续分位数 P99.9: {df_blocks['fee'].quantile(0.999):.4f} BTC")
    print("=" * 50 + "\n")
    # 打印表2
    avg_lambda_t = df_daily['lambda_t'].mean()
    avg_lambda_w = df_daily['lambda_w'].mean()
    avg_e_rw = df_daily['E_Rw'].mean()

    print("\n" + "=" * 50)
    print("Table 2: Parameter Estimation")
    print("=" * 50)
    print(f"交易费用累计速率均值 (lambda_t): {avg_lambda_t:.8e} BTC/s")
    print(f"鲸鱼到达速率均值 (lambda_w): {avg_lambda_w:.8e} 次/s")
    print(f"鲸鱼超额奖励期望均值 (e_rw): {avg_e_rw:.4f} BTC")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    input_blocks_filename = '../dataset/block_metrics.csv'
    input_daily_filename = '../dataset/daily_params.csv'
    print_tables(input_blocks_filename, input_daily_filename)