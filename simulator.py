import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


class Simulator:
    def __init__(self, alpha, beta, eta, gamma, r1, target_strategy='Mine'):
        # 算力分布
        self.alpha = alpha  # 攻击者算力
        self.beta = beta  # 受害者算力
        self.eta = eta  # 目标算力
        self.delta = 1.0 - alpha - beta - eta  # 其他算力
        if self.delta < 0.0 or self.delta > 1.0:
            raise ValueError(f"算力总和异常！当前 delta 为 {self.delta:.4f}，请检查 alpha, beta, eta 的总和是否超过 1.0")
        self.r1 = r1 # 初始渗透算力比例
        self.r2 = 0.0 # 调整算力比例
        # 网络参数
        self.gamma = gamma  # 传播优势
        self.target_strategy = target_strategy  # 目标策略，'Mine', 'Stop', 'SPV'
        self.LAMBDA = 1.0  # 基准出块率

    def run(self, omega_b, num_events=100000):
        # 系统当前状态与累计时间
        state = 0
        time_elapsed = 0.0
        # 区块产出统计
        blocks_atk = 0
        blocks_vic = 0
        blocks_tar = 0
        blocks_oth = 0
        # 份额统计
        shares_atk = 0.0
        shares_vic = 0.0
        # 开机时间统计
        time_atk = 0.0
        time_tar = 0.0
        # 记录外部竞争状态下的新块归属临时变量
        ext_owner = None
        for _ in range(num_events):  # 每个循环对应一个出块轮次
            # 除状态1、2外的全网有效算力始终为1
            active_hashrate = self.alpha + self.beta + self.eta + self.delta
            if state == 0:  # 初始状态
                # 采样物理时间
                dt = random.expovariate(self.LAMBDA * active_hashrate)
                time_elapsed += dt
                # 累计份额与时间
                shares_atk += self.r1 * self.alpha * dt
                shares_vic += self.beta * dt
                time_atk += self.alpha * dt
                time_tar += self.eta * dt
                # 随机事件模逆
                rand = random.random() * active_hashrate
                if rand < self.delta:  # 其他矿工挖到
                    blocks_oth += 1
                elif rand < self.delta + self.eta:  # 目标矿工挖到
                    blocks_tar += 1
                elif rand < self.delta + self.eta + self.beta:  # 受害者矿池挖到
                    blocks_vic += 1
                elif rand < self.delta + self.eta + self.beta + self.alpha * (1 - self.r1):  # 攻击者私有算力挖到
                    state = 1
                else:  # 攻击者渗透算力挖到
                    state = 2

            elif state == 1:  # 私有领先状态，攻击者私有算力关机，调整渗透算力比例为 1 且只挖PPoW
                self.r2=1.0
                eta_act = 0.0 if self.target_strategy == 'Stop' else self.eta  # 目标矿工的活跃状态
                active_hashrate = self.beta + eta_act + self.delta
                # 采样物理时间
                dt = random.expovariate(self.LAMBDA * active_hashrate)
                time_elapsed += dt
                # 累计份额与时间
                shares_atk += self.r2 * self.alpha * dt
                shares_vic += self.beta * dt
                time_atk += self.alpha * dt
                time_tar += eta_act * dt
                # 随机事件模逆
                rand = random.random() * active_hashrate
                if rand < self.beta:  # 受害者矿池挖到
                    state = 3
                elif rand < self.beta + eta_act:
                    if self.target_strategy == 'SPV':  # 目标踩中SPV陷阱
                        state = 0
                    else:  # 目标矿工挖到
                        ext_owner = 'eta'
                        state = 4
                else:  # 其他矿工挖到
                    ext_owner = 'delta'
                    state = 4

            elif state == 2:  # 渗透领先状态，攻击者私有算力关机，调整渗透算力比例为 r 且只挖PPoW
                self.r2=1.0
                eta_act = 0.0 if self.target_strategy == 'Stop' else self.eta
                active_hashrate = self.beta + eta_act + self.delta
                # 采样物理时间
                dt = random.expovariate(self.LAMBDA * active_hashrate)
                time_elapsed += dt
                # 累计份额与时间
                shares_atk += self.r2 * self.alpha * dt
                shares_vic += self.beta * dt
                time_atk += self.alpha * dt
                time_tar += eta_act * dt
                # 随机事件模逆
                rand = random.random() * active_hashrate
                if rand < self.beta:  # 受害者矿池挖到
                    blocks_vic += 1
                    state = 0
                elif rand < self.beta + eta_act:
                    if self.target_strategy == 'SPV':  # 目标踩中SPV陷阱
                        state = 0
                    else:  # 目标矿工挖到
                        ext_owner = 'eta'
                        state = 5
                else:  # 其他矿工挖到
                    ext_owner = 'delta'
                    state = 5

            elif state == 3:  # PvV
                self.r2=0.0
                # 采样物理时间
                dt = random.expovariate(self.LAMBDA * active_hashrate)
                time_elapsed += dt
                # 累计份额与时间
                shares_atk += self.r2 * self.alpha * dt
                shares_vic += self.beta * dt
                time_atk += self.alpha * dt
                time_tar += self.eta * dt
                # 随机事件模逆
                rand = random.random() * active_hashrate
                if rand<self.alpha:# 攻击者私有算力挖到
                    blocks_atk += 2
                elif rand<self.alpha+self.beta:# 受害者矿池挖到
                    blocks_vic += 2
                elif rand<self.alpha+self.beta+self.eta:# 目标矿工挖到
                    blocks_tar += 1
                    if random.random()<self.gamma:# 支持私有分支
                        blocks_atk += 1
                    else:
                        blocks_vic += 1
                else:# 其他矿工挖到
                    blocks_oth += 1
                    if random.random()<self.gamma:# 支持私有分支
                        blocks_atk += 1
                    else:
                        blocks_vic += 1
                state=0

            elif state == 4:  # PvE
                self.r2 = 0.0
                # 采样物理时间
                dt = random.expovariate(self.LAMBDA * active_hashrate)
                time_elapsed += dt
                # 累计份额与时间
                shares_atk += self.r2 * self.alpha * dt
                shares_vic += self.beta * dt
                time_atk += self.alpha * dt
                time_tar += self.eta * dt
                # 随机事件模逆
                rand = random.random() * active_hashrate
                if rand < self.alpha:  # 攻击者私有算力挖到
                    blocks_atk += 2
                elif rand < self.alpha + self.beta:  # 受害者矿池挖到
                    blocks_vic += 1
                    if random.random() < self.gamma:  # 支持私有分支
                        blocks_atk += 1
                    else:
                        if ext_owner == 'eta':
                            blocks_tar += 1
                        else:
                            blocks_oth += 1
                elif rand < self.alpha + self.beta + self.eta:  # 目标矿工挖到
                    blocks_tar += 1
                    if random.random() < self.gamma:  # 支持私有分支
                        blocks_atk += 1
                    else:
                        if ext_owner == 'eta':
                            blocks_tar += 1
                        else:
                            blocks_oth += 1
                else:  # 其他矿工挖到
                    blocks_oth += 1
                    if random.random() < self.gamma:  # 支持私有分支
                        blocks_atk += 1
                    else:
                        if ext_owner == 'eta':
                            blocks_tar += 1
                        else:
                            blocks_oth += 1
                state = 0

            elif state == 5:  # IvE
                self.r2 = 1.0
                # 采样物理时间
                dt = random.expovariate(self.LAMBDA * active_hashrate)
                time_elapsed += dt
                # 累计份额与时间
                shares_atk += self.r2 * self.alpha * dt
                shares_vic += self.beta * dt
                time_atk += self.alpha * dt
                time_tar += self.eta * dt
                # 随机事件模逆
                rand = random.random() * active_hashrate
                if rand < self.alpha + self.beta:  # 攻击者渗透算力或受害者矿池挖到
                    blocks_vic += 2
                elif rand < self.alpha + self.beta + self.eta:  # 目标矿工挖到
                    blocks_tar += 1
                    if random.random() < self.gamma:  # 支持渗透分支
                        blocks_vic += 1
                    else:
                        if ext_owner == 'eta':
                            blocks_tar += 1
                        else:
                            blocks_oth += 1
                else:  # 其他矿工挖到
                    blocks_oth += 1
                    if random.random() < self.gamma:  # 支持渗透分支
                        blocks_vic += 1
                    else:
                        if ext_owner == 'eta':
                            blocks_tar += 1
                        else:
                            blocks_oth += 1
                state = 0

        # 最终收益结算
        share_ratio = shares_atk / (shares_atk + shares_vic) if (shares_atk + shares_vic) > 0 else 0
        pool_payout_to_atk = blocks_vic * share_ratio

        revenue_atk = blocks_atk + pool_payout_to_atk
        revenue_tar = blocks_tar

        # 归一化效用 (U = omega_b * R_rate - Cost_rate)
        # 假设单算力成本为 1，则 U_i = (omega_b * revenue - cost_time) / (hashrate * time_elapsed)
        rate_atk = revenue_atk / time_elapsed
        rate_tar = revenue_tar / time_elapsed

        u_atk = (omega_b * rate_atk - (time_atk / time_elapsed)) / self.alpha
        u_tar = (omega_b * rate_tar - (time_tar / time_elapsed)) / self.eta

        return {
            'alpha': self.alpha, 'omega_b': omega_b, 'gamma': self.gamma,
            'Target_Strategy': self.target_strategy,
            'Optimal_r1': self.r1,
            'Attacker_Utility': u_atk, 'Target_Utility': u_tar
        }


def find_optimal_r(alpha, beta, eta, gamma, omega_b, strategy):
    """
    通过离散网格搜索，寻找在当前环境参数下，能使攻击者效用最大化的最优 r*
    """
    best_r = 0.0
    max_utility = -np.inf

    # 构建 r 的搜索网格 (从 0.0 到 1.0，步长 0.05 即可平衡精度与速度)
    r_candidates = np.linspace(0.0, 1.0, 21)

    for r_test in r_candidates:
        # 实例化模拟器进行试探性运行 (num_events 设小一点以加速搜索，比如 5000)
        sim = Simulator(
            alpha=alpha,
            beta=beta,
            eta=eta,
            gamma=gamma,
            r1=r_test,
            target_strategy=strategy
        )
        result = sim.run(omega_b=omega_b, num_events=10000)
        u_atk_test = result['Attacker_Utility']

        # 更新最优解
        if u_atk_test > max_utility:
            max_utility = u_atk_test
            best_r = r_test

    return best_r


def run_experiment(params):
    alpha, omega_b, gamma, strategy = params
    best_r1 = find_optimal_r(alpha, 0.20, 0.10, gamma, omega_b, strategy)
    sim = Simulator(
        alpha=alpha, beta=0.20, eta=0.10, gamma=gamma,
        r1=best_r1, target_strategy=strategy
    )
    return sim.run(omega_b=omega_b, num_events=50000)



if __name__ == '__main__':
    print("开始生成规模化仿真数据 CSV...")

    # 构建多维度参数网格
    alphas = np.linspace(0.05, 0.45, 15)
    omega_bs = np.linspace(1.0, 2.5, 15)
    gammas = np.linspace(0.0, 1.0, 11)
    strategies = ['Mine', 'Stop', 'SPV']

    tasks = []
    for a in alphas:
        for w in omega_bs:
            for g in gammas:
                for s in strategies:
                    tasks.append((a, w, g, s))

    num_cores = max(1, cpu_count() - 1)
    with Pool(num_cores) as p:
        results = list(tqdm(p.imap_unordered(run_experiment, tasks), total=len(tasks)))

    # 保存为 CSV
    df = pd.DataFrame(results)
    df.to_csv('simulation_results.csv', index=False)
    print("数据生成完毕！已保存至 simulation_results.csv")
