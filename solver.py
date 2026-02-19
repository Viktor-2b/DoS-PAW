import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space


class SPV_PAW_Solver:
    def __init__(self, alpha, beta, eta, gamma=0.5, r1=0.0, r2=1.0):
        """
        初始化参数
        alpha: 攻击者总算力
        beta: 受害者算力
        eta: 目标矿工算力 (Target/Rational)
        delta: 其他诚实矿工算力 (自动计算)
        gamma: 抢跑能力
        r1: 初始渗透率
        r2: 扣块后渗透率
        """
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.delta = 1 - alpha - beta - eta
        self.gamma = gamma
        self.r1 = r1
        self.r2 = r2
        self.Lambda = 1.0  # 归一化出块速率

    def get_steady_state(self, strategy):
        """
        根据策略构建 Q 矩阵并求解稳态分布 pi
        strategy: 'Mine', 'Stop', 'SPV'
        """
        # 基础算力拆分
        alpha_priv = (1 - self.r2) * self.alpha
        # 注意：State 0/1 使用 r1，State 2-5 使用 r2 (简化模型假设进入攻击态即调整)

        # 动态调整参与算力
        eta_val = 0 if strategy == 'Stop' else self.eta

        # 诚实算力集合 (H) 和 外部算力集合 (E)
        H_power = self.beta + eta_val + self.delta
        E_power = eta_val + self.delta

        # 初始化 6x6 矩阵
        Q = np.zeros((6, 6))

        # --- 1. 构建转移速率 (Rate) ---

        # State 0: Initial
        # 0 -> 1: Private Lead
        Q[0, 1] = self.Lambda * (1 - self.r1) * self.alpha
        # 0 -> 2: Infiltration Lead
        Q[0, 2] = self.Lambda * self.r1 * self.alpha

        # State 1: Private Lead
        # 1 -> 3: Race PvV (Victim finds)
        Q[1, 3] = self.Lambda * self.beta
        # 1 -> 4: Race PvE (External finds)
        Q[1, 4] = self.Lambda * E_power

        if strategy == 'SPV':
            # [Trap] 1 -> 0: Target SPV finds
            Q[1, 0] += self.Lambda * self.eta
            # 修正 1->4: E_power 中去掉了 eta，上面 Q[1,4] 计算时 eta_val 已经是 eta，
            # 但如果是 SPV，E_power 里的 eta 不会触发 Race，而是触发 Trap。
            # 所以这里需要覆盖上面的 Q[1,4]
            Q[1, 4] = self.Lambda * self.delta

            # State 2: Infiltration Lead
        # 2 -> 0: Victim finds (Reset)
        Q[2, 0] += self.Lambda * self.beta
        # 2 -> 5: Race IvE (External finds)
        Q[2, 5] = self.Lambda * E_power

        if strategy == 'SPV':
            # [Trap] 2 -> 0: Target SPV finds (Double Trap)
            Q[2, 0] += self.Lambda * self.eta
            # 修正 2->5
            Q[2, 5] = self.Lambda * self.delta

        # State 3: Race PvV
        # 3 -> 0: Resolution
        # 速率 = 全网有效算力 (Stop模式下分母变小)
        total_rate_3 = alpha_priv + self.beta + E_power
        Q[3, 0] = self.Lambda * total_rate_3

        # State 4: Race PvE
        total_rate_4 = alpha_priv + self.beta + E_power
        Q[4, 0] = self.Lambda * total_rate_4

        # State 5: Race IvE
        # 渗透算力此时全开
        total_rate_5 = alpha_priv + (self.r2 * self.alpha) + self.beta + E_power
        Q[5, 0] = self.Lambda * total_rate_5

        # --- 2. 填充对角线 ---
        for i in range(6):
            Q[i, i] = -np.sum(Q[i, :])

        # --- 3. 求解 pi * Q = 0 ---
        try:
            # 使用 scipy 求零空间
            ns = null_space(Q.T)
            pi = ns[:, 0]
            pi = pi / np.sum(pi)  # 归一化
            return np.abs(pi)  # 避免 -0.0
        except:
            return np.zeros(6)

    def calculate_rewards(self, strategy):
        pi = self.get_steady_state(strategy)

        # 算力定义
        alpha_priv = (1 - self.r2) * self.alpha
        alpha_inf = self.r2 * self.alpha

        # 动态调整
        eta_val = 0 if strategy == 'Stop' else self.eta
        E_power = eta_val + self.delta
        # SPV 模式下，Race 阶段 eta 恢复正常 (根据之前的设定)
        if strategy == 'SPV':
            E_power_race = self.eta + self.delta
        else:
            E_power_race = E_power

        # --- 收益计算 (Revenue Rate) ---
        # 我们只计算 受害者(Victim) 和 攻击者(Attacker) 的收益
        # R = pi_state * transition_rate * prob_win * reward

        R_vic = 0
        R_atk = 0

        # 1. State 0 正常出块
        # Victim 挖到: pi[0] * lambda * beta
        R_vic += pi[0] * self.Lambda * self.beta
        # Attacker 无收益 (在 State 0 纯投入渗透)

        # 2. State 2 -> 0 (Victim finds / Trap)
        # Victim finds: pi[2] * lambda * beta
        R_vic += pi[2] * self.Lambda * self.beta

        # 3. State 3 (Race PvV)
        # Victim Wins: pi[3] * lambda * [beta + (1-gamma)E]
        # Victim 赢了拿 2 块
        # Attacker 拿 Share (假设 Share 比例 s)
        s = alpha_inf / (self.beta + alpha_inf)

        # 胜率计算 (Race 阶段假设 eta 恢复)
        denom = alpha_priv + self.beta + E_power_race
        prob_vic_win = (self.beta + (1 - self.gamma) * E_power_race) / denom
        prob_atk_win = (alpha_priv + self.gamma * E_power_race) / denom

        rate_3 = self.Lambda * denom

        # 结算
        R_vic += pi[3] * rate_3 * prob_vic_win * 2 * (1 - s)  # 扣除给攻击者的Share
        R_atk += pi[3] * rate_3 * prob_vic_win * 2 * s  # 攻击者的Share
        R_atk += pi[3] * rate_3 * prob_atk_win * 2  # 攻击者私有赢

        # 4. State 4 (Race PvE)
        # ... (类似逻辑，为了简化演示，这里暂略，完整代码需补全)

        # 5. State 5 (Race IvE)
        # Infiltration Wins: pi[5] * lambda * [alpha + beta + gamma*E]
        denom_5 = alpha_priv + alpha_inf + self.beta + E_power_race
        prob_inf_win = (alpha_priv + alpha_inf + self.beta + self.gamma * E_power_race) / denom_5

        rate_5 = self.Lambda * denom_5

        # 结算: 受害者名义上赢了，但要分红
        R_vic += pi[5] * rate_5 * prob_inf_win * 2 * (1 - s)
        R_atk += pi[5] * rate_5 * prob_inf_win * 2 * s

        return R_vic, R_atk


# --- 绘图：展示 DoS 效果 ---
# 场景：固定攻击者算力，观察随着目标矿工(eta)比例增加，受害者的收益变化
alphas = 0.2
betas = 0.2
gammas = 0.5
eta_range = np.linspace(0, 0.4, 20)

vic_rev_mine = []
vic_rev_spv = []

for e in eta_range:
    solver = SPV_PAW_Solver(alphas, betas, e, gammas, r1=0.0, r2=1.0)  # 假设攻击者全渗透

    # 1. 正常情况 (Mine)
    rv_m, _ = solver.calculate_rewards('Mine')
    vic_rev_mine.append(rv_m)

    # 2. 攻击情况 (SPV)
    rv_s, _ = solver.calculate_rewards('SPV')
    vic_rev_spv.append(rv_s)

# 画图
plt.figure(figsize=(10, 6))
plt.plot(eta_range, vic_rev_mine, 'g--', label='Target chooses Mine (Normal)')
plt.plot(eta_range, vic_rev_spv, 'r-', label='Target chooses SPV (Under Attack)')
plt.title(f'Impact of SPV-PAW Attack on Victim Pool (alpha={alphas}, beta={betas})')
plt.xlabel('Target Miner Hashrate ($\eta$)')
plt.ylabel('Victim Pool Revenue')
plt.legend()
plt.grid(True)
plt.show()