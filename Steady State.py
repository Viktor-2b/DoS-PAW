import numpy as np
from scipy.linalg import null_space


def get_steady_state_scenario_1(alpha, beta, eta, delta, r1, r2):
    """
    Scenario 1: Target chooses Mine
    """
    _lambda = 1.0
    h = eta + delta  # External miners in Scenario 1

    # 初始化 6x6 矩阵
    q = np.zeros((6, 6))

    # Row 0: Initial
    q[0, 1] = _lambda * (1 - r1) * alpha
    q[0, 2] = _lambda * r1 * alpha
    q[0, 0] = -np.sum(q[0, :])  # Diagonal

    # Row 1: Private Lead
    q[1, 3] = _lambda * beta
    q[1, 4] = _lambda * h
    q[1, 1] = -np.sum(q[1, :])

    # Row 2: Infiltration Lead
    q[2, 0] = _lambda * beta  # Victim finds -> Reset
    q[2, 5] = _lambda * h  # Others find -> IvE
    q[2, 2] = -np.sum(q[2, :])

    # Row 3: Race PvV (Attacker stops, rate reduced)
    q[3, 0] = _lambda * (1 - r2 * alpha)
    q[3, 3] = -q[3, 0]

    # Row 4: Race PvE (Attacker stops, rate reduced)
    q[4, 0] = _lambda * (1 - r2 * alpha)
    q[4, 4] = -q[4, 0]

    # Row 5: Race IvE (Full power, including infiltration)
    q[5, 0] = _lambda
    q[5, 5] = -q[5, 0]

    # 求解 pi * Q = 0
    # 也就是 Q.T 的零空间
    ns = null_space(q.T)
    pi = ns[:, 0]
    pi = pi / np.sum(pi)  # 归一化

    return pi


