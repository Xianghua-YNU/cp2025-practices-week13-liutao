#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
薛定谔方程 - 方势阱能级计算

本模块实现了一维方势阱中粒子能级的计算方法。
"""

import numpy as np
import matplotlib.pyplot as plt

# 物理常数
HBAR = 1.0545718e-34  # 约化普朗克常数 (J·s)
ELECTRON_MASS = 9.1094e-31  # 电子质量 (kg)
EV_TO_JOULE = 1.6021766208e-19  # 电子伏转换为焦耳的系数


def calculate_y_values(E_values, V, w, m):
    """
    计算方势阱能级方程中的三个函数值
    """
    # TODO: 实现计算y1, y2, y3的代码 (约10行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 注意单位转换和避免数值计算中的溢出或下溢
    E_J = E_values * EV_TO_JOULE  # 将能量转换为焦耳
    k = np.sqrt((w**2 * m * E_J) / (2 * HBAR**2))  # 计算k值
    y1 = np.tan(k)  # y1 = tan(k)
    y2 = np.sqrt((V - E_values) / E_values)  # 偶宇称函数
    y3 = -np.sqrt(E_values / (V - E_values))  # 奇宇称函数
    return y1, y2, y3


def plot_energy_functions(E_values, y1, y2, y3):
    """
    绘制能级方程的三个函数曲线
    """
    # TODO: 实现绘制三个函数曲线的代码 (约15行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 使用不同颜色和线型，添加适当的标签、图例和标题
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(E_values, y1, label='$y_1 = \\tan(k)$', color='blue', linestyle='-')
    ax.plot(E_values, y2, label='$y_2 = \\sqrt{(V-E)/E}$', color='green', linestyle='--')
    ax.plot(E_values, y3, label='$y_3 = -\\sqrt{E/(V-E)}$', color='red', linestyle=':')
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Function Value')
    ax.set_title('Schrödinger Equation Transcendental Functions')
    ax.legend()
    ax.grid(True)
    ax.set_ylim(-20, 20)  # 限制y轴范围以清晰显示交点
    return fig
    

def find_energy_level_bisection(n, V, w, m, precision=0.001, E_min=0.001, E_max=None):
    """
    使用二分法求解方势阱中的第n个能级
    """
    # TODO: 实现二分法求解能级的代码 (约25行代码)
    # [STUDENT_CODE_HERE]
    # 提示: 需要考虑能级的奇偶性，偶数能级使用偶宇称方程，奇数能级使用奇宇称方程
    if E_max is None:
        E_max = V - precision  # 防止除零错误
    parity = 'even' if n % 2 == 0 else 'odd'
    m_level = (n // 2) + 1  # 确定解的序数

    def f(E):
        E_J = E * EV_TO_JOULE
        k = np.sqrt((w**2 * m * E_J) / (2 * HBAR**2))
        if parity == 'even':
            right = np.sqrt((V - E) / E)
        else:
            right = -np.sqrt(E / (V - E))
        return np.tan(k) - right

    # 扫描能量区间寻找符号变化
    step = 0.001
    E_scan = np.arange(E_min, E_max, step)
    intervals = []
    if len(E_scan) == 0:
        raise ValueError("E_scan is empty. Check parameters.")
    prev_f = f(E_scan[0])
    for i in range(1, len(E_scan)):
        current_f = f(E_scan[i])
        if np.sign(current_f) != np.sign(prev_f):
            intervals.append((E_scan[i-1], E_scan[i]))
        prev_f = current_f

    if len(intervals) < m_level:
        raise ValueError(f"Not enough intervals for n={n}. Needed {m_level}, found {len(intervals)}")

    E_low, E_high = intervals[m_level - 1]
    # 应用二分法
    for _ in range(100):  # 防止无限循环
        E_mid = (E_low + E_high) / 2
        if E_high - E_low <= precision:
            break
        f_mid = f(E_mid)
        if f_mid * f(E_low) <= 0:
            E_high = E_mid
        else:
            E_low = E_mid
    return round((E_low + E_high) / 2, 3)  # 保留三位小数


def main():
    """
    主函数，执行方势阱能级的计算和可视化
    """
    # 参数设置
    V = 20.0  # 势阱高度 (eV)
    w = 1e-9  # 势阱宽度 (m)
    m = ELECTRON_MASS  # 粒子质量 (kg)
    
    # 1. 计算并绘制函数曲线
    E_values = np.linspace(0.001, 19.999, 1000)  # 能量范围 (eV)
    y1, y2, y3 = calculate_y_values(E_values, V, w, m)
    fig = plot_energy_functions(E_values, y1, y2, y3)
    plt.savefig('energy_functions.png', dpi=300)
    plt.show()
    
    # 2. 使用二分法计算前6个能级
    energy_levels = []
    for n in range(6):
        energy = find_energy_level_bisection(n, V, w, m)
        energy_levels.append(energy)
        print(f"能级 {n}: {energy:.3f} eV")
    
    # 与参考值比较
    reference_levels = [0.318, 1.270, 2.851, 5.050, 7.850, 11.215]
    print("\n参考能级值:")
    for n, ref in enumerate(reference_levels):
        print(f"能级 {n}: {ref:.3f} eV")


if __name__ == "__main__":
    main()
