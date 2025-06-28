import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import pi, sin, cos, atan, asin, sqrt, log
import pandas as pd

# =======================
# 1. 参数定义部分
# =======================

board_head_length = 2.86      # 龙头长度
board_body_length = 1.65      # 龙身单节长度
total_sections = 223          # 节数
width = 0.3                   # 板宽
d = 1.7                       # 螺距
turn_radius = 4.5             # 调头空间半径

theta = 2 * pi * turn_radius / d        # 螺线终点极角
alpha = atan(theta)                     # 螺线切线与x轴夹角

# 调头圆弧半径
turn_radius1 = 3 / sin(alpha)
turn_radius2 = 3 / (2 * sin(alpha))

# 调头圆心坐标
turn_center1_x = turn_radius * cos(theta) - turn_radius1 * sin(theta + alpha)
turn_center1_y = turn_radius * sin(theta) + turn_radius1 * cos(theta + alpha)
turn_center2_x = -turn_radius * cos(theta) + turn_radius2 * sin(theta + alpha)
turn_center2_y = -turn_radius * sin(theta) - turn_radius2 * cos(theta + alpha)

# 关键点坐标
spiral_end_x = d / (2 * pi) * theta * cos(theta)
spiral_end_y = d / (2 * pi) * theta * sin(theta)
spiral_start_x = -spiral_end_x
spiral_start_y = -spiral_end_y
spiral_middle_x = spiral_end_x / 3 + spiral_start_x * 2 / 3
spiral_middle_y = spiral_end_y / 3 + spiral_start_y * 2 / 3

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =======================
# 2. 基础工具函数
# =======================

def binary_search(fn, lower, upper, increasing, eps=1e-8):
    """二分法求解方程根"""
    mid = (lower + upper) / 2
    while upper - lower > eps:
        mid_res = fn(mid)
        if increasing:
            if mid_res < 0:
                lower = mid
            else:
                upper = mid
        else:
            if mid_res > 0:
                lower = mid
            else:
                upper = mid
        mid = (lower + upper) / 2
    return mid

def euclidean_distance(p1, p2):
    """两点间欧氏距离"""
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def arc_to_x(arc):
    """根据段类型和角度返回x坐标"""
    seg_type, angle = arc
    if seg_type == 1:
        return d / (2 * pi) * angle * cos(angle)
    elif seg_type == 2:
        return turn_center1_x + turn_radius1 * cos(angle)
    elif seg_type == 3:
        return turn_center2_x - turn_radius2 * cos(angle)
    elif seg_type == 4:
        return d / (2 * pi) * (angle + pi) * cos(angle)

def arc_to_y(arc):
    """根据段类型和角度返回y坐标"""
    seg_type, angle = arc
    if seg_type == 1:
        return d / (2 * pi) * angle * sin(angle)
    elif seg_type == 2:
        return turn_center1_y + turn_radius1 * sin(angle)
    elif seg_type == 3:
        return turn_center2_y - turn_radius2 * sin(angle)
    elif seg_type == 4:
        return d / (2 * pi) * (angle + pi) * sin(angle)

# =======================
# 3. 运动学递推主链
# =======================

def get_head_position(t, v0=1):
    """返回龙头当前段编号和角度"""
    if t <= 0:
        par1 = theta * sqrt(1 + theta**2)
        par2 = log(theta + sqrt(1 + theta**2))
        par3 = 4 * pi * v0 * t / d
        def fn(x):
            return par1 + par2 - par3 - x * sqrt(1 + x**2) - log(x + sqrt(1 + x**2))
        return 1, binary_search(fn, theta, theta * 2, increasing=False)
    elif t < 6 * alpha / (sin(alpha) * v0):
        theta_n = theta + alpha - pi / 2 - v0 * t / turn_radius1
        return 2, theta_n
    elif t < 9 * alpha / (sin(alpha) * v0):
        theta_n = theta - alpha - pi / 2 + v0 * (t - 2 * alpha * turn_radius1 / v0) / turn_radius2
        return 3, theta_n
    else:
        par1 = theta * sqrt(1 + theta**2)
        par2 = log(theta + sqrt(1 + theta**2))
        par3 = 4 * pi * v0 * (t - 2 * alpha * (turn_radius1 + turn_radius2) / v0) / d
        def fn(x):
            return x * sqrt(1 + x**2) + log(x + sqrt(1 + x**2)) - par1 - par2 - par3
        return 4, binary_search(fn, theta, theta * 20, increasing=True) - pi

def get_next_position(theta_n, is_head=False):
    """递推计算下一个节点的段编号和角度"""
    l = board_head_length if is_head else board_body_length
    seg_type, angle = theta_n
    if seg_type == 1:
        par1 = (2 * pi * l / d) ** 2
        def fn(x):
            return angle**2 + x**2 - 2 * angle * x * cos(x - angle) - par1
        return 1, binary_search(fn, angle, angle + pi, increasing=True)
    elif seg_type == 2:
        xn, yn = arc_to_x(theta_n), arc_to_y(theta_n)
        if euclidean_distance((xn, yn), (spiral_end_x, spiral_end_y)) <= l:
            def fn(x):
                par1 = d / (2 * pi) * x * cos(x)
                par2 = d / (2 * pi) * x * sin(x)
                return (xn - par1)**2 + (yn - par2)**2 - l**2
            return 1, binary_search(fn, angle, angle + pi, increasing=True)
        else:
            delta_theta = 2 * asin(l / (2 * turn_radius1))
            return 2, angle + delta_theta
    elif seg_type == 3:
        xn, yn = arc_to_x(theta_n), arc_to_y(theta_n)
        if euclidean_distance((xn, yn), (spiral_middle_x, spiral_middle_y)) <= l:
            def fn(x):
                par1 = xn - turn_center1_x - turn_radius1 * cos(x)
                par2 = yn - turn_center1_y - turn_radius1 * sin(x)
                return par1**2 + par2**2 - l**2
            return 2, binary_search(fn, theta - alpha - pi / 2, theta + alpha - pi / 2, increasing=True)
        else:
            delta_theta = 2 * asin(l / (2 * turn_radius2))
            return 3, angle - delta_theta
    elif seg_type == 4:
        xn, yn = arc_to_x(theta_n), arc_to_y(theta_n)
        if euclidean_distance((xn, yn), (spiral_start_x, spiral_start_y)) <= l and angle <= theta:
            def fn(x):
                par1 = xn - turn_center2_x + turn_radius2 * cos(x)
                par2 = yn - turn_center2_y + turn_radius2 * sin(x)
                return par1**2 + par2**2 - l**2
            return 3, binary_search(fn, theta - alpha - pi / 2, theta + alpha - pi / 2, increasing=False)
        else:
            par1 = (2 * pi * l / d) ** 2
            def fn(x):
                return (angle + pi)**2 + (x + pi)**2 - 2 * (angle + pi) * (x + pi) * cos(x - angle) - par1
            return 4, binary_search(fn, angle - pi, angle, increasing=False)

def get_all_positions_and_angles(t, v0=1):
    """返回所有节点的段编号、角度和坐标"""
    angs = []
    poses = []
    theta_n = get_head_position(t, v0)
    angs.append(theta_n)
    poses.append((arc_to_x(theta_n), arc_to_y(theta_n)))
    for n in range(total_sections):
        theta_n = get_next_position(theta_n, is_head=(n==0))
        angs.append(theta_n)
        poses.append((arc_to_x(theta_n), arc_to_y(theta_n)))
    return angs, poses

def get_all_positions(t, v0=1):
    """返回所有节点的段编号和坐标"""
    angs, poses = get_all_positions_and_angles(t, v0)
    return angs, poses

def get_all_velocities_analytic(t, v0=1):
    """
    解析递推法计算所有节点在时刻t的速度
    返回速度列表，顺序与get_all_positions_and_angles一致
    """
    angs, poses = get_all_positions_and_angles(t, v0)
    velocities = []

    # 龙头速度
    seg_type, angle = angs[0]
    if seg_type == 1:
        d_theta = -2 * pi * v0 / d / sqrt(1 + angle ** 2)
    elif seg_type == 2:
        d_theta = -v0 / turn_radius1
    elif seg_type == 3:
        d_theta = v0 / turn_radius2
    elif seg_type == 4:
        d_theta = 2 * pi * v0 / d / sqrt(1 + (angle + pi) ** 2)
    velocities.append(v0)
    d_theta_prev = d_theta

    for n in range(total_sections):
        l = board_head_length if n == 0 else board_body_length
        seg_type1, angle1 = angs[n]
        seg_type2, angle2 = angs[n + 1]
        x1, y1 = poses[n]
        x2, y2 = poses[n + 1]

        if seg_type1 == 1:
            par1 = angle1 - angle2 * cos(angle2 - angle1) - angle1 * angle2 * sin(angle2 - angle1)
            par2 = angle2 - angle1 * cos(angle2 - angle1) + angle1 * angle2 * sin(angle2 - angle1)
            d_theta1 = -par1 / par2 * d_theta_prev
            v_next = d / (2 * pi) * abs(d_theta1) * sqrt(1 + angle2 ** 2)
            velocities.append(v_next)
        elif seg_type1 == 2:
            if seg_type2 == 2:
                d_theta1 = d_theta_prev
                velocities.append(velocities[n])
            elif seg_type2 == 1:
                x1d = velocities[n] * sin(angle1)
                y1d = -velocities[n] * cos(angle1)
                par1 = (x1 - d / (2 * pi) * angle2 * cos(angle2)) * x1d + (y1 - d / (2 * pi) * angle2 * sin(angle2)) * y1d
                par2 = (x1 - d / (2 * pi) * angle2 * cos(angle2)) * (cos(angle2) - angle2 * sin(angle2)) + (y1 - d / (2 * pi) * angle2 * sin(angle2)) * (sin(angle2) + angle2 * cos(angle2))
                d_theta1 = 2 * pi / d * par1 / par2
                v_next = d / (2 * pi) * abs(d_theta1) * sqrt(1 + angle2 ** 2)
                velocities.append(v_next)
        elif seg_type1 == 3:
            if seg_type2 == 3:
                d_theta1 = d_theta_prev
                velocities.append(velocities[n])
            elif seg_type2 == 2:
                x1d = velocities[n] * sin(angle1)
                y1d = -velocities[n] * cos(angle1)
                par1 = (x1 - turn_center1_x - turn_radius1 * cos(angle2)) * x1d + (y1 - turn_center1_y - turn_radius1 * sin(angle2)) * y1d
                par2 = (x1 - turn_center1_x - turn_radius1 * cos(angle2)) * sin(angle2) - (y1 - turn_center1_y - turn_radius1 * sin(angle2)) * cos(angle2)
                d_theta1 = -par1 / (turn_radius1 * par2)
                v_next = abs(d_theta1) * turn_radius1
                velocities.append(v_next)
        elif seg_type1 == 4:
            if seg_type2 == 4:
                angle1p, angle2p = angle1 + pi, angle2 + pi
                par1 = angle1p - angle2p * cos(angle2p - angle1p) - angle1p * angle2p * sin(angle2p - angle1p)
                par2 = angle2p - angle1p * cos(angle2p - angle1p) + angle1p * angle2p * sin(angle2p - angle1p)
                d_theta1 = -par1 / par2 * d_theta_prev
                v_next = d / (2 * pi) * abs(d_theta1) * sqrt(1 + angle2p ** 2)
                velocities.append(v_next)
            elif seg_type2 == 3:
                beta = angle1 + atan(angle1 + pi)
                x1d = velocities[n] * cos(beta)
                y1d = velocities[n] * sin(beta)
                par1 = (x1 - turn_center2_x + turn_radius2 * cos(angle2)) * x1d + (y1 - turn_center2_y + turn_radius2 * sin(angle2)) * y1d
                par2 = (x1 - turn_center2_x + turn_radius2 * cos(angle2)) * sin(angle2) - (y1 - turn_center2_y + turn_radius2 * sin(angle2)) * cos(angle2)
                d_theta1 = par1 / (turn_radius2 * par2)
                v_next = abs(d_theta1) * turn_radius2
                velocities.append(v_next)
        d_theta_prev = d_theta1
    return velocities

def find_max_safe_head_velocity(t_range, v_limit=2.0, tol=1e-3):
    """
    二分法搜索使所有节点在所有时刻速度都不超过v_limit的最大龙头速度v0
    每次只输出当前尝试的v0和本轮所有节点的最大速度
    """
    v0_low = 0.1
    v0_high = 10.0
    step = 0
    while v0_high - v0_low > tol:
        v0_mid = (v0_low + v0_high) / 2
        max_speed = 0
        safe = True
        for t in t_range:
            velocities = get_all_velocities_analytic(t, v0_mid)
            cur_max = max(velocities)
            if cur_max > max_speed:
                max_speed = cur_max
            if cur_max > v_limit:
                safe = False
                break
        print(f"尝试v0={v0_mid:.5f}，本轮最大速度={max_speed:.5f} m/s")
        step += 1
        if safe:
            v0_low = v0_mid
        else:
            v0_high = v0_mid
    return v0_low

if __name__ == '__main__':
    t_range = range(-100, 101, 1)
    max_safe_v0 = find_max_safe_head_velocity(t_range, v_limit=2.0)
    print(f"所有节点在所有时刻速度都不超过2m/s时，龙头最大允许速度为：{max_safe_v0:.4f} m/s")