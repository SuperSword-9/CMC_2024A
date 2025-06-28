import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import pi, sin, cos, atan, asin, sqrt, log
import pandas as pd

# =======================
# 1. 参数定义部分
# =======================

width = 0.3  # 板凳宽度
board_head_length = 2.86  # 龙头板长
board_body_length = 1.65  # 龙身及龙尾板长度
total_sections = 223  # 板凳总节数

d = 1.7  # 螺距
turn_radius = 4.5  # 调头空间半径

theta = 2 * pi * turn_radius / d  # 螺线终点极角
alpha = atan(theta)  # 螺线切线与x轴夹角

turn_radius1 = 3 / sin(alpha)  # 大圆半径
turn_radius2 = 3 / (2 * sin(alpha))  # 小圆半径

# 大圆圆心坐标
turn_center1_x = turn_radius * cos(theta) - turn_radius1 * sin(theta + alpha)
turn_center1_y = turn_radius * sin(theta) + turn_radius1 * cos(theta + alpha)
# 小圆圆心坐标
turn_center2_x = -turn_radius * cos(theta) + turn_radius2 * sin(theta + alpha)
turn_center2_y = -turn_radius * sin(theta) - turn_radius2 * cos(theta + alpha)

# 螺线与大圆、小圆的切点坐标
spiral_end_x = d / (2 * pi) * theta * cos(theta)  # 螺线终点（大圆切点）
spiral_end_y = d / (2 * pi) * theta * sin(theta)
spiral_start_x = -spiral_end_x  # 螺线起点（小圆切点）
spiral_start_y = -spiral_end_y
spiral_middle_x = spiral_end_x / 3 + spiral_start_x * 2 / 3  # 螺线与小圆的中间切点
spiral_middle_y = spiral_end_y / 3 + spiral_start_y * 2 / 3

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =======================
# 2. 基础工具函数
# =======================

def binary_search(fn, lower, upper, increasing, eps=1e-8):
    """二分法求解方程根，用于递推极角"""
    cur = (lower + upper) / 2
    while upper - lower > eps:
        cur_res = fn(cur)
        if increasing:
            if cur_res < 0:
                lower = cur
            else:
                upper = cur
        else:
            if cur_res > 0:
                lower = cur
            else:
                upper = cur
        cur = (lower + upper) / 2
    return cur

def dist(p1, p2):
    """两点欧氏距离"""
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def arc_x(arc):
    """根据段类型和极角，返回x坐标"""
    seg_type, angle = arc
    if seg_type == 1:
        return d / (2 * pi) * angle * cos(angle)  # 螺线段
    elif seg_type == 2:
        return turn_center1_x + turn_radius1 * cos(angle)  # 大圆段
    elif seg_type == 3:
        return turn_center2_x - turn_radius2 * cos(angle)  # 小圆段
    elif seg_type == 4:
        return d / (2 * pi) * (angle + pi) * cos(angle)  # 盘出螺线段

def arc_y(arc):
    """根据段类型和极角，返回y坐标"""
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

def head_position(t, v0=1):
    """
    给定时刻t和龙头速度v0，返回龙头当前段类型和极角
    1: 螺线段 2: 大圆段 3: 小圆段 4: 盘出螺线段
    """
    if t <= 0:
        # 盘入螺线段，需二分求解极角
        par1 = theta * sqrt(1 + theta**2)
        par2 = log(theta + sqrt(1 + theta**2))
        par3 = 4 * pi * v0 * t / d
        def fn(x):
            return par1 + par2 - par3 - x * sqrt(1 + x**2) - log(x + sqrt(1 + x**2))
        return 1, binary_search(fn, theta, theta * 2, increasing=False)
    elif t < 6 * alpha / (sin(alpha) * v0):
        # 大圆段，直接线性推进极角
        theta_n = theta + alpha - pi / 2 - v0 * t / turn_radius1
        return 2, theta_n
    elif t < 9 * alpha / (sin(alpha) * v0):
        # 小圆段
        theta_n = theta - alpha - pi / 2 + v0 * (t - 2 * alpha * turn_radius1 / v0) / turn_radius2
        return 3, theta_n
    else:
        # 盘出螺线段，需二分求解极角
        par1 = theta * sqrt(1 + theta**2)
        par2 = log(theta + sqrt(1 + theta**2))
        par3 = 4 * pi * v0 * (t - 2 * alpha * (turn_radius1 + turn_radius2) / v0) / d
        def fn(x):
            return x * sqrt(1 + x**2) + log(x + sqrt(1 + x**2)) - par1 - par2 - par3
        return 4, binary_search(fn, theta, theta * 20, increasing=True) - pi

def next_position(theta_n, is_head=False):
    """
    递推求解下一个节点的段类型和极角
    is_head: 是否为龙头后的第一个节点（决定用头长还是身长）
    """
    l = board_head_length if is_head else board_body_length
    seg_type, angle = theta_n
    if seg_type == 1:
        # 螺线段递推
        par1 = (2 * pi * l / d) ** 2
        def fn(x):
            return angle**2 + x**2 - 2 * angle * x * cos(x - angle) - par1
        return 1, binary_search(fn, angle, angle + pi, increasing=True)
    elif seg_type == 2:
        # 大圆段递推
        xn, yn = arc_x(theta_n), arc_y(theta_n)
        if dist((xn, yn), (spiral_end_x, spiral_end_y)) <= l:
            # 跳回螺线段
            def fn(x):
                par1 = d / (2 * pi) * x * cos(x)
                par2 = d / (2 * pi) * x * sin(x)
                return (xn - par1)**2 + (yn - par2)**2 - l**2
            return 1, binary_search(fn, angle, angle + pi, increasing=True)
        else:
            # 继续大圆段
            delta_theta = 2 * asin(l / (2 * turn_radius1))
            return 2, angle + delta_theta
    elif seg_type == 3:
        # 小圆段递推
        xn, yn = arc_x(theta_n), arc_y(theta_n)
        if dist((xn, yn), (spiral_middle_x, spiral_middle_y)) <= l:
            # 跳回大圆段
            def fn(x):
                par1 = xn - turn_center1_x - turn_radius1 * cos(x)
                par2 = yn - turn_center1_y - turn_radius1 * sin(x)
                return par1**2 + par2**2 - l**2
            return 2, binary_search(fn, theta - alpha - pi / 2, theta + alpha - pi / 2, increasing=True)
        else:
            # 继续小圆段
            delta_theta = 2 * asin(l / (2 * turn_radius2))
            return 3, angle - delta_theta
    elif seg_type == 4:
        # 盘出螺线段递推
        xn, yn = arc_x(theta_n), arc_y(theta_n)
        if dist((xn, yn), (spiral_start_x, spiral_start_y)) <= l and angle <= theta:
            # 跳回小圆段
            def fn(x):
                par1 = xn - turn_center2_x + turn_radius2 * cos(x)
                par2 = yn - turn_center2_y + turn_radius2 * sin(x)
                return par1**2 + par2**2 - l**2
            return 3, binary_search(fn, theta - alpha - pi / 2, theta + alpha - pi / 2, increasing=False)
        else:
            # 继续盘出螺线段
            par1 = (2 * pi * l / d) ** 2
            def fn(x):
                return (angle + pi)**2 + (x + pi)**2 - 2 * (angle + pi) * (x + pi) * cos(x - angle) - par1
            return 4, binary_search(fn, angle - pi, angle, increasing=False)

def get_all_positions_and_angles(t, v0=1):
    """
    返回所有节点的段编号、角度和坐标
    递推链条，先算龙头，再递推每一节
    """
    angs = []
    poses = []
    theta_n = head_position(t, v0)
    angs.append(theta_n)
    poses.append((arc_x(theta_n), arc_y(theta_n)))
    for n in range(total_sections):
        theta_n = next_position(theta_n, is_head=(n==0))
        angs.append(theta_n)
        poses.append((arc_x(theta_n), arc_y(theta_n)))
    return angs, poses

def get_all_positions(t, v0=1):
    """返回所有节点的段编号、角度和坐标"""
    angs, poses = get_all_positions_and_angles(t, v0)
    return angs, poses

def get_all_velocities_analytic(t, v0=1):
    """
    解析递推法计算所有节点在时刻t的速度
    返回速度列表，顺序与get_all_positions_and_angles一致
    """
    angs, poses = get_all_positions_and_angles(t, v0)
    velocities = []

    # 先算龙头速度
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

    # 递推每一节速度
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

# =======================
# 4. 可视化与输出
# =======================

def plot_dragon_status(poses):
    """绘制某一时刻所有节点的位置"""
    ang = np.linspace(theta, 10 * 2 * pi, 1000)
    xx1 = d / (2 * pi) * ang * np.cos(ang)
    yy1 = d / (2 * pi) * ang * np.sin(ang)
    xx2 = -xx1
    yy2 = -yy1

    plt.figure(figsize=(8, 8))
    plt.plot(xx1, yy1, color='red', linewidth=2, label='盘入螺线', alpha=0.5)
    plt.plot(xx2, yy2, color='blue', linewidth=2, label='盘出螺线', alpha=0.5)
    plt.plot([], [], color='black', linewidth=2, label='调头路径', alpha=0.5)
    circle = plt.Circle((0, 0), 4.5, color='yellow', fill=True, alpha=0.4)
    plt.gca().add_artist(circle)
    arc1 = patches.Arc((turn_center1_x, turn_center1_y), width=turn_radius1 * 2, height=turn_radius1 * 2,
                       theta1=np.rad2deg(theta - alpha - pi / 2),
                       theta2=np.rad2deg(theta + alpha - pi / 2),
                       color='black', linewidth=2, alpha=0.5)
    plt.gca().add_patch(arc1)
    arc2 = patches.Arc((turn_center2_x, turn_center2_y), width=turn_radius2 * 2, height=turn_radius2 * 2,
                       theta1=np.rad2deg(theta - alpha + pi / 2),
                       theta2=np.rad2deg(theta + alpha + pi / 2),
                       color='black', linewidth=2, alpha=0.5)
    plt.gca().add_patch(arc2)
    x_lst = [p[0] for p in poses]
    y_lst = [p[1] for p in poses]
    plt.scatter(x_lst, y_lst, color='black', marker='.')
    plt.axis('equal')
    plt.xlabel('X', size=12)
    plt.ylabel('Y', size=12)
    plt.legend()
    plt.title('板凳龙在50s时各把手状态', size=14)
    plt.grid(True, alpha=0.3)
    plt.show()

def transpose_and_relabel(df, labels):
    """用于输出Excel时转置并重命名行列"""
    df_t = df.set_index("Time(s)").T
    df_t.index.name = None
    df_t.index = labels
    df_t.columns = [f"{int(t)}s" for t in df_t.columns]
    return df_t.reset_index().rename(columns={"index": ""})

def main():
    """
    主流程：
    1. 批量计算所有时刻所有节点的位置和速度
    2. 输出到Excel
    3. 绘制某一时刻的龙形态
    """
    t_range = range(-100, 101, 1)
    n_time = len(t_range)
    n_points = total_sections + 1

    all_positions = np.zeros((n_time, n_points, 2))
    all_velocities = np.zeros((n_time, n_points))

    for idx, t in enumerate(t_range):
        angs, poses = get_all_positions(t)
        for i, pos in enumerate(poses):
            all_positions[idx, i, 0] = pos[0]
            all_positions[idx, i, 1] = pos[1]
        velocities = get_all_velocities_analytic(t)
        all_velocities[idx, :] = velocities

    columns_pos = []
    for i in range(n_points):
        columns_pos.extend([f"P{i+1}_x", f"P{i+1}_y"])
    pos_data = all_positions.reshape(n_time, -1)
    df_all = pd.DataFrame(np.round(pos_data, 6), columns=columns_pos)
    df_all.insert(0, "Time(s)", list(range(-100, 101)))

    columns_v = [f"v_{i+1}" for i in range(n_points)]
    df_v = pd.DataFrame(np.round(all_velocities, 6), columns=columns_v)
    df_v.insert(0, "Time(s)", list(range(-100, 101)))

    labels_positions = ["龙头x（m）", "龙头y（m）"]
    for i in range(1, n_points-2):
        labels_positions += [f"第{i}节龙身x（m）", f"第{i}节龙身y（m）"]
    labels_positions += ["龙尾x（m）", "龙尾y（m）", "龙尾（后）x（m）", "龙尾（后）y（m）"]

    labels_velocity = ["龙头（m/s）"]
    for i in range(1, n_points-2):
        labels_velocity += [f"第{i}节龙身（m/s）"]
    labels_velocity += ["龙尾（m/s）"]
    labels_velocity += ["龙尾（后）（m/s）"]

    df_all_t = transpose_and_relabel(df_all, labels_positions)
    df_v_t = transpose_and_relabel(df_v, labels_velocity)

    with pd.ExcelWriter("result4.xlsx", engine="openpyxl") as writer:
        df_all_t.to_excel(writer, sheet_name="位置", index=False, float_format="%.6f")
        df_v_t.to_excel(writer, sheet_name="速度", index=False, float_format="%.6f")

    print("所有时刻位置和速度已输出为 result4.xlsx 文件。")

    t = 50
    angs, poses = get_all_positions(t)
    plot_dragon_status(poses)

if __name__ == '__main__':
    main()