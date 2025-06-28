import numpy as np
from scipy.optimize import newton
import time
import pandas as pd

# ==== 常量定义，与原始脚本完全一致 ====
k = 0.55 / (2 * np.pi)  # 螺线参数 (m)
L_head = 2.86  # 龙头两孔距离 (m)
L_body = 1.65  # 龙身/龙尾两孔距离 (m)
total_sections = 223  # 总节数
points = total_sections + 1  # 总把手数
L_list = [L_head] + [L_body] * (total_sections - 1)
board_lengths = [3.41] + [2.2] * (total_sections - 1)
theta0 = 32.0 * np.pi  # 初始极角 (32π)
width = 0.3  # 板凳板宽 (m)

# 弧长函数
def arc_length(theta_start, theta_end, k):
    def F(th):
        return 0.5 * k * np.sqrt(1 + th**2) * th + 0.5 * k * np.arcsinh(th)
    return abs(F(theta_end) - F(theta_start))

# 约束函数和导数
def constraint(theta_next, theta_curr, L):
    r1, r2 = k * theta_curr, k * theta_next
    delta = theta_next - theta_curr
    return (r1**2 + r2**2 - 2 * r1 * r2 * np.cos(delta)) - L**2

def constraint_prime(theta_next, theta_curr, L):
    r1, r2 = k * theta_curr, k * theta_next
    delta = theta_next - theta_curr
    dr2_dtheta = k
    ddelta_dtheta = 1
    term1 = 2 * r2 * dr2_dtheta
    term2 = -2 * r1 * (dr2_dtheta * np.cos(delta) - r2 * np.sin(delta) * ddelta_dtheta)
    return term1 + term2

# ==== 1. 计算把手极角与坐标 ====
def compute_states(t):
    # 1. 求龙头极角
    theta_head = newton(lambda th: arc_length(th, theta0, k) - t,
                         theta0 - 0.01, tol=1e-9, maxiter=100)
    thetas = [theta_head]
    # 2. 递推其余把手极角
    for L in L_list:
        prev = thetas[-1]
        g = lambda th: constraint(th, prev, L)
        gprime = lambda th: constraint_prime(th, prev, L)
        guess = prev + L / (k * prev)
        theta_i = newton(g, guess, fprime=gprime, tol=1e-9, maxiter=100)
        thetas.append(theta_i)
    thetas = np.array(thetas)
    # 3. 坐标转换
    r = k * thetas
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    positions = np.vstack((x, y)).T
    return thetas, positions

# ==== 2. 生成矩形四角 ====
def rectangle_corners(p1, p2, seg_length):
    """
    生成给定板凳段的矩形四角：
    - seg_length: 该段两孔之间的标准距离（L_head 或 L_body）
    """
    axis = p2 - p1
    length = np.linalg.norm(axis)
    u = axis / length
    v = np.array([-u[1], u[0]])
    center = (p1 + p2) / 2
    # 长边半长用标准段长/2
    hl = seg_length / 2
    # 宽度半宽用固定板宽/2
    hw = width / 2
    corners = np.array([
        center + hl * u + hw * v,
        center + hl * u - hw * v,
        center - hl * u - hw * v,
        center - hl * u + hw * v,
    ])
    return corners

# ==== 3. SAT 碰撞判断 ====
def polygons_intersect(poly1, poly2):
    def project(poly, axis):
        dots = poly.dot(axis)
        return dots.min(), dots.max()
    def overlap(a, b):
        return not (a[1] < b[0] or b[1] < a[0])
    for poly in (poly1, poly2):
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            axis /= np.linalg.norm(axis)
            if not overlap(project(poly1, axis), project(poly2, axis)):
                return False
    return True

# ==== 4. 配对检测碰撞 ====
def detect_any_collision(t):
    thetas, pos = compute_states(t)
    N = len(thetas) - 1
    rects = [rectangle_corners(pos[i], pos[i+1], board_lengths[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 2, N):  # 跳过自身及相邻段
            avg_theta_j = 0.5 * (thetas[j] + thetas[j+1])
            avg_theta_i = 0.5 * (thetas[i] + thetas[i+1])
            if avg_theta_j - avg_theta_i > 3 * np.pi:
                continue  # 剪枝：极角差距过大
            if polygons_intersect(rects[i], rects[j]):
                return True
    return False

# ==== 5. 二分法查找首碰时刻 ====
def find_collision_time(t_low=0.0, t_high=300.0, tol=1e-3):
    if detect_any_collision(t_low):
        return t_low
    while not detect_any_collision(t_high):
        t_high += 40
        if t_high > 600:
            return None
    while t_high - t_low > tol:
        mid = 0.5 * (t_low + t_high)
        if detect_any_collision(mid):
            t_high = mid
        else:
            t_low = mid
    return t_high

# ==== 主程序 ====
if __name__ == '__main__':
    start_time = time.time()
    collided_412 = detect_any_collision(412)
    print(f"t=412s 是否碰撞: {collided_412}")
    collided_413 = detect_any_collision(413)
    print(f"t=413s 是否碰撞: {collided_413}")
    tcol = find_collision_time()
    if tcol is None:
        print("未检测到碰撞")
    else:
        print(f"预计首次碰撞时间: {tcol:.3f} 秒")
        # 1. 计算所有把手在碰撞时刻的坐标
        thetas, positions = compute_states(tcol)
        df_pos = pd.DataFrame(positions, columns=['x', 'y'])
        df_pos['theta'] = thetas
        df_pos.index.name = 'handle_index'
        df_pos.to_excel('collision_positions.xlsx')

        # 2. 计算所有把手在碰撞时刻的速度
        dt = 1e-4
        thetas2, positions2 = compute_states(tcol + dt)
        vels = (positions2 - positions) / dt
        speed = np.linalg.norm(vels, axis=1)
        df_vel = pd.DataFrame(vels, columns=['vx', 'vy'])
        df_vel['speed'] = speed
        df_vel.index.name = 'handle_index'
        df_vel.to_excel('collision_velocities.xlsx')
        print("已输出碰撞时刻所有把手的坐标和速度到Excel。")
    end_time = time.time()
    print(f"总运行时间: {end_time - start_time:.2f} 秒")

    # ==== 输出首次碰撞时刻各节点坐标和速度 ====
    if tcol is not None:
        # 计算极角和坐标
        thetas, positions = compute_states(tcol)
        # 计算速度
        velocities = np.zeros_like(thetas)
        # 龙头速度恒为1
        velocities[0] = 1
        # 递推出每一节速度
        k_val = k
        theta_prime_prev = velocities[0] / (k_val * np.sqrt(1 + thetas[0]**2))
        for i in range(1, len(thetas)):
            theta_prev = thetas[i - 1]
            theta_now = thetas[i]
            delta_theta = theta_now - theta_prev
            numerator = (2 * theta_prev
                         - 2 * theta_now * np.cos(delta_theta)
                         - 2 * theta_prev * theta_now * np.sin(delta_theta))
            denominator = (2 * theta_now
                           - 2 * theta_prev * np.cos(delta_theta)
                           + 2 * theta_prev * theta_now * np.sin(delta_theta))
            if abs(denominator) < 1e-12:
                theta_prime_now = theta_prime_prev
            else:
                theta_prime_now = theta_prime_prev * (numerator / denominator)
            v_now = k_val * abs(theta_prime_now) * np.sqrt(1 + theta_now**2)
            velocities[i] = v_now
            theta_prime_prev = theta_prime_now

        # 组织表格
        import pandas as pd
        labels = ["龙头"] + [f"第{i}节龙身" for i in range(1, len(thetas)-1)] + ["龙尾", "龙尾（后）"]
        # “龙尾（后）”坐标和速度用最后一节的延长线推算
        tail_vec = positions[-1] - positions[-2]
        tail_unit = tail_vec / np.linalg.norm(tail_vec)
        tail_after_pos = positions[-1] + L_body * tail_unit
        tail_after_theta = thetas[-1] + (thetas[-1] - thetas[-2])
        # 速度用最后一节速度近似
        tail_after_v = velocities[-1]
        # 合并数据
        xs = np.append(positions[:,0], tail_after_pos[0])
        ys = np.append(positions[:,1], tail_after_pos[1])
        vs = np.append(velocities, tail_after_v)
        df = pd.DataFrame({
            "横坐标x (m)": xs,
            "纵坐标y (m)": ys,
            "速度 (m/s)": vs
        }, index=labels)
        # 第一行第一列空
        df.index.name = ""
        # 输出Excel
        df.to_excel("result2.xlsx", float_format="%.6f")
