import numpy as np
from scipy.optimize import newton

# ==== 常量定义 ====
L_head = 2.86  # 龙头两孔距离 (m)
L_body = 1.65  # 龙身/龙尾两孔距离 (m)
total_sections = 223  # 总节数
points = total_sections + 1  # 总把手数
L_list = [L_head] + [L_body] * (total_sections - 1)
board_lengths = [3.41] + [2.2] * (total_sections - 1)  
width = 0.3  # 板凳板宽 (m)
theta0 = 32.0 * np.pi  # 初始极角 (32π)
radius_limit = 4.5  # 调头空间半径 (m)

# 弧长函数
def arc_length(theta_start, theta_end, k):
    def F(th):
        return 0.5 * k * np.sqrt(1 + th**2) * th + 0.5 * k * np.arcsinh(th)
    return abs(F(theta_end) - F(theta_start))

# 约束方程及导数（参数传递k）
def constraint(theta_next, theta_curr, L, k):
    r1 = k * theta_curr
    r2 = k * theta_next
    delta = theta_next - theta_curr
    return (r1**2 + r2**2 - 2 * r1 * r2 * np.cos(delta)) - L**2

def constraint_prime(theta_next, theta_curr, L, k):
    r1 = k * theta_curr
    r2 = k * theta_next
    delta = theta_next - theta_curr
    dr2_dtheta = k
    ddelta_dtheta = 1
    term1 = 2 * r2 * dr2_dtheta
    term2 = -2 * r1 * (dr2_dtheta * np.cos(delta) - r2 * np.sin(delta) * ddelta_dtheta)
    return term1 + term2

# ==== 1. 计算把手极角与坐标（参数传递k）====
def compute_states_at_boundary(k):
    # 龙头到达边界时的极角
    theta_head = radius_limit / k

    # 递推其余把手极角
    thetas = [theta_head]
    for L in L_list:
        prev = thetas[-1]
        g = lambda th: constraint(th, prev, L, k)
        gprime = lambda th: constraint_prime(th, prev, L, k)
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

# ==== 2. 生成矩形四角（保持不变）====
def rectangle_corners(p1, p2, seg_length):
    axis = p2 - p1
    length = np.linalg.norm(axis)
    u = axis / length
    v = np.array([-u[1], u[0]])
    center = (p1 + p2) / 2
    hl = seg_length / 2
    hw = width / 2
    corners = np.array([
        center + hl * u + hw * v,
        center + hl * u - hw * v,
        center - hl * u - hw * v,
        center - hl * u + hw * v,
    ])
    return corners

# ==== 3. SAT 碰撞判断（保持不变）====
def polygons_intersect(poly1, poly2):
    def project(poly, axis):
        dots = poly.dot(axis)
        return dots.min(), dots.max()

    def overlap(a, b):
        return not (a[1] < b[0] or b[1] < a[0])

    for i in range(len(poly1)):
        edge = poly1[(i + 1) % len(poly1)] - poly1[i]
        axis = np.array([-edge[1], edge[0]])
        axis /= np.linalg.norm(axis)
        a_min, a_max = project(poly1, axis)
        b_min, b_max = project(poly2, axis)
        if not overlap((a_min, a_max), (b_min, b_max)):
            return False

    for i in range(len(poly2)):
        edge = poly2[(i + 1) % len(poly2)] - poly2[i]
        axis = np.array([-edge[1], edge[0]])
        axis /= np.linalg.norm(axis)
        a_min, a_max = project(poly1, axis)
        b_min, b_max = project(poly2, axis)
        if not overlap((a_min, a_max), (b_min, b_max)):
            return False

    return True

# ==== 4. 检测碰撞（参数传递k）====
def detect_collision_at_boundary(k):
    thetas, pos = compute_states_at_boundary(k)
    N = total_sections
    rects = []

    # 生成所有矩形
    for i in range(N):
        rects.append(rectangle_corners(pos[i], pos[i+1],board_lengths[i]))

    # 检测非相邻节段碰撞
    for i in range(N):
        for j in range(i + 2, N):
            # 极角剪枝
            avg_theta_i = 0.5 * (thetas[i] + thetas[i+1])
            avg_theta_j = 0.5 * (thetas[j] + thetas[j+1])
            if abs(avg_theta_j - avg_theta_i) > 3 * np.pi:
                continue

            if polygons_intersect(rects[i], rects[j]):
                return True
    return False

def get_theta1_for_k(k):
    return radius_limit / k

def get_total_time_for_k(k):
    theta1 = get_theta1_for_k(k)
    s = arc_length(theta1, theta0, k)
    return s / 1.0

def find_collision_time_for_k(k, tol=1e-3):
    def compute_states(t):
        theta_head = newton(lambda th: arc_length(th, theta0, k) - t,
                             theta0 - 0.01, tol=1e-9, maxiter=100)
        thetas = [theta_head]
        for L in L_list:
            prev = thetas[-1]
            g = lambda th: constraint(th, prev, L, k)
            gprime = lambda th: constraint_prime(th, prev, L, k)
            guess = prev + L / (k * prev)
            theta_i = newton(g, guess, fprime=gprime, tol=1e-9, maxiter=100)
            thetas.append(theta_i)
        thetas = np.array(thetas)
        r = k * thetas
        x = r * np.cos(thetas)
        y = r * np.sin(thetas)
        positions = np.vstack((x, y)).T
        return thetas, positions

    def detect_any_collision(t):
        thetas, pos = compute_states(t)
        N = len(thetas) - 1
        rects = [rectangle_corners(pos[i], pos[i+1], board_lengths[i]) for i in range(N)]
        for i in range(N):
            for j in range(i + 2, N):
                avg_theta_j = 0.5 * (thetas[j] + thetas[j+1])
                avg_theta_i = 0.5 * (thetas[i] + thetas[i+1])
                if avg_theta_j - avg_theta_i > 3 * np.pi:
                    continue  # 剪枝：极角差距过大
                if polygons_intersect(rects[i], rects[j]):
                    return True
        return False

    # 二分查找首次碰撞时刻
    t_low = 0.0
    t_high = get_total_time_for_k(k)
    if detect_any_collision(t_low):
        return t_low
    while t_high - t_low > tol:
        mid = 0.5 * (t_low + t_high)
        if detect_any_collision(mid):
            t_high = mid
        else:
            t_low = mid
    if detect_any_collision(t_high):
        return t_high
    return None

def find_min_k():
    k_low = radius_limit / theta0
    k_high = 0.0875

    while k_high - k_low > 1e-4:
        mid = 0.5 * (k_low + k_high)
        t_total = get_total_time_for_k(mid)
        tcol = find_collision_time_for_k(mid)
        if tcol is None or tcol > t_total:
            # 全程无碰撞，可以尝试更小的k
            k_high = mid
        else:
            # 有碰撞，k要变大
            k_low = mid
    return k_high

# ==== 主程序 ====
if __name__ == '__main__': 
    # 查找最小k值
    min_k = find_min_k()
    
    # 计算螺距（螺距 = 2πk）
    min_pitch = 2 * np.pi * min_k
    
    # 输出结果
    print(f"最小螺距k值: {min_k:.6f} m")
    print(f"最小螺距: {min_pitch:.6f} m")
