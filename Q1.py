import numpy as np
import pandas as pd
from scipy.optimize import newton
import matplotlib.pyplot as plt

# 常量定义
k = 0.55 / (2 * np.pi)  # 螺线参数
L_head = 2.86           # 龙头节长
L_body = 1.65           # 龙身/龙尾节长
total_sections = 223    # 总节数
points = total_sections + 1  # 总把手数

# 构建每节长度列表
L_list = [L_head] + [L_body] * (total_sections - 1)

theta0 = 32.0 * np.pi   # 初始龙头极角

# 螺线弧长
def arc_length(theta_start, theta_end, k):
    def F(th):
        return 0.5 * k * np.sqrt(1 + th**2) * th + 0.5 * k * np.arcsinh(th)
    return abs(F(theta_end) - F(theta_start))

# 两节点极角约束
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

# 时间步
time_range = range(0, 301)
n_times = len(time_range)

# 结果数组
all_theta = np.zeros((n_times, points))      # 各节点极角
all_velocities = np.zeros((n_times, points)) # 各节点速度

# 主循环：每个时刻递推所有节点极角和速度
for t_idx, t in enumerate(time_range):
    s_head = t

    # 求龙头极角
    def f(th):
        return arc_length(th, theta0, k) - s_head
    theta_head = newton(f, theta0 - 0.01, maxiter=100, tol=1e-8)
    all_theta[t_idx, 0] = theta_head

    # 递推其余节点极角
    for i in range(1, points):
        theta_prev = all_theta[t_idx, i - 1]
        L_curr = L_list[i - 1]
        g = lambda th: constraint(th, theta_prev, L_curr)
        g_prime = lambda th: constraint_prime(th, theta_prev, L_curr)
        guess = theta_prev + L_curr / (k * theta_prev)
        theta_i = newton(g, guess, fprime=g_prime, maxiter=100, tol=1e-8)
        all_theta[t_idx, i] = theta_i

    # 速度递推
    v1 = 1
    theta1 = all_theta[t_idx, 0]
    theta1_prime = v1 / (k * np.sqrt(1 + theta1**2))
    all_velocities[t_idx, 0] = v1

    theta_prime_prev = theta1_prime
    for i in range(1, points):
        theta_prev = all_theta[t_idx, i - 1]
        theta_now = all_theta[t_idx, i]
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

        v_now = k * abs(theta_prime_now) * np.sqrt(1 + theta_now**2)
        all_velocities[t_idx, i] = v_now
        theta_prime_prev = theta_prime_now

# 极角转空间坐标
positions = np.zeros((n_times, points, 2))
for t_idx in range(n_times):
    for i in range(points):
        theta = all_theta[t_idx, i]
        r = k * theta
        positions[t_idx, i, 0] = r * np.cos(theta)
        positions[t_idx, i, 1] = r * np.sin(theta)

# 输出所有时刻的节点坐标
columns_pos = []
for i in range(points):
    columns_pos.extend([f"P{i+1}_x", f"P{i+1}_y"])
pos_data = positions.reshape(n_times, -1)
df_all = pd.DataFrame(np.round(pos_data, 6), columns=columns_pos)
df_all.insert(0, "Time(s)", list(time_range))
df_all.to_excel("所有时刻龙身位置.xlsx", index=False, float_format="%.6f")

# 输出所有时刻的节点速度
columns_v = [f"v_{i+1}" for i in range(points)]
df_v = pd.DataFrame(np.round(all_velocities, 6), columns=columns_v)
df_v.insert(0, "Time(s)", list(time_range))
df_v.to_excel("dragon_velocity_only.xlsx", index=False, float_format="%.6f")

# 转置并重命名行列
def transpose_and_relabel(df, labels):
    df_t = df.set_index("Time(s)").T
    df_t.index.name = None
    df_t.index = labels
    df_t.columns = [f"{int(t)}s" for t in df_t.columns]
    return df_t.reset_index().rename(columns={"index": ""})

labels_positions = ["龙头x（m）", "龙头y（m）"]
for i in range(1, 222):
    labels_positions += [f"第{i}节龙身x（m）", f"第{i}节龙身y（m）"]
labels_positions += ["龙尾x（m）", "龙尾y（m）", "龙尾（后）x（m）", "龙尾（后）y（m）"]

labels_velocity = ["龙头（m/s）"]
for i in range(1, 222):
    labels_velocity += [f"第{i}节龙身（m/s）"]
labels_velocity += ["龙尾（m/s）"]
labels_velocity += ["龙尾（后）（m/s）"]

df_all_t = transpose_and_relabel(df_all, labels_positions)
df_v_t = transpose_and_relabel(df_v, labels_velocity)

with pd.ExcelWriter("result1.xlsx", engine="openpyxl") as writer:
    df_all_t.to_excel(writer, sheet_name="位置", index=False, float_format="%.6f")
    df_v_t.to_excel(writer, sheet_name="速度",  index=False, float_format="%.6f")

print("所有输出已完成。")

# 可视化部分

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取速度数据并绘制部分节点速度随时间变化
df_v = pd.read_excel('dragon_velocity_only.xlsx')
time = df_v['Time(s)']
v_head = df_v['v_1']
v_mid  = df_v['v_52']
v_tail = df_v['v_224']

plt.figure()
plt.plot(time, v_head)
plt.plot(time, v_mid)
plt.plot(time, v_tail)
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('板凳龙部分节点得速度随时间变化')
plt.legend(['龙头', '第51节', '龙尾'])
plt.show()

# 读取位置数据并绘制关键节点轨迹
df_pos  = pd.read_excel('所有时刻龙身位置.xlsx')
x_head  = df_pos['P1_x'];   y_head  = df_pos['P1_y']
x_mid   = df_pos['P52_x'];  y_mid   = df_pos['P52_y']
x_tail  = df_pos['P224_x']; y_tail  = df_pos['P224_y']

plt.figure()
plt.plot(x_head,  y_head)
plt.plot(x_mid,   y_mid)
plt.plot(x_tail,  y_tail)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('板凳龙关键节点轨迹')
plt.legend(['龙头', '第51节', '龙尾'])
plt.axis('equal')
plt.show()

# 绘制0s与300s时刻的把手位置及理论螺旋
df_pos = pd.read_excel('所有时刻龙身位置.xlsx')
df0 = df_pos[df_pos['Time(s)'] == 0].iloc[0]
df300 = df_pos[df_pos['Time(s)'] == 300].iloc[0]
xs0 = df0.filter(regex='_x').values
ys0 = df0.filter(regex='_y').values
xs300 = df300.filter(regex='_x').values
ys300 = df300.filter(regex='_y').values

k = 0.55 / (2 * np.pi)
theta_max = 32 * np.pi
theta_min = xs300.size and min(np.arctan2(ys300, xs300) / 1)
theta = np.linspace(theta_min, theta_max, 1000)
r = k * theta
spiral_x = r * np.cos(theta)
spiral_y = r * np.sin(theta)

plt.figure(figsize=(8, 8))
plt.plot(spiral_x, spiral_y, color='gray', alpha=0.3, label='阿基米德螺旋 (理论)')
plt.plot(xs0, ys0, '-k', linewidth=1.5, label='初始形状 (0s)')
plt.scatter(xs0, ys0, color='red', s=20, label='把手 (0s)')
plt.plot(xs300, ys300, '--', linewidth=1.5, color='blue', label='收缩后 (300s)')
plt.scatter(xs300, ys300, color='cyan', s=20, label='把手 (300s)')
plt.axis('equal')
plt.title('板凳龙在0s与300s时把手位置示意图')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()
plt.grid(True)
plt.show()