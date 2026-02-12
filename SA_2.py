from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import sqrt, dist
import numpy as np
import numba as nb

# =========================
# 1. 场景构造与几何工具 (保持不变)
# =========================


def create_complex_steiner_scene():
    """
    创建一个更大、更复杂的Steiner TSP场景
    地图范围: ~20x20
    特点: 起终点重合(S/E), 包含必须经由的点(Waypoints)与可选经过的点(Steiner Points)
    修正: 调整障碍物位置，确保不覆盖任何节点
    """
    # 起点与终点重合（入口Base）
    start_end_coord = (0.0, 0.0)

    # 必须经过的点（订单屋顶 - 8个）
    waypoints = [
        (8.0, 2.0),    # 订单屋顶A
        (14.0, 2.0),   # 订单屋顶B
        (20.0, 2.0),   # 订单屋顶C
        (8.0, 10.0),   # 订单屋顶D
        (14.0, 10.0),  # 订单屋顶E
        (20.0, 10.0),  # 订单屋顶F
        (0.0, 10.0)    # 订单屋顶G
    ]

    # 可选经过的点（充电桩、走廊接入点 - 5个）
    optional_points = [
        (4.0, 5.0),    # 充电桩C1
        (12.0, 6.0),   # 充电桩C2
        (18.0, 6.0),   # 充电桩C3
        (6.0, -4.0),   # 走廊接入点W1
        (12.0, -4.0)   # 走廊接入点W2
    ]

    # 障碍物（高层建筑、电力线）
    obstacles = [
        # 中央建筑群（禁飞区）
        {'center': (10.0, 6.0), 'radius': 2.5},
        
        # 东侧高层建筑
        {'center': (17.0, 3.0), 'radius': 1.8},
        
        # 北侧建筑
        {'center': (6.0, 8.0), 'radius': 1.5},
        
        # 电力线区域（东北角）
        {'center': (16.0, 9.0), 'radius': 1.2},
        
        # 南侧接入点障碍
        {'center': (9.0, -2.0), 'radius': 1.0}
    ]

    # 整合所有点位
    all_points = [start_end_coord] + waypoints + optional_points + [start_end_coord]
    
    # 标签生成
    point_labels = (
        ['S'] 
        + [f'W{i+1}' for i in range(len(waypoints))] 
        + [f'O{i+1}' for i in range(len(optional_points))] 
        + ['E']
    )
    
    label_to_coord = {label: coord for label, coord in zip(point_labels, all_points)}

    return all_points, point_labels, label_to_coord, waypoints, optional_points, obstacles

def line_circle_intersection(p1, p2, center, radius):
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = center
    dx, dy = x2 - x1, y2 - y1
    a = dx * dx + dy * dy
    if a == 0: return (x1 - cx) ** 2 + (y1 - cy) ** 2 <= radius ** 2
    fx, fy = x1 - cx, y1 - cy
    b = 2 * (fx * dx + fy * dy)
    c = (fx * fx + fy * fy) - radius * radius
    discriminant = b * b - 4 * a * c
    if discriminant < 0: return False
    discriminant = sqrt(discriminant)
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)

def calculate_edge_weight(p1, p2, obstacles, obstacle_penalty=1e6, base_distance_weight=1.0):
    for obstacle in obstacles:
        if line_circle_intersection(p1, p2, obstacle['center'], obstacle['radius']):
            return float(obstacle_penalty)
    return dist(p1, p2) * float(base_distance_weight)

# =========================
# 2. 可视化工具 (保持不变)
# =========================
def visualize_solution(all_points, point_labels, label_to_coord, obstacles, selected_edges, total_distance):
    fig, ax = plt.subplots(figsize=(12, 10))
    for label, coord in label_to_coord.items():
        x, y = coord
        if label == 'S': ax.plot(x, y, 'go', markersize=15, label='Start', markeredgecolor='black', markeredgewidth=2)
        elif label == 'E': ax.plot(x, y, 'ro', markersize=10, label='End', markeredgecolor='black', markeredgewidth=2, alpha=0.7) 
        elif label.startswith('W'): ax.plot(x, y, 'bs', markersize=12, label='Waypoint' if label == 'W1' else '', markeredgecolor='black', markeredgewidth=2)
        else: ax.plot(x, y, 'ko', markersize=8, alpha=0.5, label='Optional' if label == 'O1' else '')
        ax.text(x + 0.1, y + 0.1, label, fontsize=10, fontweight='bold')
    for obstacle in obstacles:
        ax.add_patch(patches.Circle(obstacle['center'], obstacle['radius'], edgecolor='red', facecolor='red', alpha=0.3, linewidth=2))
    for u, v in selected_edges:
        p1, p2 = label_to_coord[u], label_to_coord[v]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=2, alpha=0.8)
        mid_x, mid_y = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.arrow(mid_x, mid_y, (p2[0]-p1[0])*0.1, (p2[1]-p1[1])*0.1, head_width=0.2, head_length=0.3, alpha=0.8)
    
    ax.set_xlim(-5, 22)
    ax.set_ylim(-5, 12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')
    ax.set_title(f'无人机低空配送 - 园区走廊网络 (Ising SA -1/+1 Mode)\nTotal Distance: {total_distance:.2f}')
    return fig, ax
@nb.njit
def _numba_sa_core(N_vars, num_reads, t_max, decay_rate, h_isi, J_isi, num_sweeps):
    """
    内部纯数值计算核心：
    - 优化1：降温操作 (np.exp) 提取到 sweep 层。
    - 优化2：利用 CPU 空间局部性进行顺序遍历，替代随机采样。
    """
    # 初始化全局最优记录
    global_best_state = np.zeros(N_vars, dtype=np.int8)
    global_best_energy = np.inf
    global_best_history = np.zeros(num_sweeps, dtype=np.float64)

    for read in range(num_reads):
        # 随机初始化状态 [-1, 1]
        state = np.ones(N_vars, dtype=np.int8)
        for i in range(N_vars):
            if np.random.random() < 0.5:
                state[i] = -1
        
        # 计算初始能量
        current_energy = 0.0
        for i in range(N_vars):
            current_energy += h_isi[i] * state[i]
            for j in range(N_vars):
                current_energy += 0.5 * J_isi[i, j] * state[i] * state[j]
        
        local_best_energy = current_energy
        local_best_state = state.copy()
        current_history = np.zeros(num_sweeps, dtype=np.float64)
        
        # ---------------- 核心优化区域 ----------------
        for sweep in range(num_sweeps):
            # 【优化1】指数降温提取到外层：每个 sweep 仅计算 1 次，而非 N_vars 次
            temp = t_max * np.exp(decay_rate * (sweep / num_sweeps))
            
            # 【优化2】顺序遍历：摒弃 randint，按内存连续地址顺序访问，极大提升缓存命中率
            for idx in range(N_vars):
                s_old = state[idx]
                
                # 计算局部点积用于能量差计算
                dot_J_state = 0.0
                for j in range(N_vars):
                    dot_J_state += J_isi[idx, j] * state[j]
                    
                delta_E = -2.0 * s_old * (h_isi[idx] + dot_J_state)
                
                # Metropolis 接受准则
                if delta_E < 0 or np.random.random() < np.exp(-delta_E / temp):
                    state[idx] = -s_old
                    current_energy += delta_E
                    # 随时记录局部最优
                    if current_energy < local_best_energy:
                        local_best_energy = current_energy
                        local_best_state = state.copy()
            
            # 记录本 sweep 结束后的最优能量
            current_history[sweep] = local_best_energy
        # ----------------------------------------------

        # 更新全局最优
        if local_best_energy < global_best_energy:
            global_best_energy = local_best_energy
            global_best_state = local_best_state.copy()
            global_best_history = current_history.copy()

    return global_best_state, global_best_energy, global_best_history


def custom_simulated_annealing(Q, num_reads=32, num_sweeps=3000, t_max=10.0, t_min=0.01):
    """
    外部接口保持完全不变。
    """
    variables = sorted(list(set(k for pair in Q.keys() for k in pair)))
    N_vars = len(variables)
    var_to_idx = {v: i for i, v in enumerate(variables)}
    
    # 构造 NumPy 矩阵
    h_isi = np.zeros(N_vars, dtype=np.float64)
    J_isi = np.zeros((N_vars, N_vars), dtype=np.float64)
    
    for (u, v), w in Q.items():
        u_idx, v_idx = var_to_idx[u], var_to_idx[v]
        if u_idx == v_idx:
            h_isi[u_idx] += w / 2.0
        else:
            val = w / 4.0
            J_isi[u_idx, v_idx] += val
            J_isi[v_idx, u_idx] += val
            h_isi[u_idx] += val
            h_isi[v_idx] += val

    decay_rate = np.log(t_min / t_max)

    # 移除 total_steps，直接传入 num_sweeps
    global_best_state, global_best_energy, global_best_history = _numba_sa_core(
        N_vars, num_reads, t_max, decay_rate, h_isi, J_isi, num_sweeps
    )
    
    final_sample = {variables[i]: int(global_best_state[i]) for i in range(N_vars)}
    return final_sample, global_best_energy, list(global_best_history)

# =========================
# 4. 显式 QUBO 构建与求解 (透传 history)
# =========================
def solve_steiner_path_explicit(U, u_star, distances, **kwargs):
    print(f"开始构建 QUBO...")
    N = len(U)
    node_to_idx = {node: i for i, node in enumerate(U)}
    idx_to_node = {i: node for i, node in enumerate(U)}
    E_var = lambda u, v: f"e_{u}_{v}"
    X_var = lambda u, v: f"x_{u}_{v}"
    Q = defaultdict(float)

    # 1. 目标函数: 距离
    for u_idx in range(N):
        for v_idx in range(N):
            if u_idx == v_idx: continue
            dist_val = distances.get((idx_to_node[u_idx], idx_to_node[v_idx]), 0.0)
            Q[(E_var(u_idx, v_idx), E_var(u_idx, v_idx))] += kwargs.get('w_obj', 1.0) * dist_val

    # 2. 流约束
    targets = {node_to_idx['S']: (1, 0), node_to_idx['E']: (0, 1)}
    for wp in u_star:
        if wp not in ['S', 'E']: targets[node_to_idx[wp]] = (1, 1)
            
    for idx, (t_out, t_in) in targets.items():
        w = kwargs.get('w_c1', 5.0) if idx in [node_to_idx['S'], node_to_idx['E']] else kwargs.get('w_c2', 10.0)
        out_vars = [E_var(idx, v) for v in range(N) if idx != v]
        for v in out_vars: Q[(v, v)] += w * (1 - 2 * t_out)
        for i in range(len(out_vars)):
            for j in range(i + 1, len(out_vars)): Q[tuple(sorted((out_vars[i], out_vars[j])))] += 2.0 * w
        in_vars = [E_var(u, idx) for u in range(N) if idx != u]
        for v in in_vars: Q[(v, v)] += w * (1 - 2 * t_in)
        for i in range(len(in_vars)):
            for j in range(i + 1, len(in_vars)): Q[tuple(sorted((in_vars[i], in_vars[j])))] += 2.0 * w

    # 3-5. 其它约束
    optional_indices = [i for i in range(N) if i not in targets]
    for idx in optional_indices:
        w = kwargs.get('w_c3', 15.0)
        out_v = [E_var(idx, v) for v in range(N) if idx != v]
        in_v = [E_var(u, idx) for u in range(N) if idx != u]
        for v in out_v + in_v: Q[(v, v)] += w
        for i in range(len(out_v)):
            for j in range(i + 1, len(out_v)): Q[tuple(sorted((out_v[i], out_v[j])))] += 3.0 * w
        for i in range(len(in_v)):
            for j in range(i + 1, len(in_v)): Q[tuple(sorted((in_v[i], in_v[j])))] += 3.0 * w
        for u_v in out_v:
            for v_v in in_v: Q[tuple(sorted((u_v, v_v)))] -= 2.0 * w

    for i in range(N):
        for j in range(i + 1, N): Q[tuple(sorted((E_var(i, j), E_var(j, i))))] += kwargs.get('w_bi', 10.0)

    w4 = kwargs.get('w_c4', 10.0)
    for i in range(N - 1):
        for j in range(i + 1, N):
            x, e_uv, e_vu = X_var(i, j), E_var(i, j), E_var(j, i)
            Q[(e_uv, e_uv)] += w4; Q[tuple(sorted((e_uv, x)))] -= w4; Q[tuple(sorted((e_vu, x)))] += w4
            
    for i in range(N - 2):
        for j in range(i + 1, N - 1):
            for k in range(j + 1, N):
                x_ij, x_jk, x_ik = X_var(i, j), X_var(j, k), X_var(i, k)
                Q[(x_ik, x_ik)] += w4; Q[tuple(sorted((x_ij, x_jk)))] += w4
                Q[tuple(sorted((x_ij, x_ik)))] -= w4; Q[tuple(sorted((x_jk, x_ik)))] -= w4

    # 求解，接收 history
    best_sample, best_energy, history = custom_simulated_annealing(
        Q, 
        num_reads=kwargs.get('num_reads', 32), 
        num_sweeps=kwargs.get('num_sweeps', 3000)
    )
    
    # 解析
    selected_edges, total_distance = [], 0.0
    for u_idx in range(N):
        for v_idx in range(N):
            if u_idx == v_idx: continue
            var = E_var(u_idx, v_idx)
            if best_sample.get(var, -1) == 1: 
                u, v = idx_to_node[u_idx], idx_to_node[v_idx]
                selected_edges.append((u, v))
                total_distance += distances.get((u, v), 0.0)
    return selected_edges, total_distance, best_energy, history

# =========================
# 5. 主程序 (增加绘图部分)
# =========================
def main():
    all_points, point_labels, label_to_coord, waypoints, optional_points, obstacles = create_complex_steiner_scene()
    
    U = point_labels
    u_star = ['S'] + [f'W{i+1}' for i in range(len(waypoints))] + ['E']
    distances = {}
    for u in U:
        for v in U:
            if u == v: continue
            distances[(u, v)] = calculate_edge_weight(label_to_coord[u], label_to_coord[v], obstacles)

    # 增加 num_sweeps 以适应更复杂的问题规模，并解包 history
    edges, dist_val, energy, history = solve_steiner_path_explicit(U, u_star, distances, num_reads=32, num_sweeps=3000)
    
    print(f"\n求解结果 (Ising Mode):\n总距离: {dist_val:.2f}\nIsing能量: {energy:.2f}")
    if edges:
        path_dict, curr, path_str, visited = {u:v for u, v in edges}, 'S', ['S'], {'S'}
        while curr != 'E' and curr in path_dict:
            curr = path_dict[curr]
            if curr in visited: break
            visited.add(curr); path_str.append(curr)
        print("路径顺序:", " -> ".join(path_str))
    
    # === 新增：能量收敛曲线可视化 ===
    plt.figure(figsize=(10, 5))
    plt.plot(history, label='Best Minimum Energy', color='blue', linewidth=1.5)
    plt.title('Simulated Annealing Convergence (Best Run)') 
    plt.xlabel('Sweeps')
    plt.ylabel('Minimum Energy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
    # ==============================

    visualize_solution(all_points, point_labels, label_to_coord, obstacles, edges, dist_val)
    plt.show()

if __name__ == "__main__": main()