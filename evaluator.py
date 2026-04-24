"""
评估单条路径及完整方案的成本，包含出发时间优化
新增功能：
- 启发式最优出发时间估计（用于插入决策）
- 燃油车限行弧违规检查
- 出发时间网格搜索 + 局部精化
- 车辆数量超限罚款计入总成本
"""
import numpy as np
from scipy.optimize import minimize_scalar
from config import WAIT_COST_PER_HOUR, TARDY_COST_PER_HOUR, VEHICLE_TYPES, RESTRICTED_END
from cost_calculator import calculate_trip

def heuristic_departure(route_custs, vtype, customers, dist):
    """
    快速估计一条路径的较优出发时间，用于插入决策。
    原则：让车辆尽可能晚出发，但又能避免迟到，并减少经过不同速度时段的偏差。
    使用道路距离和固定平均速度(35km/h)估算累积行驶时间。
    """
    if not route_custs:
        return 0.0
    avg_speed = 35.0   # 近似整体平均速度，误差比固定0.0小得多
    best_dep = 0.0
    cumul_dist = 0.0
    prev = 0
    for cust_id in route_custs:
        cumul_dist += dist[prev][cust_id]
        prev = cust_id
        ready_time = customers[cust_id-1]['ready_time']
        # 如果车辆立即出发，到达该客户的时间 = best_dep + 累积时间
        # 如果这个时间早于 ready_time，可以推迟出发来消除等待
        if best_dep + cumul_dist / avg_speed < ready_time:
            best_dep = max(best_dep, ready_time - cumul_dist / avg_speed)
    return max(0.0, best_dep)

def evaluate_route(route_custs, vtype, depart_time, customers, dist):
    """
    评估单条路径（不含始末0点），返回 (total_cost, detail_dict)
    增加：燃油车禁止在限行时段内从非绿区进入绿区（弧检查）
    """
    seq = [0] + route_custs + [0]
    t = depart_time
    cost = vtype.start_cost
    energy_cost = 0.0
    carbon_cost = 0.0
    penalty_cost = 0.0

    load = sum(customers[c-1]['demand_weight'] for c in route_custs)

    timeline = []
    prev_depart = depart_time   # 离开上一个节点的时刻，用于弧检查
    for i in range(len(seq)-1):
        cur = seq[i]
        nxt = seq[i+1]
        load_ratio = load / vtype.cap_weight if vtype.cap_weight > 0 else 0
        trip = calculate_trip(cur, nxt, t, vtype, load_ratio, dist)
        energy_cost += trip['energy_cost']
        carbon_cost += trip['carbon_cost']
        t = trip['arrive_time']

        if nxt != 0:
            cust = customers[nxt-1]
            # 获取对应车型的时间窗
            if vtype.fuel_type == 'fuel':
                ready, due = cust.get('tw_fuel', (cust['ready_time'], cust['due_time']))
            else:
                ready, due = cust.get('tw_elec', (cust['ready_time'], cust['due_time']))

            # 完全不可服务标记
            if ready < 0 and due < 0:
                return float('inf'), None

            # 燃油车弧违规检查：如果进入绿区客户，且离开上一节点时刻 < 16:00，则违规
            if vtype.fuel_type == 'fuel' and cust.get('is_green_zone', False):
                if prev_depart < RESTRICTED_END:
                    return float('inf'), None

            arr = t
            if t < ready:
                wait = ready - t
                penalty_cost += wait * WAIT_COST_PER_HOUR
                t = ready + cust['service_time']
                timeline.append((nxt, arr, ready))
            elif t > due:
                tardiness = t - due
                penalty_cost += tardiness * TARDY_COST_PER_HOUR
                t = t + cust['service_time']
                timeline.append((nxt, arr, t - cust['service_time']))
            else:
                t = t + cust['service_time']
                timeline.append((nxt, arr, t - cust['service_time']))
            load -= cust['demand_weight']
            prev_depart = t   # 离开该客户的时刻
        else:
            timeline.append((0, t, t))
            prev_depart = t

    total = cost + energy_cost + carbon_cost + penalty_cost
    return total, {
        'start_cost': cost,
        'energy_cost': energy_cost,
        'carbon_cost': carbon_cost,
        'penalty_cost': penalty_cost,
        'timeline': timeline,
        'departure_time': depart_time
    }

def optimize_departure_time(route_custs, vtype, customers, dist):
    """ 网格搜索 + 局部精化寻找最优出发时间 """
    def f(dep):
        c, _ = evaluate_route(route_custs, vtype, dep, customers, dist)
        return c if c < float('inf') else 1e12

    # 粗网格搜索，步长 0.5 小时
    best_dep = 0.0
    best_val = float('inf')
    for x in np.arange(0.0, 10.0 + 0.5, 0.5):
        val = f(x)
        if val < best_val:
            best_val = val
            best_dep = x
    # 局部精化，区间宽度 2 小时
    lb = max(0.0, best_dep - 1.0)
    ub = min(10.0, best_dep + 1.0)
    res = minimize_scalar(f, bounds=(lb, ub), method='bounded')
    best_cost, details = evaluate_route(route_custs, vtype, res.x, customers, dist)
    details['departure_time'] = res.x
    return best_cost, res.x, details

def evaluate_solution(routes, customers, dist):
    total = 0.0
    route_details = []
    type_count = {vtype.id: 0 for vtype in VEHICLE_TYPES}
    for r in routes:
        vtype = r['vtype']
        type_count[vtype.id] += 1
        custs = r['customers']
        w = sum(customers[c-1]['demand_weight'] for c in custs)
        v = sum(customers[c-1]['demand_volume'] for c in custs)
        if w > vtype.cap_weight + 1e-6 or v > vtype.cap_volume + 1e-6:
            return float('inf'), None

    # 车辆数量超限罚款
    penalty_excess = 0.0
    for vtype in VEHICLE_TYPES:
        if type_count[vtype.id] > vtype.count:
            excess = type_count[vtype.id] - vtype.count
            penalty_excess += excess * 100000.0

    for r in routes:
        vtype = r['vtype']
        custs = r['customers']
        cost, dep, det = optimize_departure_time(custs, vtype, customers, dist)
        total += cost
        det['vtype_name'] = vtype.name
        det['vtype'] = vtype
        det['customers'] = custs
        route_details.append(det)

    total += penalty_excess   # 计入罚款
    return total, route_details