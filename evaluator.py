"""
评估单条路径及完整方案的成本，包含出发时间优化
"""
import numpy as np
from scipy.optimize import minimize_scalar
from config import WAIT_COST_PER_HOUR, TARDY_COST_PER_HOUR, VEHICLE_TYPES, RESTRICTED_END
from cost_calculator import calculate_trip

def heuristic_departure(route_custs, vtype, customers, dist):
    if not route_custs:
        return 0.0
    avg_speed = 35.0
    best_dep = 0.0
    cumul_dist = 0.0
    prev = 0
    for cust_id in route_custs:
        cumul_dist += dist[prev][cust_id]
        prev = cust_id
        ready_time = customers[cust_id-1]['ready_time']
        if best_dep + cumul_dist / avg_speed < ready_time:
            best_dep = max(best_dep, ready_time - cumul_dist / avg_speed)
    return max(0.0, best_dep)

def evaluate_route(route_custs, vtype, depart_time, customers, dist):
    seq = [0] + route_custs + [0]
    t = depart_time
    cost = vtype.start_cost
    energy_cost = 0.0
    carbon_cost = 0.0
    penalty_cost = 0.0

    load = sum(customers[c-1]['demand_weight'] for c in route_custs)

    timeline = []
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
            if vtype.fuel_type == 'fuel':
                ready, due = cust.get('tw_fuel', (cust['ready_time'], cust['due_time']))
            else:
                ready, due = cust.get('tw_elec', (cust['ready_time'], cust['due_time']))

            if ready < 0 and due < 0:
                return float('inf'), None

            # 绿区限行检查：燃油车到达绿区客户的时刻必须 >= 16:00
            if vtype.fuel_type == 'fuel' and cust.get('is_green_zone', False):
                if t < RESTRICTED_END:
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
        else:
            timeline.append((0, t, t))

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
    def f(dep):
        c, _ = evaluate_route(route_custs, vtype, dep, customers, dist)
        return c if c < float('inf') else 1e12

    best_dep = 0.0
    best_val = float('inf')
    for x in np.arange(0.0, 20.0 + 0.5, 0.5):   # 扩大到0~20小时
        val = f(x)
        if val < best_val:
            best_val = val
            best_dep = x
    lb = max(0.0, best_dep - 1.0)
    ub = min(20.0, best_dep + 1.0)
    res = minimize_scalar(f, bounds=(lb, ub), method='bounded')
    best_cost, details = evaluate_route(route_custs, vtype, res.x, customers, dist)
    if details is None:
        return float('inf'), None, None
    details['departure_time'] = res.x
    return best_cost, res.x, details

def evaluate_solution(routes, customers, dist):
    # 1. 客户覆盖完整性检查
    covered = set()
    for r in routes:
        covered.update(r['customers'])
    all_ids = set(c['id'] for c in customers)
    if covered != all_ids:
        return float('inf'), None

    # 2. 容量检查与车辆数统计
    type_count = {vtype.id: 0 for vtype in VEHICLE_TYPES}
    for r in routes:
        vtype = r['vtype']
        type_count[vtype.id] += 1
        custs = r['customers']
        w = sum(customers[c-1]['demand_weight'] for c in custs)
        v = sum(customers[c-1]['demand_volume'] for c in custs)
        if w > vtype.cap_weight + 1e-6 or v > vtype.cap_volume + 1e-6:
            return float('inf'), None

    penalty_excess = 0.0
    for vtype in VEHICLE_TYPES:
        if type_count[vtype.id] > vtype.count:
            excess = type_count[vtype.id] - vtype.count
            penalty_excess += excess * 100000.0

    total = 0.0
    route_details = []
    for r in routes:
        vtype = r['vtype']
        custs = r['customers']
        cost, dep, det = optimize_departure_time(custs, vtype, customers, dist)
        if det is None:                     # 不可行路径
            return float('inf'), None
        total += cost
        det['vtype_name'] = vtype.name
        det['vtype'] = vtype
        det['customers'] = custs
        route_details.append(det)

    total += penalty_excess
    return total, route_details