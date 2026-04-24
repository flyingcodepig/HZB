"""
自适应大邻域搜索 (ALNS) 主求解器
"""
import random
import math
from copy import deepcopy
from config import VEHICLE_TYPES
from evaluator import evaluate_route, evaluate_solution, heuristic_departure
from operators import random_removal, worst_removal, greedy_insert, regret2_insert

OPERATORS_DESTROY = ['random_removal', 'worst_removal']
OPERATORS_REPAIR = ['greedy_insert', 'regret2_insert']
WEIGHTS = {op: 1.0 for op in OPERATORS_DESTROY + OPERATORS_REPAIR}
COUNTS = {op: 0 for op in OPERATORS_DESTROY + OPERATORS_REPAIR}

def select_operator(op_list):
    total = sum(WEIGHTS[op] for op in op_list)
    rnd = random.random() * total
    accum = 0.0
    for op in op_list:
        accum += WEIGHTS[op]
        if rnd <= accum:
            return op
    return op_list[-1]

def update_weights(destroy, repair, improvement):
    delta = 0.1 * improvement
    for op in [destroy, repair]:
        WEIGHTS[op] = max(0.1, WEIGHTS[op] + delta)
        COUNTS[op] += 1
    if sum(COUNTS.values()) % 100 == 0:
        for op in WEIGHTS:
            WEIGHTS[op] *= 0.9

def _can_serve(vtype, cust):
    if vtype.fuel_type == 'fuel':
        ready, due = cust.get('tw_fuel', (cust['ready_time'], cust['due_time']))
    else:
        ready, due = cust.get('tw_elec', (cust['ready_time'], cust['due_time']))
    return not (ready < 0 and due < 0)


def construct_initial_solution(customers, dist):
    available = {tid: vtype.count for tid, vtype in enumerate(VEHICLE_TYPES)}
    routes = []
    unassigned = set(range(1, len(customers)+1))

    # 识别必须电动客户
    must_electric = []
    for cid in unassigned.copy():
        cust = customers[cid-1]
        tf = cust.get('tw_fuel')
        if tf and tf[0] < 0 and tf[1] < 0:
            must_electric.append(cid)
            unassigned.remove(cid)

    electric_order = [3, 4]
    for cid in must_electric:
        inserted = False
        for tid in electric_order:
            if available[tid] == 0:
                continue
            vtype = VEHICLE_TYPES[tid]
            if customers[cid-1]['demand_weight'] <= vtype.cap_weight and \
               customers[cid-1]['demand_volume'] <= vtype.cap_volume:
                routes.append({'vtype': vtype, 'customers': [cid]})
                available[tid] -= 1
                inserted = True
                break
        if not inserted:
            best_insert = None
            best_cost = float('inf')
            for ri, r in enumerate(routes):
                vtype = r['vtype']
                if vtype.fuel_type != 'electric':
                    continue
                cust = customers[cid-1]
                w = sum(customers[c-1]['demand_weight'] for c in r['customers']) + cust['demand_weight']
                v = sum(customers[c-1]['demand_volume'] for c in r['customers']) + cust['demand_volume']
                if w > vtype.cap_weight or v > vtype.cap_volume:
                    continue
                for pos in range(len(r['customers'])+1):
                    new_route = r['customers'][:pos] + [cid] + r['customers'][pos:]
                    dep = heuristic_departure(new_route, vtype, customers, dist)
                    cost, _ = evaluate_route(new_route, vtype, dep, customers, dist)
                    if cost < best_cost:
                        best_cost = cost
                        best_insert = (ri, pos)
            if best_insert:
                ri, pos = best_insert
                routes[ri]['customers'].insert(pos, cid)

    # 常规贪心构造
    order = [3, 0, 1, 4, 2]
    while unassigned:
        best_route = None
        best_cost = float('inf')
        best_vtype = None
        for tid in order:
            if available[tid] <= 0:
                continue
            vtype = VEHICLE_TYPES[tid]
            route_custs = []
            remaining = set(unassigned)
            while remaining:
                best_insert = None
                best_insert_cost = float('inf')
                for c in remaining:
                    cust = customers[c-1]
                    if not _can_serve(vtype, cust):
                        continue
                    w = sum(customers[i-1]['demand_weight'] for i in route_custs) + cust['demand_weight']
                    vol = sum(customers[i-1]['demand_volume'] for i in route_custs) + cust['demand_volume']
                    if w > vtype.cap_weight or vol > vtype.cap_volume:
                        continue
                    temp_route = route_custs + [c]
                    dep = heuristic_departure(temp_route, vtype, customers, dist)
                    cost, _ = evaluate_route(temp_route, vtype, dep, customers, dist)
                    if cost < best_insert_cost:
                        best_insert_cost = cost
                        best_insert = c
                if best_insert is not None:
                    route_custs.append(best_insert)
                    remaining.remove(best_insert)
                else:
                    break
            if route_custs:
                dep = heuristic_departure(route_custs, vtype, customers, dist)
                cost, _ = evaluate_route(route_custs, vtype, dep, customers, dist)
                if cost < best_cost:
                    best_cost = cost
                    best_route = route_custs
                    best_vtype = vtype
        if best_route is None:
            break
        routes.append({'vtype': best_vtype, 'customers': best_route})
        unassigned -= set(best_route)
        available[best_vtype.id] -= 1

    if unassigned:
        print(f"警告: 贪心构造后有 {len(unassigned)} 个客户未分配，将交由 ALNS 尝试插入。")
    return routes


def alns_solve(customers, dist, max_iter=2000, init_temp=100, cooling_rate=0.9995,
               num_remove_min=1, num_remove_ratio=0.15):
    routes = construct_initial_solution(customers, dist)
    current_cost, _, cur_feasible = evaluate_solution(routes, customers, dist)
    best_routes = deepcopy(routes)
    best_cost = current_cost
    best_feasible = cur_feasible
    T = init_temp
    num_remove_max = max(1, int(len(customers)*num_remove_ratio))

    if not cur_feasible:
        print(f"初始解不合规 (成本: {current_cost:.2f})，将通过探索寻找合规解...")

    for it in range(max_iter):
        destroy_op = select_operator(OPERATORS_DESTROY)
        repair_op = select_operator(OPERATORS_REPAIR)
        num = random.randint(num_remove_min, min(num_remove_max, sum(len(r['customers']) for r in routes)))

        if destroy_op == 'random_removal':
            new_routes, removed = random_removal(deepcopy(routes), num)
        else:
            new_routes, removed = worst_removal(deepcopy(routes), num, customers, dist)
        if not removed:
            continue

        if repair_op == 'greedy_insert':
            new_routes = greedy_insert(new_routes, removed, customers, dist)
        else:
            new_routes = regret2_insert(new_routes, removed, customers, dist)

        new_cost, _, new_feasible = evaluate_solution(new_routes, customers, dist)
        improvement = current_cost - new_cost
        accept = False
        # 合规解总是优先接受
        if new_feasible and not cur_feasible:
            accept = True
        elif new_cost < current_cost:
            accept = True
        elif T > 0.01 and random.random() < math.exp(improvement / T):
            accept = True

        if accept:
            routes = new_routes
            current_cost = new_cost
            cur_feasible = new_feasible
            update_weights(destroy_op, repair_op, max(0, improvement))
            # 只有合规解才能成为全局最优
            if new_feasible and (not best_feasible or new_cost < best_cost):
                best_routes = deepcopy(new_routes)
                best_cost = new_cost
                best_feasible = True

        T *= cooling_rate
        if it % 10 == 0:
            print(f"  Iter {it}: curr={current_cost:.2f} (合规:{cur_feasible}), best={best_cost:.2f} (合规:{best_feasible}), T={T:.3f}")

    if not best_feasible:
        print("警告：未能找到完全合规的解，请增加迭代次数。")
    return best_routes, best_cost