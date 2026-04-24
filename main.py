"""
主程序：加载数据，运行 ALNS，输出调度方案与成本明细
支持命令行参数选择问题编号：python main.py [problem_id]
"""
import sys
import time
import random
import numpy as np
import pandas as pd
import math
from data_loader import load_and_preprocess, load_raw_data, build_customers_and_dist
from alns_solver import alns_solve
from evaluator import optimize_departure_time, evaluate_solution
from event_generator import generate_random_events
from dynamic_scheduler import dynamic_reschedule
from config import RESTRICTED_END, GREEN_ZONE_CENTER, GREEN_ZONE_RADIUS

def format_time(hours_after_8):
    total_min = int(round(hours_after_8 * 60))
    h = (8 + total_min // 60) % 24
    m = total_min % 60
    return f"{h:02d}:{m:02d}"

def print_solution(routes, customers, dist):
    total_cost, details_list, feasible = evaluate_solution(routes, customers, dist)
    if not feasible:
        print("警告：当前方案不合规（存在车辆超限或客户漏派），成本仅供参考。")
    print("\n" + "=" * 80)
    print("最优车辆调度方案".center(80))
    print("=" * 80)
    for i, detail in enumerate(details_list):
        print(f"\n车辆 {i+1}: {detail['vtype_name']} (载重{detail['vtype'].cap_weight}kg, 容积{detail['vtype'].cap_volume}m³)")
        print(f"  出发时间: {format_time(detail['departure_time'])}")
        custs = detail['customers']
        print(f"  路径: 配送中心 -> " + " -> ".join(str(c) for c in custs) + " -> 配送中心")
        print("  到达时刻:")
        for node, arr, start in detail['timeline']:
            if node == 0:
                print(f"    返回配送中心: {format_time(arr)}")
            else:
                cust = next((c for c in customers if c['id'] == node), None)
                if cust:
                    print(f"    客户{node}: 到达{format_time(arr)}, 开始服务{format_time(start)} "
                          f"(时间窗 {format_time(cust['ready_time'])}-{format_time(cust['due_time'])})")
        w = sum(next(c for c in customers if c['id'] == x)['demand_weight'] for x in custs)
        vol = sum(next(c for c in customers if c['id'] == x)['demand_volume'] for x in custs)
        print(f"  装载: {w:.2f}kg / {vol:.3f}m³ (载重率 {w / detail['vtype'].cap_weight * 100:.1f}%)")
        print(f"  成本分解: 启动{detail['start_cost']} + 能耗{detail['energy_cost']:.2f} "
              f"+ 碳排放{detail['carbon_cost']:.2f} + 时间窗{detail['penalty_cost']:.2f}")

    total_start = sum(d['start_cost'] for d in details_list)
    total_energy = sum(d['energy_cost'] for d in details_list)
    total_carbon = sum(d['carbon_cost'] for d in details_list)
    total_penalty = sum(d['penalty_cost'] for d in details_list)
    print("\n" + "-" * 80)
    print(f"总成本: {total_cost:,.2f} 元")
    print(f"  启动成本: {total_start:,.2f}")
    print(f"  能耗费用: {total_energy:,.2f}")
    print(f"  碳排放费: {total_carbon:,.2f}")
    print(f"  时间窗惩罚: {total_penalty:,.2f}")
    print(f"使用车辆数: {len(routes)}")
    print("=" * 80)

def add_customer_to_data(customers, dist, new_cust_id, x, y, weight, volume, ready, due, service_time, problem_id):
    """在现有 customers/dist 上直接添加一个客户，返回更新后的 dist"""
    from config import RESTRICTED_END, GREEN_ZONE_CENTER, GREEN_ZONE_RADIUS
    dist_to_center = math.hypot(x - GREEN_ZONE_CENTER[0], y - GREEN_ZONE_CENTER[1])
    is_green = dist_to_center <= GREEN_ZONE_RADIUS
    tw_elec = (ready, due)
    if problem_id == 2 and is_green:
        fuel_ready = max(ready, RESTRICTED_END)
        fuel_due = due
        tw_fuel = (fuel_ready, fuel_due) if fuel_ready <= fuel_due else (-1.0, -1.0)
    else:
        tw_fuel = (ready, due)
    new_cust = {
        'id': new_cust_id,
        'original_id': new_cust_id,
        'x': x, 'y': y,
        'demand_weight': weight,
        'demand_volume': volume,
        'ready_time': ready,
        'due_time': due,
        'tw_fuel': tw_fuel,
        'tw_elec': tw_elec,
        'service_time': service_time,
        'is_green_zone': is_green,
        'is_depot': False
    }
    customers.append(new_cust)
    n = len(dist)
    new_dist = np.zeros((n+1, n+1))
    new_dist[:n, :n] = dist
    for i in range(n):
        if i == 0:
            xi, yi = 20, 20
        else:
            xi, yi = customers[i-1]['x'], customers[i-1]['y']
        d_est = math.hypot(xi - x, yi - y) * 1.3
        new_dist[i][n] = d_est
        new_dist[n][i] = d_est
    new_dist[n][n] = 0.0
    return new_cust, new_dist

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    if len(sys.argv) > 1:
        problem_id = int(sys.argv[1])
    else:
        problem_id = 1

    ORDER_PATH = "order_information.xlsx"
    DIST_PATH = "distance_matrix.xlsx"
    COORD_PATH = "customer_coordinate_information.xlsx"
    TW_PATH = "time_window.xlsx"

    print(f"求解问题 {problem_id}...")

    if problem_id in [1, 2]:
        customers, dist = load_and_preprocess(ORDER_PATH, DIST_PATH, COORD_PATH, TW_PATH,
                                              problem_id=problem_id)
        print(f"客户数量: {len(customers)}")
        t0 = time.time()
        best_routes, best_cost = alns_solve(customers, dist,
                                            max_iter=2000, init_temp=200, cooling_rate=0.9993)
        t1 = time.time()
        print(f"\n求解耗时: {t1 - t0:.1f} 秒")
        print_solution(best_routes, customers, dist)

    elif problem_id == 3:
        df_order_raw, df_dist_raw, coord_dict_raw, tw_dict_raw = load_raw_data(
            ORDER_PATH, DIST_PATH, COORD_PATH, TW_PATH)
        customers, dist = build_customers_and_dist(
            df_order_raw, df_dist_raw, coord_dict_raw, tw_dict_raw, problem_id=2)
        print(f"初始客户数量: {len(customers)}")

        print("求解静态问题2（初始调度）...")
        t0 = time.time()
        static_routes, _ = alns_solve(customers, dist, max_iter=2000,
                                      init_temp=200, cooling_rate=0.9993)
        for r in static_routes:
            _, dep, _ = optimize_departure_time(r['customers'], r['vtype'], customers, dist)
            r['depart_time'] = dep
        t1 = time.time()
        print(f"静态求解耗时: {t1 - t0:.1f} 秒")

        events = generate_random_events(df_order_raw, customers, seed=123, num_events=4)
        print(f"生成 {len(events)} 个随机事件")

        cur_custs = customers.copy()
        cur_dist = dist.copy()
        cur_routes = static_routes.copy()
        cur_orders = df_order_raw.copy()

        original_total_cost, _, _ = evaluate_solution(static_routes, customers, dist)
        current_total_cost = original_total_cost
        print(f"\n初始静态方案总成本: {original_total_cost:,.2f} 元")

        for i, evt in enumerate(events):
            print(f"\n事件{i+1}: {evt['event_type']} 在 {format_time(evt['time'])}")

            if evt['event_type'] == 'cancel':
                cid = evt['data']['customer_id']
                cur_orders = cur_orders[cur_orders['目标客户编号'] != cid]
                for cust in cur_custs:
                    if cust['original_id'] == cid:
                        cust['demand_weight'] = 0.0
                        cust['demand_volume'] = 0.0
            elif evt['event_type'] == 'add':
                loc = evt['data']['new_location']
                w = evt['data']['weight']
                v = evt['data']['volume']
                ready = evt['data']['ready_time']
                due = evt['data']['due_time']
                new_id = max([c['original_id'] for c in cur_custs] + [98]) + 1
                evt['data']['new_customer_id'] = new_id
                _, cur_dist = add_customer_to_data(
                    cur_custs, cur_dist, new_id, loc[0], loc[1],
                    w, v, ready, due, 20/60.0, problem_id=2)
            elif evt['event_type'] == 'address_change':
                idx = evt['data']['order_idx']
                cur_orders.at[idx, '目标客户编号'] = evt['data']['new_customer_id']
            elif evt['event_type'] == 'time_adjust':
                cid = evt['data']['customer_id']
                ns = evt['data']['new_start']
                ne = evt['data']['new_end']
                for cust in cur_custs:
                    if cust['original_id'] == cid:
                        cust['ready_time'] = ns
                        cust['due_time'] = ne
                        cust['tw_elec'] = (ns, ne)
                        if cust['is_green_zone'] and problem_id == 2:
                            f_ready = max(ns, RESTRICTED_END)
                            cust['tw_fuel'] = (f_ready, ne) if f_ready <= ne else (-1.0, -1.0)
                        else:
                            cust['tw_fuel'] = (ns, ne)

            new_routes, new_cost = dynamic_reschedule(
                cur_routes, evt['time'], evt,
                cur_custs, cur_dist, problem_id=2
            )

            if new_routes is not None and new_cost is not None:
                cur_routes = new_routes
                delta = new_cost - current_total_cost
                current_total_cost = new_cost
                print(f"  => 调整后总成本: {new_cost:,.2f} 元 (变化 {delta:+,.2f} 元)")
            else:
                print("  => 调整未改变方案或失败，维持原方案。")

        print(f"\n动态调度结束，最终总成本: {current_total_cost:,.2f} 元")