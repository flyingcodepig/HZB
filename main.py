"""
主程序：加载数据，运行 ALNS，输出调度方案与成本明细
支持命令行参数选择问题编号：python main.py [problem_id]
"""
import sys
import time
from data_loader import load_and_preprocess
from alns_solver import alns_solve
from evaluator import optimize_departure_time

def format_time(hours_after_8):
    total_min = int(round(hours_after_8 * 60))
    h = (8 + total_min // 60) % 24
    m = total_min % 60
    return f"{h:02d}:{m:02d}"

def print_solution(routes, customers, dist):
    total_start = 0
    total_energy = 0
    total_carbon = 0
    total_penalty = 0
    print("\n" + "="*80)
    print("最优车辆调度方案".center(80))
    print("="*80)
    for i, r in enumerate(routes):
        vtype = r['vtype']
        custs = r['customers']
        _, _, detail = optimize_departure_time(custs, vtype, customers, dist)
        total_start += detail['start_cost']
        total_energy += detail['energy_cost']
        total_carbon += detail['carbon_cost']
        total_penalty += detail['penalty_cost']

        print(f"\n车辆 {i+1}: {vtype.name} (载重{vtype.cap_weight}kg, 容积{vtype.cap_volume}m^3)")
        print(f"  出发时间: {format_time(detail['departure_time'])}")
        print(f"  路径: 配送中心 -> " + " -> ".join(str(c) for c in custs) + " -> 配送中心")
        print("  到达时刻:")
        for node, arr, start in detail['timeline']:
            if node == 0:
                print(f"    返回配送中心: {format_time(arr)}")
            else:
                cust = customers[node-1]
                print(f"    客户{node}: 到达{format_time(arr)}, 开始服务{format_time(start)} "
                      f"(时间窗 {format_time(cust['ready_time'])}-{format_time(cust['due_time'])})")
        w = sum(customers[c-1]['demand_weight'] for c in custs)
        vol = sum(customers[c-1]['demand_volume'] for c in custs)
        print(f"  装载: {w:.2f}kg / {vol:.3f}m³ (载重率 {w/vtype.cap_weight*100:.1f}%)")
        print(f"  成本分解: 启动{detail['start_cost']} + 能耗{detail['energy_cost']:.2f} "
              f"+ 碳排放{detail['carbon_cost']:.2f} + 时间窗{detail['penalty_cost']:.2f}")

    total = total_start + total_energy + total_carbon + total_penalty
    print("\n" + "-"*80)
    print(f"总成本: {total:,.2f} 元")
    print(f"  启动成本: {total_start:,.2f}")
    print(f"  能耗费用: {total_energy:,.2f}")
    print(f"  碳排放费: {total_carbon:,.2f}")
    print(f"  时间窗惩罚: {total_penalty:,.2f}")
    print(f"使用车辆数: {len(routes)}")
    print("="*80)

if __name__ == "__main__":
    # 读取命令行参数，默认问题1
    if len(sys.argv) > 1:
        problem_id = int(sys.argv[1])
    else:
        problem_id = 1

    ORDER_PATH = "order_information.xlsx"
    DIST_PATH = "distance_matrix.xlsx"
    COORD_PATH = "customer_coordinate_information.xlsx"
    TW_PATH = "time_window.xlsx"

    print(f"求解问题 {problem_id}...")
    customers, dist = load_and_preprocess(ORDER_PATH, DIST_PATH, COORD_PATH, TW_PATH,
                                          problem_id=problem_id,
                                          save_cleaned_path=f"cleaned_data_p{problem_id}.xlsx")
    print(f"客户数量: {len(customers)}")

    t0 = time.time()
    best_routes, best_cost = alns_solve(customers, dist,
                                        max_iter=1000,
                                        init_temp=200,
                                        cooling_rate=0.9993)
    t1 = time.time()
    print(f"\n求解耗时: {t1-t0:.1f} 秒")

    print_solution(best_routes, customers, dist)