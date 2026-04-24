"""
动态调度引擎：模拟车辆状态，并执行简单的路径调整策略
"""
import pandas as pd
import numpy as np
from copy import deepcopy
from evaluator import evaluate_solution
from operators import greedy_insert

def dynamic_reschedule(static_routes, t_event, event, customers, dist, problem_id=2):
    new_routes = deepcopy(static_routes)

    if event['event_type'] == 'cancel':
        cid = event['data']['customer_id']
        print(f"  [策略] 取消客户 {cid} 的订单。")
        for r in new_routes:
            if cid in r['customers']:
                r['customers'].remove(cid)
                print(f"      -> 从车辆路径中移除了客户 {cid}")
        new_routes = [r for r in new_routes if len(r['customers']) > 0]

    elif event['event_type'] == 'add':
        new_cust = None
        for c in customers:
            if c['original_id'] == event['data']['new_customer_id']:
                new_cust = c
                break
        if new_cust is None:
            print("  [策略] 新增客户未找到，无法插入。")
            return None, None
        removed = [new_cust['id']]
        print(f"  [策略] 新增客户 {new_cust['original_id']} (节点{new_cust['id']})，重量{new_cust['demand_weight']:.1f}kg，"
              f"时间窗 {new_cust['ready_time']:.2f}-{new_cust['due_time']:.2f}")
        new_routes = greedy_insert(new_routes, removed, customers, dist)
        inserted = any(new_cust['id'] in r['customers'] for r in new_routes)
        if inserted:
            for r in new_routes:
                if new_cust['id'] in r['customers']:
                    print(f"      -> 插入到车型 {r['vtype'].name} 的路径中")
        else:
            print("      -> 插入失败，可能无可用车辆或容量不足。")

    elif event['event_type'] == 'address_change':
        old_cid = event['data']['old_customer_id']
        new_cid = event['data']['new_customer_id']
        print(f"  [策略] 订单地址变更：原客户 {old_cid} -> 新客户 {new_cid}")
        for r in new_routes:
            if old_cid in r['customers']:
                r['customers'].remove(old_cid)
                print(f"      -> 从车辆路径中移除了原客户 {old_cid}")
        new_routes = [r for r in new_routes if len(r['customers']) > 0]
        new_cust = None
        for c in customers:
            if c['original_id'] == new_cid and c['demand_weight'] > 0:
                new_cust = c
                break
        if new_cust is not None:
            removed = [new_cust['id']]
            new_routes = greedy_insert(new_routes, removed, customers, dist)
            inserted = any(new_cust['id'] in r['customers'] for r in new_routes)
            if inserted:
                print(f"      -> 新客户 {new_cid} 已插入路径")
            else:
                print(f"      -> 新客户 {new_cid} 无法插入（可能无可用车辆）")
        else:
            print(f"      -> 新客户 {new_cid} 无需求，已忽略。")

    elif event['event_type'] == 'time_adjust':
        cid = event['data']['customer_id']
        ns = event['data']['new_start']
        ne = event['data']['new_end']
        print(f"  [策略] 客户 {cid} 时间窗调整为 {ns:.2f}-{ne:.2f} (无需改变路径)")

    new_cost, _, feasible = evaluate_solution(new_routes, customers, dist)
    if new_cost >= 1e12 or not feasible:
        print("  [警告] 调整后方案不可行，回退到原方案。")
        return static_routes, None
    return new_routes, new_cost