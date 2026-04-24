"""
随机突发事件生成器
"""
import random
import math

def generate_random_events(orders_df, customers, seed=42, num_events=4):
    """
    随机生成 num_events 个事件，返回事件列表。
    orders_df: 原始订单DataFrame (订单编号、重量、体积、目标客户编号)
    customers: 静态客户列表（包含坐标等信息）
    """
    rng = random.Random(seed)
    events = []
    # 事件发生时刻在 1.0 到 7.0 小时之间（9:00～15:00）
    times = sorted([rng.uniform(1.0, 7.0) for _ in range(num_events)])

    for t in times:
        event_type = rng.choice(['cancel', 'add', 'address_change', 'time_adjust'])
        if event_type == 'cancel':
            # 随机选一个客户取消其所有订单（移除需求量）
            cust_id = rng.choice(customers)['original_id']
            events.append({
                'event_type': 'cancel',
                'time': t,
                'data': {'customer_id': cust_id}
            })
        elif event_type == 'add':
            # 随机生成一个新客户（坐标在已知客户附近生成，重量不超过500kg）
            base_cust = rng.choice(customers)
            new_x = base_cust['x'] + rng.uniform(-2, 2)
            new_y = base_cust['y'] + rng.uniform(-2, 2)
            weight = rng.uniform(20, 300)
            volume = weight * rng.uniform(0.8, 1.2) / 1000.0  # 密度约0.8-1.2 t/m^3 转 m^3/kg 实际不合适，这里粗略估计
            # 简化：体积设为重量/500左右
            volume = max(0.01, weight / 500.0)
            # 时间窗设为事件后1～2小时开始，窗口1小时
            start_time = t + rng.uniform(1.0, 2.0)
            due_time = start_time + rng.uniform(0.5, 1.5)
            events.append({
                'event_type': 'add',
                'time': t,
                'data': {
                    'weight': weight,
                    'volume': volume,
                    'new_location': (new_x, new_y),
                    'ready_time': start_time,
                    'due_time': due_time
                }
            })
        elif event_type == 'address_change':
            # 随机选一个订单，将其目标客户改为另一现有客户
            if orders_df is not None and len(orders_df) > 0:
                order_idx = rng.randint(0, len(orders_df)-1)
                old_cust = int(orders_df.iloc[order_idx]['目标客户编号'])
                new_cust = rng.choice([c['original_id'] for c in customers if c['original_id'] != old_cust])
                events.append({
                    'event_type': 'address_change',
                    'time': t,
                    'data': {
                        'order_idx': order_idx,
                        'old_customer_id': old_cust,
                        'new_customer_id': new_cust
                    }
                })
        else:  # time_adjust
            cust = rng.choice(customers)
            # 时间窗平移 -0.5 ~ +0.5 小时
            shift = rng.uniform(-0.5, 0.5)
            new_start = max(0.0, cust['ready_time'] + shift)
            new_end = max(new_start + 0.5, cust['due_time'] + shift)
            events.append({
                'event_type': 'time_adjust',
                'time': t,
                'data': {
                    'customer_id': cust['original_id'],
                    'new_start': new_start,
                    'new_end': new_end
                }
            })
    return events