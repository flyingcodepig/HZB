"""
读取Excel文件，清洗数据，按客户汇总，拆分超大客户，构建距离矩阵，
并根据问题类型为绿色配送区生成紧缩时间窗。
"""
import pandas as pd
import numpy as np
from copy import deepcopy
import math
from config import (
    MAX_WEIGHT, MAX_VOLUME, SERVICE_TIME, START_EARLIEST,
    GREEN_ZONE_CENTER, GREEN_ZONE_RADIUS,
    RESTRICTED_END
)

def time_str_to_hour(t_str):
    parts = t_str.strip().split(':')
    h, m = int(parts[0]), int(parts[1])
    return h + m/60.0 - START_EARLIEST

def load_and_preprocess(order_path, dist_path, coord_path, tw_path,
                        problem_id=1, save_cleaned_path=None):
    # 1. 订单汇总
    df_order = pd.read_excel(order_path, sheet_name=0)
    df_order['重量'] = df_order['重量'].fillna(0)
    df_order['体积'] = df_order['体积'].fillna(0)
    demand_agg = df_order.groupby('目标客户编号').agg({'重量': 'sum', '体积': 'sum'})

    # 2. 距离矩阵
    df_dist = pd.read_excel(dist_path, sheet_name=0, index_col=0)
    df_dist.columns = [int(c) for c in df_dist.columns]
    df_dist.index = [int(i) for i in df_dist.index]
    dist_full = df_dist.values

    # 3. 时间窗
    df_tw = pd.read_excel(tw_path, sheet_name=0)
    tw_dict = {}
    for _, row in df_tw.iterrows():
        s = time_str_to_hour(row['开始时间'])
        e = time_str_to_hour(row['结束时间'])
        tw_dict[int(row['客户编号'])] = (s, e)
    tw_dict[0] = (0.0, 24.0)

    # 4. 坐标
    df_coord = pd.read_excel(coord_path, sheet_name=0)
    coord_dict = {int(row['ID']): (row['X (km)'], row['Y (km)']) for _, row in df_coord.iterrows()}

    # 5. 构建原始客户列表（跳过零需求客户）
    raw_customers = []
    for cid in range(1, 99):
        w = demand_agg.loc[cid, '重量'] if cid in demand_agg.index else 0
        v = demand_agg.loc[cid, '体积'] if cid in demand_agg.index else 0
        if w <= 0 and v <= 0:
            continue   # 跳过无需求客户
        tw = tw_dict.get(cid, (0.0, 24.0))
        raw_customers.append({
            'original_id': cid,
            'x': coord_dict[cid][0], 'y': coord_dict[cid][1],
            'demand_weight': w, 'demand_volume': v,
            'ready_time': tw[0], 'due_time': tw[1],
            'service_time': SERVICE_TIME
        })

    # 6. 按比例拆分超大客户
    customers = []
    for cust in raw_customers:
        w = cust['demand_weight']
        v = cust['demand_volume']
        k_w = math.ceil(w / MAX_WEIGHT) if w > 0 else 1
        k_v = math.ceil(v / MAX_VOLUME) if v > 0 else 1
        k = max(k_w, k_v)
        sub_w = w / k
        sub_v = v / k
        for i in range(1, k+1):
            new_cust = deepcopy(cust)
            new_cust['demand_weight'] = sub_w
            new_cust['demand_volume'] = sub_v
            new_cust['id'] = f"{cust['original_id']}_{i}"
            customers.append(new_cust)

    # 7. 分配整数 node_id，扩展距离矩阵
    N = len(customers) + 1
    dist_matrix = np.zeros((N, N))
    id_map = {0: 0}
    for idx, c in enumerate(customers, start=1):
        c['node_id'] = idx
        original = int(c['id'].split('_')[0]) if '_' in c['id'] else c['id']
        c['original_id'] = original
        id_map[idx] = original

    for i in range(N):
        for j in range(N):
            orig_i = id_map[i]
            orig_j = id_map[j]
            dist_matrix[i][j] = dist_full[orig_i][orig_j]

    # 构建最终客户列表，添加双时间窗和绿色区标识
    final_customers = []
    for c in customers:
        x, y = c['x'], c['y']
        dist_to_center = math.hypot(x - GREEN_ZONE_CENTER[0], y - GREEN_ZONE_CENTER[1])
        is_green = dist_to_center <= GREEN_ZONE_RADIUS
        orig_ready = c['ready_time']
        orig_due = c['due_time']

        tw_elec = (orig_ready, orig_due)
        if problem_id == 2 and is_green:
            fuel_ready = max(orig_ready, RESTRICTED_END)
            fuel_due = orig_due
            if fuel_ready <= fuel_due:
                tw_fuel = (fuel_ready, fuel_due)
            else:
                tw_fuel = (-1.0, -1.0)
        else:
            tw_fuel = (orig_ready, orig_due)

        final_customers.append({
            'id': c['node_id'],
            'original_id': c['original_id'],
            'x': x, 'y': y,
            'demand_weight': c['demand_weight'],
            'demand_volume': c['demand_volume'],
            'ready_time': orig_ready,
            'due_time': orig_due,
            'tw_fuel': tw_fuel,
            'tw_elec': tw_elec,
            'service_time': c['service_time'],
            'is_green_zone': is_green
        })

    # 8. 可选保存清洗数据
    if save_cleaned_path is not None:
        data_for_export = []
        for c in final_customers:
            data_for_export.append({
                'node_id': c['id'],
                'original_id': c['original_id'],
                'x': c['x'],
                'y': c['y'],
                'demand_weight_kg': c['demand_weight'],
                'demand_volume_m3': c['demand_volume'],
                'ready_time_rel8': c['ready_time'],
                'due_time_rel8': c['due_time'],
                'tw_fuel_ready': c['tw_fuel'][0],
                'tw_fuel_due': c['tw_fuel'][1],
                'tw_elec_ready': c['tw_elec'][0],
                'tw_elec_due': c['tw_elec'][1],
                'service_time_h': c['service_time'],
                'is_green_zone': c['is_green_zone']
            })
        data_for_export.append({
            'node_id': 0,
            'original_id': 0,
            'x': coord_dict[0][0],
            'y': coord_dict[0][1],
            'demand_weight_kg': 0.0,
            'demand_volume_m3': 0.0,
            'ready_time_rel8': 0.0,
            'due_time_rel8': 24.0,
            'tw_fuel_ready': 0.0,
            'tw_fuel_due': 24.0,
            'tw_elec_ready': 0.0,
            'tw_elec_due': 24.0,
            'service_time_h': 0.0,
            'is_green_zone': False
        })
        df_export = pd.DataFrame(data_for_export)
        df_export = df_export.sort_values('node_id').reset_index(drop=True)
        df_export.to_excel(save_cleaned_path, index=False, sheet_name='cleaned_data')
        print(f"清洗后的数据已保存至：{save_cleaned_path}")

    return final_customers, dist_matrix