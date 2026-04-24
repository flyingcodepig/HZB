"""
读取Excel文件，清洗数据，按客户汇总，拆分超大客户，构建距离矩阵，
并根据问题类型为绿色配送区生成紧缩时间窗。
新增 load_raw_data 函数供动态调度使用。
"""
import pandas as pd
import numpy as np
from copy import deepcopy
import math
from config import (
    MAX_WEIGHT, MAX_VOLUME, SERVICE_TIME, START_EARLIEST,
    GREEN_ZONE_CENTER, GREEN_ZONE_RADIUS, RESTRICTED_END
)

def time_str_to_hour(t_str):
    parts = t_str.strip().split(':')
    h, m = int(parts[0]), int(parts[1])
    return h + m/60.0 - START_EARLIEST

def load_raw_data(order_path, dist_path, coord_path, tw_path):
    """返回原始距离矩阵(DataFrame)、坐标字典、时间窗字典、订单DataFrame"""
    df_order = pd.read_excel(order_path, sheet_name=0)
    df_dist = pd.read_excel(dist_path, sheet_name=0, index_col=0)
    df_dist.columns = [int(c) for c in df_dist.columns]
    df_dist.index = [int(i) for i in df_dist.index]
    df_coord = pd.read_excel(coord_path, sheet_name=0)
    coord_dict = {int(row['ID']): (row['X (km)'], row['Y (km)']) for _, row in df_coord.iterrows()}
    df_tw = pd.read_excel(tw_path, sheet_name=0)
    tw_dict = {}
    for _, row in df_tw.iterrows():
        s = time_str_to_hour(row['开始时间'])
        e = time_str_to_hour(row['结束时间'])
        tw_dict[int(row['客户编号'])] = (s, e)
    tw_dict[0] = (0.0, 24.0)
    return df_order, df_dist, coord_dict, tw_dict

def build_customers_and_dist(orders_df, dist_full, coord_dict, tw_dict, problem_id=1):
    """根据订单表、原始距离矩阵、坐标、时间窗，构建客户列表和距离矩阵"""
    # 汇总订单
    orders = orders_df.copy()
    orders['重量'] = orders['重量'].fillna(0)
    orders['体积'] = orders['体积'].fillna(0)
    demand_agg = orders.groupby('目标客户编号').agg({'重量': 'sum', '体积': 'sum'})

    # 构建原始客户列表（排除零需求）
    raw_customers = []
    for cid in demand_agg.index:
        if cid == 0: continue
        w = demand_agg.loc[cid, '重量']
        v = demand_agg.loc[cid, '体积']
        if w <= 0 and v <= 0: continue
        if cid not in coord_dict: continue  # 新增客户坐标在外部添加
        tw = tw_dict.get(cid, (0.0, 24.0))
        raw_customers.append({
            'original_id': cid,
            'x': coord_dict[cid][0], 'y': coord_dict[cid][1],
            'demand_weight': w, 'demand_volume': v,
            'ready_time': tw[0], 'due_time': tw[1],
            'service_time': SERVICE_TIME
        })

    # 拆分超大客户
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

    # 分配 node_id，构建距离矩阵
    N = len(customers) + 1
    dist_matrix = np.zeros((N, N))
    id_map = {0: 0}
    for idx, c in enumerate(customers, start=1):
        c['node_id'] = idx
        original = int(c['id'].split('_')[0]) if '_' in c['id'] else c['id']
        c['original_id'] = original
        id_map[idx] = original

    # 原始距离矩阵大小（假设为 99x99，编号0-98）
    for i in range(N):
        for j in range(N):
            orig_i = id_map[i]
            orig_j = id_map[j]
            if orig_i < dist_full.shape[0] and orig_j < dist_full.shape[0]:
                dist_matrix[i][j] = dist_full.iloc[orig_i, orig_j]
            else:
                # 新增客户，用坐标直线距离*1.3 近似
                xi, yi = coord_dict.get(orig_i, (0,0)) if orig_i in coord_dict else customers[i-1]['x'], customers[i-1]['y']
                xj, yj = coord_dict.get(orig_j, (0,0)) if orig_j in coord_dict else customers[j-1]['x'], customers[j-1]['y']
                dist_matrix[i][j] = math.hypot(xi-xj, yi-yj) * 1.3

    # 构建最终客户列表，添加时间窗和绿色区标识
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
            'is_green_zone': is_green,
            'is_depot': False
        })
    return final_customers, dist_matrix

def load_and_preprocess(order_path, dist_path, coord_path, tw_path,
                        problem_id=1, save_cleaned_path=None, orders_df=None):
    # 如果外部传入了 orders_df，则不读文件
    if orders_df is None:
        df_order, df_dist_full, coord_dict, tw_dict = load_raw_data(order_path, dist_path, coord_path, tw_path)
    else:
        _, df_dist_full, coord_dict, tw_dict = load_raw_data(order_path, dist_path, coord_path, tw_path)
        df_order = orders_df.copy()
    customers, dist_matrix = build_customers_and_dist(df_order, df_dist_full, coord_dict, tw_dict, problem_id)
    # 可选保存
    if save_cleaned_path:
        # 保存逻辑（略，可保留原代码）
        pass
    return customers, dist_matrix