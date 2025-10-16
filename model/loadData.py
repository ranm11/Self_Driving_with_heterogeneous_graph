import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler
from functools import partial
import argparse
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
import random

#### 
parser = argparse.ArgumentParser('Interface for  Training')
##### Optimizer - Scheduler
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--val_batch_size', type=int, default=128, help='batch size')
parser.add_argument('--n_epoch', type=int, default=30, help='number of epochs')
parser.add_argument('--warmup', type=float, default=1.0, help='the number of epoch for warmup')
parser.add_argument('--lr_decay_epoch', type=str, default="4-8-16-24-26", help='the index of epoch where the lr decays to lr*0.5')
parser.add_argument('--num_prediction', type=int,default=6, help='the number of modality')
parser.add_argument('--cls_weight', type=float,default=0.1, help='the weight of classification loss')
parser.add_argument('--reg_weight', type=float,default=50.0, help='the weight of regression loss')

#### Speed Up
parser.add_argument('--num_of_gnn_layer', type=int, default=6, help='the number of  layer')
parser.add_argument('--hidden_dim', type=int, default=256, help='init hidden dimension')
parser.add_argument('--head_dim', type=int, default=32, help='the dimension of attention head')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability')
parser.add_argument('--num_worker', type=int, default=8, help='number of worker per dataloader')

#### Setting
parser.add_argument('--agent_drop', type=float, default='0.0', help='the ratio of randomly dropping agent')
parser.add_argument('--data_folder', type=str,default="hdgt_waymo_dev_tmp", help='training set')

parser.add_argument('--refine_num', type=int, default=5, help='temporally refine the trajectory')
parser.add_argument('--output_vel', type=str, default="True", help='output in form of velocity') 
parser.add_argument('--cumsum_vel', type=str, default="True", help='cumulate velocity for reg loss')


#### Initialize
parser.add_argument('--checkpoint', type=str, default="none", help='load checkpoint')
parser.add_argument('--start_epoch', type=int, default=1, help='the index of start epoch (for resume training)')
parser.add_argument('--dev_mode', type=str, default="False", help='develop_mode')

parser.add_argument('--ddp_mode', type=str, default="False", help='False, True, multi_node')
parser.add_argument('--port', type=str, default="31243", help='DDP')

parser.add_argument('--amp', type=str, default="none", help='type of fp16')

#### Log
parser.add_argument('--val_every_train_step', type=int, default=-1, help='every number of training step to conduct one evaluation')
parser.add_argument('--name', type=str, default="hdgt_waymo_dev", help='the name of this setting')
args = parser.parse_args()
os.environ["DGLBACKEND"] = "pytorch"
####
map_size_lis = {1.0:30, 2.0:10, 3.0:20}

@torch.no_grad()
def SceneGraph_collate_fn(batch, args, max_range=80.0, lane_connect_thresh=6.0):
    """
    Minimal collate that builds a HeteroData scene graph per sample and batches them.
    Nodes:
      - vehicle: [x, y, cos(psi), sin(psi), speed, length, width]
      - pedestrian: [x, y, cos(psi), sin(psi), speed]
      - lane: [x, y, yaw, speed_limit]
      - traffic_light: [x, y, state0] (state placeholder if raw state unavailable)
      - stop_sign: [x, y]
      - ego: [x, y, cos(psi), sin(psi), speed] (position assumed (0,0) if not in sample)
    Edges:
      - (vehicle,pedestrian) fully-connected within max_range (undirected via both dirs)
      - (vehicle,lane) if within 12m
      - (lane,lane) sequential/adjacent via polyline neighbors if provided
      - (traffic_light,lane) per-lane association (one TL node per lane with signal)
      - (stop_sign,lane) per-lane association
      - (ego,vehicle) and (ego,lane) within ranges (30m, 20m)
    Output: dict with batched graph and convenience fields.
    """
    out_graphs = []
    vehicle_counts = []
    ped_counts = []

    for sample in batch:
        agent_fea = sample["agent_feature"]  # [N, T, F]
        agent_types = sample["agent_type"]   # [N]
        map_fea = sample["map_fea"]          # lanes+polygons as in HDGT

        data = HeteroData()

        # === Agents at last timestep ===
        last = agent_fea[:, -1, :]
        x = last[:, 0]; y = last[:, 1]
        vx = last[:, 3]; vy = last[:, 4]
        psi = last[:, 5]
        speed = np.sqrt(vx * vx + vy * vy)
        cospsi = np.cos(psi); sinpsi = np.sin(psi)

        # Vehicles (type==1)
        veh_mask = (agent_types == 1)
        veh_idx = np.where(veh_mask)[0]
        if veh_idx.size > 0:
            veh_feats = np.stack([
                x[veh_idx], y[veh_idx], cospsi[veh_idx], sinpsi[veh_idx],
                speed[veh_idx], last[veh_idx, 6], last[veh_idx, 7]
            ], axis=1).astype(np.float32)
            data['vehicle'].x = torch.as_tensor(veh_feats)
            data['vehicle'].pos = torch.as_tensor(np.stack([x[veh_idx], y[veh_idx]], axis=1).astype(np.float32))
        else:
            data['vehicle'].x = torch.zeros((0, 7), dtype=torch.float32)
            data['vehicle'].pos = torch.zeros((0, 2), dtype=torch.float32)
        vehicle_counts.append(int(veh_idx.size))

        # Pedestrians (type==2)
        ped_mask = (agent_types == 2)
        ped_idx = np.where(ped_mask)[0]
        if ped_idx.size > 0:
            ped_feats = np.stack([
                x[ped_idx], y[ped_idx], cospsi[ped_idx], sinpsi[ped_idx], speed[ped_idx]
            ], axis=1).astype(np.float32)
            data['pedestrian'].x = torch.as_tensor(ped_feats)
            data['pedestrian'].pos = torch.as_tensor(np.stack([x[ped_idx], y[ped_idx]], axis=1).astype(np.float32))
        else:
            data['pedestrian'].x = torch.zeros((0, 5), dtype=torch.float32)
            data['pedestrian'].pos = torch.zeros((0, 2), dtype=torch.float32)
        ped_counts.append(int(ped_idx.size))

        # Ego (assume ego at origin unless provided)
        ego_x, ego_y, ego_psi = 0.0, 0.0, 0.0
        ego_speed = 0.0
        data['ego'].x = torch.as_tensor([[ego_x, ego_y, np.cos(ego_psi), np.sin(ego_psi), ego_speed]], dtype=torch.float32)
        data['ego'].pos = torch.as_tensor([[ego_x, ego_y]], dtype=torch.float32)

        # === Lanes ===
        lanes = map_fea[0] if len(map_fea) > 0 else []
        lane_pos = []
        lane_feats = []
        lane_has_signal = []
        lane_has_stop = []
        for ln in lanes:
            if 'xyz' not in ln or len(ln['xyz']) < 3:
                continue
            # Use middle point as representative
            pt = np.asarray(ln['xyz'][2], dtype=float)
            lane_pos.append([pt[0], pt[1]])
            yaw = float(ln.get('yaw', 0.0))
            spd = float(ln.get('speed_limit', 0.0))
            lane_feats.append([pt[0], pt[1], yaw, spd])
            lane_has_signal.append(1 if (len(ln.get('signal', [])) > 0) else 0)
            lane_has_stop.append(1 if (len(ln.get('stop', [])) > 0) else 0)
        if len(lane_feats) == 0:
            data['lane'].x = torch.zeros((0, 4), dtype=torch.float32)
            data['lane'].pos = torch.zeros((0, 2), dtype=torch.float32)
        else:
            data['lane'].x = torch.as_tensor(np.asarray(lane_feats, dtype=np.float32))
            data['lane'].pos = torch.as_tensor(np.asarray(lane_pos, dtype=np.float32))

        # === Traffic lights (one per lane having signals) ===
        tl_pos = []
        tl_state = []  # placeholder state (0.0 if unknown)
        tl_lane_src = []  # edge: tl -> lane
        if len(lanes) > 0:
            for idx, ln in enumerate(lanes):
                sigs = ln.get('signal', [])
                if sigs and len(sigs) > 0:
                    try:
                        sig_arr = np.asarray(sigs[0])
                        if sig_arr.ndim >= 2 and sig_arr.shape[0] > 0:
                            sx, sy = float(sig_arr[:, 0].mean()), float(sig_arr[:, 1].mean())
                        else:
                            sx, sy = float(sig_arr[0]), float(sig_arr[1])
                    except Exception:
                        # fallback to lane position
                        sx, sy = lane_pos[idx] if idx < len(lane_pos) else (0.0, 0.0)
                    tl_pos.append([sx, sy])
                    tl_state.append([0.0])  # unknown state placeholder
                    tl_lane_src.append(idx)
        if len(tl_pos) == 0:
            data['traffic_light'].x = torch.zeros((0, 1), dtype=torch.float32)
            data['traffic_light'].pos = torch.zeros((0, 2), dtype=torch.float32)
            data[('traffic_light','tl2l','lane')].edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            data['traffic_light'].x = torch.as_tensor(np.asarray(tl_state, dtype=np.float32))
            data['traffic_light'].pos = torch.as_tensor(np.asarray(tl_pos, dtype=np.float32))
            tl_idx = np.arange(len(tl_pos), dtype=np.int64)
            data[('traffic_light','tl2l','lane')].edge_index = torch.as_tensor(np.stack([tl_idx, np.asarray(tl_lane_src, dtype=np.int64)], axis=0))

        # === Stop signs (one per lane having stops) ===
        ss_pos = []
        ss_lane_src = []
        if len(lanes) > 0:
            for idx, ln in enumerate(lanes):
                stops = ln.get('stop', [])
                if stops and len(stops) > 0:
                    try:
                        st_arr = np.asarray(stops[0])
                        if st_arr.ndim >= 2 and st_arr.shape[0] > 0:
                            sx, sy = float(st_arr[:, 0].mean()), float(st_arr[:, 1].mean())
                        else:
                            sx, sy = float(st_arr[0]), float(st_arr[1])
                    except Exception:
                        sx, sy = lane_pos[idx] if idx < len(lane_pos) else (0.0, 0.0)
                    ss_pos.append([sx, sy])
                    ss_lane_src.append(idx)
        if len(ss_pos) == 0:
            data['stop_sign'].x = torch.zeros((0, 1), dtype=torch.float32)
            data['stop_sign'].pos = torch.zeros((0, 2), dtype=torch.float32)
            data[('stop_sign','ss2l','lane')].edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            data['stop_sign'].x = torch.zeros((len(ss_pos), 1), dtype=torch.float32)
            data['stop_sign'].pos = torch.as_tensor(np.asarray(ss_pos, dtype=np.float32))
            ss_idx = np.arange(len(ss_pos), dtype=np.int64)
            data[('stop_sign','ss2l','lane')].edge_index = torch.as_tensor(np.stack([ss_idx, np.asarray(ss_lane_src, dtype=np.int64)], axis=0))

        # === Proximity-based edges ===
        def _build_prox_edges(P, Q, r, both_dirs=True):
            if P.shape[0] == 0 or Q.shape[0] == 0:
                return torch.zeros((2, 0), dtype=torch.long)
            src = []
            dst = []
            for i in range(P.shape[0]):
                dxdy = Q - P[i]
                d2 = (dxdy[:, 0]**2 + dxdy[:, 1]**2)
                idx = np.where(d2 <= (r * r))[0]
                if idx.size > 0:
                    src.extend([i] * int(idx.size))
                    dst.extend(idx.tolist())
            if len(src) == 0:
                return torch.zeros((2, 0), dtype=torch.long)
            ed = torch.as_tensor(np.stack([np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)], axis=0))
            if both_dirs and P.shape == Q.shape and P is not None and Q is not None:
                # add reverse to mimic undirected
                rev = torch.as_tensor(np.stack([ed[1].numpy(), ed[0].numpy()], axis=0))
                ed = torch.cat([ed, rev], dim=1)
            return ed

        # vehicle-vehicle within max_range
        data[('vehicle','v2v','vehicle')].edge_index = _build_prox_edges(
            data['vehicle'].pos.numpy(), data['vehicle'].pos.numpy(), r=float(max_range), both_dirs=True
        )
        # vehicle-lane within 12m
        data[('vehicle','v2l','lane')].edge_index = _build_prox_edges(
            data['vehicle'].pos.numpy(), data['lane'].pos.numpy() if 'lane' in data.node_types else np.zeros((0,2)), r=12.0, both_dirs=False
        )
        # ego-vehicle within 30m
        data[('ego','e2v','vehicle')].edge_index = _build_prox_edges(
            data['ego'].pos.numpy(), data['vehicle'].pos.numpy(), r=30.0, both_dirs=False
        )
        # ego-lane within 20m
        data[('ego','e2l','lane')].edge_index = _build_prox_edges(
            data['ego'].pos.numpy(), data['lane'].pos.numpy() if 'lane' in data.node_types else np.zeros((0,2)), r=20.0, both_dirs=False
        )

        out_graphs.append(data)

    # Batch graphs
    batched = Batch.from_data_list(out_graphs)

    output = {
        "graph": batched,
        "vehicle_counts": np.asarray(vehicle_counts, dtype=np.int32),
        "ped_counts": np.asarray(ped_counts, dtype=np.int32),
    }
    return output
def euclid_np(label, pred):
    return np.sqrt((label[...,0]-pred[...,0])**2 + (label[...,1]-pred[...,1])**2)

uv_dict = {}
## Sparse adj mat of fully connected graph of neighborhood size
def return_uv(neighborhood_size):
    global uv_dict
    if neighborhood_size in uv_dict:
        return uv_dict[neighborhood_size]
    else:
        v = torch.LongTensor([[_]*(neighborhood_size-1) for _ in range(neighborhood_size)]).view(-1)
        u = torch.LongTensor([list(range(0, _)) +list(range(_+1,neighborhood_size)) for _ in range(neighborhood_size)]).view(-1)
        uv_dict[neighborhood_size] = (u, v)
        return (u, v)

def return_rel_e_feature(src_ref_coor, dst_ref_coor, src_ref_psi, dst_ref_psi):
    rel_coor = src_ref_coor - dst_ref_coor
    if rel_coor.ndim == 0 or rel_coor.ndim == 1:
        rel_coor = np.atleast_1d(rel_coor)[np.newaxis, :]
    rel_coor = rotate(rel_coor, np.cos(-dst_ref_psi),  np.sin(-dst_ref_psi))
    rel_psi = np.atleast_1d(src_ref_psi - dst_ref_psi)[:, np.newaxis]
    rel_sin_theta = np.sin(rel_psi)
    rel_cos_theta = np.cos(rel_psi)
    return np.concatenate([rel_coor, rel_sin_theta, rel_cos_theta], axis=-1)

def rotate(data, cos_theta, sin_theta):
    data[..., 0], data[..., 1] = data[..., 0]*cos_theta - data[..., 1]*sin_theta, data[..., 1]*cos_theta + data[..., 0]*sin_theta
    return data

def generate_heterogeneous_graph(agent_fea, map_fea, agent_map_size_lis):
    """
    PyTorch Geometric version of generate_heterogeneous_graph
    
    Returns:
        data: HeteroData object
        graphindex2polylineindex: dict mapping graph lane indices to original polyline indices
        graphindex2polygonindex: dict mapping graph polygon indices to original polygon indices  
        boundary_type_dic: dict with lane boundary type information
    """
    max_in_edge_per_type = 32
    num_of_agent = agent_fea.shape[0]
    
    # Initialize HeteroData
    data = HeteroData()
    
    # === Agent-Agent Connections ===
    # Self-loop edges
    agent_self_src = list(range(num_of_agent))
    agent_self_dst = list(range(num_of_agent))
    
    # Other agent connections
    agent_other_src, agent_other_dst = [], []
    for agent_index_i in range(num_of_agent):
        final_dist_between_agent = euclid_np(agent_fea[agent_index_i, -1, :][np.newaxis, :2], agent_fea[:, -1, :2])
        nearby_agent_index = np.where(final_dist_between_agent < np.maximum(agent_map_size_lis[agent_index_i][np.newaxis], agent_map_size_lis))[0]
        nearby_agent_index = np.delete(nearby_agent_index, obj=np.where(nearby_agent_index == agent_index_i))
        
        if len(nearby_agent_index) > max_in_edge_per_type:
            final_dist_between_agent_sorted_nearby_index = np.argsort(final_dist_between_agent[nearby_agent_index])
            nearby_agent_index = nearby_agent_index[final_dist_between_agent_sorted_nearby_index][:max_in_edge_per_type]
        
        nearby_agent_index = nearby_agent_index.tolist()
        if len(nearby_agent_index) > 0:
            agent_other_src.extend([agent_index_i] * len(nearby_agent_index))
            agent_other_dst.extend(nearby_agent_index)
    
    # Set agent edge indices
    data[('agent', 'self', 'agent')].edge_index = torch.LongTensor([agent_self_src, agent_self_dst])
    if agent_other_src:
        data[('agent', 'other', 'agent')].edge_index = torch.LongTensor([agent_other_src, agent_other_dst])
    else:
        data[('agent', 'other', 'agent')].edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # === Polygon-Agent Connections ===
    polygon_index_cnt = 0
    graphindex2polygonindex = {}
    polygon_src, polygon_dst = [], []
    
    if len(map_fea[1]) > 0:
        dist_between_agent_polygon = np.stack([(euclid_np(agent_fea[:, -1, :][:, np.newaxis, :], _[1][np.newaxis, :, :]).min(1)) for _ in map_fea[1]], axis=-1)
        all_agent_nearby_polygon_index_lis = dist_between_agent_polygon < agent_map_size_lis[:, np.newaxis]
        
        for agent_index_i in range(num_of_agent):
            nearby_polygon_index_lis = np.where(all_agent_nearby_polygon_index_lis[agent_index_i, :])[0]
            if len(nearby_polygon_index_lis) > max_in_edge_per_type:
                current_dist_between_agent_polygon = dist_between_agent_polygon[agent_index_i, :]
                nearby_polygon_index_lis_sorted = np.argsort(current_dist_between_agent_polygon[nearby_polygon_index_lis])
                nearby_polygon_index_lis = nearby_polygon_index_lis[nearby_polygon_index_lis_sorted][:max_in_edge_per_type]
            
            nearby_polygon_index_lis = nearby_polygon_index_lis.tolist()
            for now_cnt, nearby_polygon_index in enumerate(nearby_polygon_index_lis):
                polygon_src.append(polygon_index_cnt)
                polygon_dst.append(agent_index_i)
                graphindex2polygonindex[polygon_index_cnt] = nearby_polygon_index
                polygon_index_cnt += 1
    
    if polygon_src:
        data[('polygon', 'g2a', 'agent')].edge_index = torch.LongTensor([polygon_src, polygon_dst])
    else:
        data[('polygon', 'g2a', 'agent')].edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # === Lane-Agent Connections ===
    laneindex2graphindex = {}
    graphindex_cnt = 0
    lane_to_agent_src, lane_to_agent_dst = [], []
    agent_to_lane_src, agent_to_lane_dst = [], []
    
    if len(map_fea[0]) > 0:
        all_polyline_coor = np.array([_["xyz"] for _ in map_fea[0]])
        final_dist_between_agent_lane = euclid_np(agent_fea[:, -1, :2][:, np.newaxis, np.newaxis, :], all_polyline_coor[np.newaxis, :, :, :]).min(2)
        all_agent_nearby_lane_index_lis = final_dist_between_agent_lane < agent_map_size_lis[:, np.newaxis]
        
        for agent_index_i in range(num_of_agent):
            nearby_road_index_lis = np.where(all_agent_nearby_lane_index_lis[agent_index_i, :])[0]
            if len(nearby_road_index_lis) > max_in_edge_per_type:
                current_dist_between_agent_lane = final_dist_between_agent_lane[agent_index_i]
                nearby_road_index_lis_sorted = np.argsort(current_dist_between_agent_lane[nearby_road_index_lis])
                nearby_road_index_lis = nearby_road_index_lis[nearby_road_index_lis_sorted][:max_in_edge_per_type]
            
            nearby_road_index_lis = nearby_road_index_lis.tolist()
            for now_cnt, nearby_road_index in enumerate(nearby_road_index_lis):
                if nearby_road_index not in laneindex2graphindex:
                    laneindex2graphindex[nearby_road_index] = graphindex_cnt
                    graphindex_cnt += 1
                
                agent_to_lane_src.append(agent_index_i)
                agent_to_lane_dst.append(laneindex2graphindex[nearby_road_index])
                lane_to_agent_src.append(laneindex2graphindex[nearby_road_index])
                lane_to_agent_dst.append(agent_index_i)
    
    if lane_to_agent_src:
        data[('lane', 'l2a', 'agent')].edge_index = torch.LongTensor([lane_to_agent_src, lane_to_agent_dst])
        data[('agent', 'a2l', 'lane')].edge_index = torch.LongTensor([agent_to_lane_src, agent_to_lane_dst])
    else:
        data[('lane', 'l2a', 'agent')].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data[('agent', 'a2l', 'lane')].edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # === Lane-Lane Connections ===
    lane2lane_boundary_dic = {}
    for etype in ["left", "right", "prev", "follow"]:
        lane_src, lane_dst = [], []
        lane2lane_boundary_dic[("lane", etype, "lane")] = []
        
        if len(map_fea[0]) > 0:
            all_in_graph_lane = list(laneindex2graphindex.keys())
            for in_graph_lane in all_in_graph_lane:
                info_dic = map_fea[0][in_graph_lane]
                neighbors = [_ for _ in info_dic[etype] if _[0] in laneindex2graphindex]
                lane2lane_boundary_dic[("lane", etype, "lane")].extend([_[1] for _ in neighbors])
                neighbors = [_[0] for _ in neighbors]
                lane_src.extend([laneindex2graphindex[in_graph_lane]] * len(neighbors))
                lane_dst.extend([laneindex2graphindex[_] for _ in neighbors])
        
        if lane_src:
            data[('lane', etype, 'lane')].edge_index = torch.LongTensor([lane_src, lane_dst])
        else:
            data[('lane', etype, 'lane')].edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # Set number of nodes for each node type (required by PyG)
    data['agent'].num_nodes = num_of_agent
    data['polygon'].num_nodes = polygon_index_cnt
    data['lane'].num_nodes = len(laneindex2graphindex)
    
    # Create return mappings
    graphindex2polylineindex = {v: k for k, v in laneindex2graphindex.items()}
    boundary_type_dic = {k: torch.LongTensor(v) for k, v in lane2lane_boundary_dic.items()}
    
    return data, graphindex2polylineindex, graphindex2polygonindex, boundary_type_dic


@torch.no_grad()
def HDGT_collate_fn(batch, setting_dic, args, is_train):
    # KEEP: We need raw agent features and types to extract vehicle poses (x, y, psi)
    # KEEP: We need lane signal features and indices to get traffic-light states
    # NOT NEEDED for this extraction: labels, masks, polygon features, lane edge attrs, normalization for training
    agent_drop = args.agent_drop

    # KEEP: agent_feature and agent_type contain positions/orientation per agent
    agent_feature_lis = [item["agent_feature"] for item in batch]
    agent_type_lis = [item["agent_type"] for item in batch]
    #agent_map_size_lis = [np.vectorize(setting_dic["agenttype2mapsize"].get)(_) for _ in agent_type_lis]
    pred_num_lis = np.array([item["pred_num"] for item in batch])
    label_lis = [item["label"] for item in batch]
    auxiliary_label_lis =  [item["auxiliary_label"] for item in batch]
    label_mask_lis = [item["label_mask"] for item in batch]
    other_label_lis = [item["other_label"] for item in batch]
    other_label_mask_lis = [item["other_label_mask"] for item in batch]
    # KEEP: map_fea holds lane-level signal info used by normal_lane_feature
    map_fea_lis = [item["map_fea"] for item in batch]
    case_id_lis = [item["scene_id"] for item in batch]
    object_id_lis = [item["obejct_id_lis"] for item in batch]

    # NOT NEEDED for extraction, but kept to preserve training behavior
    if agent_drop > 0 and is_train:
        for i in range(len(agent_feature_lis)):
            keep_index = (np.random.random(agent_feature_lis[i].shape[0]) > agent_drop)
            while keep_index[:pred_num_lis[i]].sum() == 0:
                keep_index = (np.random.random(agent_feature_lis[i].shape[0]) > agent_drop)
            origin_pred_num = pred_num_lis[i]
            original_agent_num = agent_feature_lis[i].shape[0]
            target_keep_index = keep_index[:origin_pred_num]
            agent_feature_lis[i] = agent_feature_lis[i][keep_index]
            agent_type_lis[i] = agent_type_lis[i][keep_index]
            pred_num_lis[i] = int(target_keep_index.sum())

            label_lis[i] = label_lis[i][target_keep_index]
            auxiliary_label_lis[i] = auxiliary_label_lis[i][target_keep_index]
            label_mask_lis[i] = label_mask_lis[i][target_keep_index]
            if origin_pred_num != original_agent_num:
                other_label_lis[i] = other_label_lis[i][keep_index[origin_pred_num:]]
                other_label_mask_lis[i] = other_label_mask_lis[i][keep_index[origin_pred_num:]]
    
    neighbor_size = np.array([int(agent_feature_lis[i].shape[0]) for i in range(len(agent_feature_lis))])

    out_lane_n_stop_sign_fea_lis = []
    out_lane_n_stop_sign_index_lis = []
    out_lane_n_signal_fea_lis = []  # KEEP: will serve as traffic light states
    out_lane_n_signal_index_lis = []  # KEEP: indices mapping for traffic light states

    # NOT NEEDED for extraction
    out_normal_lis = []
    out_graph_lis = []
    out_label_lis = []
    out_label_mask_lis = []
    out_auxiliary_label_lis = []
    out_auxiliary_label_future_lis = []
    out_other_label_lis = []
    out_other_label_mask_lis = []
    # KEEP: collect vehicle poses across scenes (x, y, psi)
    out_vehicle_pose_lis = []
    lane_n_cnt = 0

    for i in range(len(agent_feature_lis)):
        # NOT NEEDED for extraction beyond map crop sizing
        all_agent_obs_final_v = np.sqrt(agent_feature_lis[i][:, -1, 3]**2+agent_feature_lis[i][:, -1, 4]**2)
        all_agent_map_size = np.vectorize(map_size_lis.__getitem__)(agent_type_lis[i])
        all_agent_map_size = all_agent_obs_final_v * 8.0 + all_agent_map_size

        # Use PyG graph generation instead of DGL (NOT NEEDED for extraction itself)
        data, graphindex2polylineindex, graphindex2polygonindex, boundary_type_dic = generate_heterogeneous_graph(
            agent_feature_lis[i], map_fea_lis[i], all_agent_map_size
        )

        polylinelaneindex = list(graphindex2polylineindex.values())
        polygonlaneindex = list(graphindex2polygonindex.values())
        now_agent_feature = agent_feature_lis[i]
        now_agent_type = agent_type_lis[i]

        # KEEP: Extract vehicle poses (x, y, psi) for all vehicles in the scene
        # Vehicle type id is 1 per mapping in this file
        veh_idx = np.where(now_agent_type == 1)[0]
        if veh_idx.size > 0:
            veh_pose = np.zeros((veh_idx.size, 3), dtype=np.float32)
            # positions at last timestep
            veh_pose[:, :2] = now_agent_feature[veh_idx, -1, :2]
            # orientation (psi) at last timestep is at index 5
            veh_pose[:, 2] = now_agent_feature[veh_idx, -1, 5]
        else:
            veh_pose = np.zeros((0, 3), dtype=np.float32)
        out_vehicle_pose_lis.append(veh_pose)

        ### Type 0 edge a2a self-loop (NOT NEEDED for extraction)
        if data[('agent', 'self', 'agent')].edge_index.size(1) > 0:
            type0_u, type0_v = data[('agent', 'self', 'agent')].edge_index
            now_t0_v_feature = now_agent_feature[type0_v, :, :]
            now_t0_e_feature = now_agent_feature[type0_u].copy()
            if len(type0_v) == 1:
                now_t0_v_feature = now_t0_v_feature[np.newaxis, :, :]
                now_t0_e_feature = now_t0_e_feature[np.newaxis, :, :]
            now_t0_e_feature = return_rel_e_feature(now_t0_e_feature[:, -1, :3], now_t0_v_feature[:, -1, :3], now_t0_e_feature[:, -1, 5], now_t0_v_feature[:, -1, 5])
            data[('agent', 'self', 'agent')].edge_attr = torch.as_tensor(now_t0_e_feature.astype(np.float32))
            data[('agent', 'self', 'agent')].edge_type = torch.as_tensor(now_agent_type[type0_u].ravel().astype(np.int32)).long()
        
        ### Type 0 edge a2a other agent (NOT NEEDED for extraction)
        if data[('agent', 'other', 'agent')].edge_index.size(1) > 0:
            type1_u, type1_v = data[('agent', 'other', 'agent')].edge_index
            now_t1_v_feature = now_agent_feature[type1_v, :, :]
            now_t1_e_feature = now_agent_feature[type1_u].copy()
            if len(type1_v) == 1:
                now_t1_v_feature = now_t1_v_feature[np.newaxis, :, :]
                now_t1_e_feature = now_t1_e_feature[np.newaxis, :, :]
            now_t1_e_feature = return_rel_e_feature(now_t1_e_feature[:, -1, :3], now_t1_v_feature[:, -1, :3], now_t1_e_feature[:, -1, 5], now_t1_v_feature[:, -1, 5])
            data[('agent', 'other', 'agent')].edge_attr = torch.as_tensor(now_t1_e_feature.astype(np.float32))
            data[('agent', 'other', 'agent')].edge_type = torch.as_tensor(now_agent_type[type1_u].ravel().astype(np.int32)).long()
        else:
            data[('agent', 'other', 'agent')].edge_attr = torch.zeros((0, 5))
            data[('agent', 'other', 'agent')].edge_type = torch.zeros((0, )).long()

        ### Type 2 Edge: Agent -> Lane  a2l (NOT NEEDED for extraction)
        if len(polylinelaneindex) > 0 and data[('agent', 'a2l', 'lane')].edge_index.size(1) > 0:
            now_polyline_info = [map_fea_lis[i][0][_] for _ in polylinelaneindex]
            now_polyline_coor = np.stack([_["xyz"] for _ in now_polyline_info], axis=0)
            now_polyline_yaw = np.array([_["yaw"] for _ in now_polyline_info])
            now_polyline_type = np.array([_["type"] for _ in now_polyline_info])
            now_polyline_speed_limit = np.array([_["speed_limit"] for _ in now_polyline_info])
            now_polyline_stop = [_["stop"] for _ in now_polyline_info]
            now_polyline_signal = [_["signal"] for _ in now_polyline_info]
            now_polyline_mean_coor = now_polyline_coor[:, 2, :]
            type2_u, type2_v = data[('agent', 'a2l', 'lane')].edge_index
            now_t2_e_feature = now_agent_feature[type2_u].copy()
            if len(now_t2_e_feature.shape) == 2:
                now_t2_e_feature = now_t2_e_feature[np.newaxis, :, :]
            now_t2_e_feature = return_rel_e_feature(now_t2_e_feature[:, -1, :3], now_polyline_mean_coor[type2_v], now_t2_e_feature[:, -1, 5], now_polyline_yaw[type2_v])
            data[('agent', 'a2l', 'lane')].edge_attr = torch.as_tensor(now_t2_e_feature.astype(np.float32))
            data[('agent', 'a2l', 'lane')].edge_type = torch.as_tensor(now_agent_type[type2_u].ravel().astype(np.int32)).long()
               

        ### Type 3 Edge: Polygon -> Agent  g2a (NOT NEEDED for extraction)
        if data[('polygon', 'g2a', 'agent')].edge_index.size(1) > 0:
            type3_u, type3_v = data[('polygon', 'g2a', 'agent')].edge_index
            now_polygon_type = np.array([map_fea_lis[i][1][_][0] for _ in polygonlaneindex])
            now_polygon_coor = np.stack([map_fea_lis[i][1][_][1] for _ in polygonlaneindex], axis=0)
            now_t3_v_feature = now_agent_feature[type3_v]
            if len(now_t3_v_feature.shape) == 2:
                now_t3_v_feature = now_t3_v_feature[np.newaxis, :, :]
            ref_coor = now_t3_v_feature[:, -1, :3][:, np.newaxis, :]
            ref_psi = now_t3_v_feature[:, -1, 5][:, np.newaxis].copy()
            sin_theta = np.sin(-ref_psi)
            cos_theta = np.cos(-ref_psi)
            now_t3_e_coor_feature, now_t3_e_type_feature = normal_polygon_feature(now_polygon_coor, now_polygon_type, ref_coor, cos_theta, sin_theta)
            data[('polygon', 'g2a', 'agent')].edge_attr = torch.as_tensor(now_t3_e_coor_feature.astype(np.float32))
            data[('polygon', 'g2a', 'agent')].edge_type = torch.as_tensor(now_t3_e_type_feature.ravel().astype(np.int32)).long()

        ### Type 4 Edge: Lane -> Agent            
        if len(polylinelaneindex) > 0 and data[('lane', 'l2a', 'agent')].edge_index.size(1) > 0:
            type4_u, type4_v = data[('lane', 'l2a', 'agent')].edge_index
            now_t4_v_feature = now_agent_feature[type4_v]
            if len(now_t4_v_feature.shape) == 2:
                now_t4_v_feature = now_t4_v_feature[np.newaxis, :, :]
            now_t4_e_feature = return_rel_e_feature(now_polyline_mean_coor[type4_u], now_t4_v_feature[:, -1, :3], now_polyline_yaw[type4_u], now_t4_v_feature[:, -1, 5])
            data[('lane', 'l2a', 'agent')].edge_attr = torch.as_tensor(now_t4_e_feature.astype(np.float32))

        ### Type 5 Edge: Lane -> Lane (NOT NEEDED for extraction)
        if len(polylinelaneindex) > 0:
            for etype in ["left", "right", "prev", "follow"]:
                if data[('lane', etype, 'lane')].edge_index.size(1) > 0:
                    type5_u, type5_v = data[('lane', etype, 'lane')].edge_index
                    now_t5_e_feature = return_rel_e_feature(now_polyline_mean_coor[type5_u], now_polyline_mean_coor[type5_v], now_polyline_yaw[type5_u], now_polyline_yaw[type5_v])
                    data[('lane', etype, 'lane')].edge_attr = torch.as_tensor(now_t5_e_feature.astype(np.float32))

        now_pred_num = pred_num_lis[i]
        selected_pred_indices = list(range(0, now_pred_num))
        non_pred_indices = list(range(now_pred_num, now_agent_feature.shape[0]))
        
        ## Label + Full Agent Feature (NOT NEEDED for extraction)
        now_full_agent_n_feature = now_agent_feature[selected_pred_indices].copy()
        ref_coor = now_full_agent_n_feature[:, -1,:3].copy()
        now_label = label_lis[i][selected_pred_indices].copy()
        now_auxiliary_label = auxiliary_label_lis[i][selected_pred_indices].copy()
        now_label = now_label - ref_coor[:, np.newaxis, :2]
        ref_psi = now_full_agent_n_feature[:, -1, 5][:, np.newaxis].copy()
        normal_val = np.concatenate([ref_coor[..., :2], ref_psi], axis=-1)
        out_normal_lis.append(normal_val)
        
        sin_theta = np.sin(-ref_psi)
        cos_theta = np.cos(-ref_psi)
        rotate(now_label, cos_theta, sin_theta)
        rotate(now_auxiliary_label, cos_theta, sin_theta)
        now_auxiliary_label[..., 2] = now_auxiliary_label[..., 2] - ref_psi
        
        now_full_agent_n_feature = normal_agent_feature(now_full_agent_n_feature, ref_coor, ref_psi, cos_theta, sin_theta)
        now_auxiliary_label_future = now_auxiliary_label.copy()
        now_auxiliary_label = np.stack([now_full_agent_n_feature[..., 3],  now_full_agent_n_feature[..., 4], now_agent_feature[selected_pred_indices, :, 5]-ref_psi, now_full_agent_n_feature[..., -1]], axis=-1)


        now_all_agent_n_feature = now_full_agent_n_feature
        if now_pred_num < now_agent_feature.shape[0]:
            now_other_agent_n_feature = now_agent_feature[non_pred_indices].copy()
            ref_coor = now_other_agent_n_feature[:, -1, :3]
            ref_psi = now_other_agent_n_feature[:, -1, 5][:, np.newaxis].copy()
            sin_theta = np.sin(-ref_psi)
            cos_theta = np.cos(-ref_psi)
            now_other_agent_n_feature = normal_agent_feature(now_other_agent_n_feature, ref_coor, ref_psi, cos_theta, sin_theta)
            now_all_agent_n_feature = np.concatenate([now_all_agent_n_feature, now_other_agent_n_feature], axis=0)
        # Set node features in HeteroData (NOT NEEDED for extraction)
        data['agent'].x = torch.as_tensor(now_all_agent_n_feature.astype(np.float32))
        # Map agent types: Type 0=Other/Cyclist, 1=Vehicle, 2=Pedestrian, 3=Unknown/Other
        # Use agent types directly as they match the new mapping
        data['agent'].node_type = torch.as_tensor(now_agent_type.astype(np.int32)).long()
        
        ## Lane Node Feature (KEEP: for signal extraction downstream)
        if len(polylinelaneindex) > 0:
            ref_coor = now_polyline_mean_coor
            ref_psi = now_polyline_yaw[:, np.newaxis].copy()
            sin_theta = np.sin(-ref_psi)
            cos_theta = np.cos(-ref_psi)
            now_lane_n_coor_feature, now_lane_n_type_feature, now_lane_n_speed_limit_feature, now_lane_n_stop_feature, now_lane_n_stop_index, now_lane_n_signal_feature, now_lane_n_signal_index = normal_lane_feature(now_polyline_coor, now_polyline_type, now_polyline_speed_limit, now_polyline_stop, now_polyline_signal, list(range(len(now_polyline_coor))), ref_coor, cos_theta, sin_theta)                
            # Flatten lane coordinates to match expected format
            lane_feature_dim = now_lane_n_coor_feature.reshape(now_lane_n_coor_feature.shape[0], -1)
            data['lane'].x = torch.as_tensor(lane_feature_dim.astype(np.float32))
            data['lane'].node_type = torch.as_tensor(now_lane_n_type_feature.astype(np.int32)).long()
        else:
            # Create empty lane features if no lanes exist
            data['lane'].x = torch.zeros((0, 15), dtype=torch.float32)  # Will be adjusted based on actual lane feature dim
            data['lane'].node_type = torch.zeros((0,), dtype=torch.long)
        
        ## Polygon Node Feature  (NOT NEEDED for extraction)
        if len(polygonlaneindex) > 0:
            # Set basic polygon features (simplified)
            data['polygon'].x = torch.ones((len(polygonlaneindex), 1), dtype=torch.float32)  # Placeholder
            data['polygon'].node_type = torch.zeros((len(polygonlaneindex),), dtype=torch.long)  # Placeholder
        else:
            # Create empty polygon features if no polygons exist
            data['polygon'].x = torch.zeros((0, 1), dtype=torch.float32)
            data['polygon'].node_type = torch.zeros((0,), dtype=torch.long)

        ## Polyline Feature (KEEP: collect signal/stop features & indices)
        if len(polylinelaneindex) > 0:
            if len(now_lane_n_stop_index) != 0:
                out_lane_n_stop_sign_fea_lis.append(now_lane_n_stop_feature)
                out_lane_n_stop_sign_index_lis.append(np.array(now_lane_n_stop_index) + lane_n_cnt)
            if len(now_lane_n_signal_index) != 0:
                out_lane_n_signal_fea_lis.append(now_lane_n_signal_feature)
                out_lane_n_signal_index_lis.append(np.array(now_lane_n_signal_index)+lane_n_cnt)
            lane_n_cnt += now_lane_n_coor_feature.shape[0]

        out_graph_lis.append(data)
        out_label_lis.append(now_label)
        out_label_mask_lis.append(label_mask_lis[i][selected_pred_indices])
        out_auxiliary_label_lis.append(now_auxiliary_label)
        out_auxiliary_label_future_lis.append(now_auxiliary_label_future)

    output_dic = {}
    #0-x, 1-y, 2-vx, 3-vy, 4-cos_psi, 5-sin_psi, 6-length, 7-width, 8-type, 9-mask
    output_dic["cuda_tensor_lis"] = ["graph_lis"]
    output_dic["cuda_tensor_lis"] += ["label_lis", "label_mask_lis", "auxiliary_label_lis", "auxiliary_label_future_lis"]
    if len(out_lane_n_stop_sign_fea_lis) > 0:
        output_dic["cuda_tensor_lis"] += ["lane_n_stop_sign_fea_lis", "lane_n_stop_sign_index_lis"]
        out_lane_n_stop_sign_index_lis = np.concatenate(out_lane_n_stop_sign_index_lis, axis=0)
        output_dic["lane_n_stop_sign_fea_lis"] = torch.as_tensor(np.concatenate(out_lane_n_stop_sign_fea_lis, axis=0).astype(np.float32))
        output_dic["lane_n_stop_sign_index_lis"] =  torch.as_tensor(out_lane_n_stop_sign_index_lis.astype(np.int32)).long()

    if len(out_lane_n_signal_fea_lis) > 0:
        output_dic["cuda_tensor_lis"] += ["lane_n_signal_fea_lis", "lane_n_signal_index_lis"]
        out_lane_n_signal_index_lis = np.concatenate(out_lane_n_signal_index_lis, axis=0)
        output_dic["lane_n_signal_fea_lis"] = torch.as_tensor(np.concatenate(out_lane_n_signal_fea_lis, axis=0).astype(np.float32))
        output_dic["lane_n_signal_index_lis"] =  torch.as_tensor(out_lane_n_signal_index_lis.astype(np.int32)).long()
    output_dic["label_lis"] = torch.as_tensor(np.concatenate(out_label_lis, axis=0).astype(np.float32))
    output_dic["auxiliary_label_lis"] = torch.as_tensor(np.concatenate(out_auxiliary_label_lis, axis=0).astype(np.float32))
    output_dic["auxiliary_label_future_lis"] = torch.as_tensor(np.concatenate(out_auxiliary_label_future_lis, axis=0).astype(np.float32))

    output_dic["label_mask_lis"] = torch.as_tensor(np.concatenate(out_label_mask_lis, axis=0).astype(np.float32))

    # NOT NEEDED for extraction: edge attrs cause batching issues; kept for training batch consistency
    def remove_edge_attributes_for_batching(graph_list):
        """Remove edge attributes to avoid PyG batching inconsistencies"""
        for data in graph_list:
            for edge_type in data.edge_types:
                edge_store = data[edge_type]
                # Keep only edge_index, remove all other attributes
                if hasattr(edge_store, 'edge_index'):
                    edge_index = edge_store.edge_index
                    # Clear the edge store and only keep edge_index
                    edge_store.clear()
                    edge_store.edge_index = edge_index
    
    # Apply the fix - remove problematic edge attributes
    remove_edge_attributes_for_batching(out_graph_lis)
    
    # Batch graphs using PyG Batch instead of dgl.batch (NOT NEEDED for extraction alone)
    output_g = Batch.from_data_list(out_graph_lis)
    
    # Create agent type groupings for the batched graph (NOT NEEDED for extraction)
    all_agent_types = output_g['agent'].node_type
    a_n_type_lis = [torch.where(all_agent_types == _)[0] for _ in range(4)]
    
    # Create edge type groupings (NOT NEEDED for extraction)
    a_e_type_dict = {}
    for out_etype in ["self", "other"]:  # Simplified for core edge types
        a_e_type_dict[out_etype] = []
        if hasattr(output_g[('agent', out_etype, 'agent')], 'edge_type'):
            edge_types = output_g[('agent', out_etype, 'agent')].edge_type
            for agent_type_index in range(4):
                a_e_type_dict[out_etype].append(torch.where(edge_types == agent_type_index)[0])
        else:
            for agent_type_index in range(4):
                a_e_type_dict[out_etype].append(torch.tensor([], dtype=torch.long))
    output_dic["a_e_type_dict"] = a_e_type_dict
    output_dic["a_n_type_lis"] = a_n_type_lis
    output_dic["graph_lis"] = output_g
    output_dic["neighbor_size_lis"] = neighbor_size
    output_dic["pred_num_lis"] = pred_num_lis
    output_dic["case_id_lis"] = case_id_lis
    output_dic["object_id_lis"] = object_id_lis
    output_dic["normal_lis"] = np.concatenate(out_normal_lis, axis=0)

    # KEEP: Expose convenient outputs for your downstream needs
    # Vehicle poses (x, y, psi) across batch
    if len(out_vehicle_pose_lis) > 0:
        output_dic["vehicle_poses"] = torch.as_tensor(np.concatenate(out_vehicle_pose_lis, axis=0).astype(np.float32))
    else:
        output_dic["vehicle_poses"] = torch.zeros((0, 3), dtype=torch.float32)
    # Traffic light states and their lane indices (aliases for convenience)
    if "lane_n_signal_fea_lis" in output_dic:
        output_dic["traffic_light_states"] = output_dic["lane_n_signal_fea_lis"]
        output_dic["traffic_light_indices"] = output_dic["lane_n_signal_index_lis"]

    if "fname" in batch[0]:
        all_filename = [item["fname"] for item in batch]
        output_dic["fname"] = []
        for _ in range(len(all_filename)):
            output_dic["fname"] += [all_filename[_]]*pred_num_lis[_]
    del batch
    return output_dic

class HDGTDataset(Dataset):
    def __init__(self, dataset_path, data_folder, is_train, num_of_data, train_sample_batch_lookup):
        self.dataset_path = dataset_path
        self.data_folder = data_folder
        self.num_of_data = num_of_data
        self.is_train = is_train
        self.train_sample_batch_lookup = train_sample_batch_lookup ## To check which folder the sample is in
    def __getitem__(self, idx):
        for i in range(1, len(self.train_sample_batch_lookup)):
            if idx >= self.train_sample_batch_lookup[i-1]["cumulative_sample_cnt"] and idx < self.train_sample_batch_lookup[i]["cumulative_sample_cnt"]:
                batch_index = i-1
                break
        file_name =  os.path.join(self.dataset_path, self.train_sample_batch_lookup[batch_index+1]["data_folder"], self.data_folder+"_case"+str(idx-self.train_sample_batch_lookup[batch_index]["cumulative_sample_cnt"])+".pkl")
        with open(file_name, "rb") as f:
            sample = pickle.load(f)
        return sample

    def __len__(self):
        return self.num_of_data

## To make sure each batch has approximately the same number of node
class BalancedBatchSampler(BatchSampler):
    def __init__(self, input_size_lis, seed_num, gpu, gpu_cnt, batch_size, is_train):
        self.batch_size = batch_size
        input_size_lis = input_size_lis
        sorted_index = input_size_lis.argsort()[::-1].tolist()
        self.index_lis = []
        self.is_train = is_train
        for i in range(self.batch_size):
            self.index_lis.append(sorted_index[int(len(sorted_index)//self.batch_size * i):int(len(sorted_index)//self.batch_size * (i+1))])
        if len(sorted_index)//self.batch_size * self.batch_size < len(sorted_index):
            self.index_lis[-1] = self.index_lis[-1] + sorted_index[len(sorted_index)//self.batch_size * self.batch_size:]
        self.seed_num = seed_num
        self.gpu = gpu
        self.sample_per_gpu = len(self.index_lis[0])//gpu_cnt

    def __iter__(self):
        if self.is_train:
            for i in range(len(self.index_lis)):
                random.Random(self.seed_num+i).shuffle(self.index_lis[i])
        self.seed_num += 1
        for i in range(int(self.gpu*self.sample_per_gpu), int((self.gpu+1)*self.sample_per_gpu)):
            yield [self.index_lis[j][i] for j in range(self.batch_size)]
    def __len__(self):
        return self.sample_per_gpu

@torch.no_grad()
def obtain_dataset(gpu, gpu_count, seed_num, args):
    dataset_path = os.path.join(os.path.dirname(os.getcwd()),"Self_Driving_with_heterogeneous_graph" ,"dataset", "waymo")
    if args.dev_mode == "True":
        seed_num = 0
    print(gpu, seed_num, flush=True)

    data_folder = args.data_folder
    train_folder = "training"
    num_of_train_folder = 1
    val_folder = "validation"
    
    # Count all matching files in the train folder (recursively)
    # train_root = os.path.join(dataset_path, train_folder)
    # num_train_files = 0
    # for root, _, files in os.walk(train_root):
    #     num_train_files += sum(1 for f in files if f.startswith(f"{data_folder}_case") and f.endswith(".pkl"))
    # # Build a minimal lookup that covers the entire train folder
    # train_sample_batch_lookup = [
    #     {"cumulative_sample_cnt": 0},
    #     {"data_folder": train_folder, "cumulative_sample_cnt": num_train_files},
    # ]
     ## Initialize
    train_num_of_agent_arr = []
    train_sample_batch_lookup = [{"cumulative_sample_cnt":0}]
    for train_pacth_index in range(num_of_train_folder):
        with open(os.path.join(dataset_path, train_folder, data_folder+str(train_pacth_index), data_folder+"_number_of_case.pkl"), "rb") as f: 
            train_num_of_agent_arr.append(pickle.load(f))
        train_sample_batch_lookup.append({"cumulative_sample_cnt":train_sample_batch_lookup[-1]["cumulative_sample_cnt"]+train_num_of_agent_arr[-1].shape[0], "data_folder":os.path.join("training", data_folder+str(train_pacth_index))})
    train_num_of_agent_arr = np.concatenate(train_num_of_agent_arr, axis=0)
    
    val_num_of_agent_arr = []
    val_sample_batch_lookup = [{"cumulative_sample_cnt":0}]
    with open(os.path.join(dataset_path, val_folder, data_folder+str(num_of_train_folder), data_folder+"_number_of_case.pkl"), "rb") as f: 
        val_num_of_agent_arr = pickle.load(f)
    val_sample_batch_lookup.append({"cumulative_sample_cnt":val_num_of_agent_arr.shape[0], "data_folder":os.path.join("validation", data_folder+str(num_of_train_folder))})


    if args.dev_mode == "True":
        args.num_worker = 0
        dev_train_num = 2
        train_num_of_agent_arr = train_num_of_agent_arr[:dev_train_num]
        val_num_of_agent_arr = val_num_of_agent_arr[:dev_train_num]

    train_sampler = BalancedBatchSampler(train_num_of_agent_arr, seed_num=seed_num, gpu=gpu, gpu_cnt=gpu_count, batch_size=args.batch_size, is_train=True)
    if gpu == 0:
        val_sampler = BalancedBatchSampler(val_num_of_agent_arr, seed_num=seed_num, gpu=0, gpu_cnt=1, batch_size=args.val_batch_size, is_train=False)
        print("train sample num:", len(train_num_of_agent_arr), "val sample num:", len(val_num_of_agent_arr), flush=True)
    

    train_dataset = HDGTDataset(dataset_path=dataset_path, data_folder=args.data_folder, is_train=True,
        num_of_data=len(train_num_of_agent_arr)//gpu_count, train_sample_batch_lookup=train_sample_batch_lookup)
    setting_dic = {}
    train_dataloader =  DataLoader(train_dataset, pin_memory=True, collate_fn=partial(HDGT_collate_fn, setting_dic=setting_dic, args=args, is_train=True),batch_sampler=train_sampler,  num_workers=args.num_worker)
    train_sample_num = len(train_dataset) * gpu_count

    val_dataloader = None
    val_sample_num = 0
    if gpu == 0:
        val_worker_num = args.num_worker
        # if args.is_local == "multi_node" or args.is_local == "FalseM":
        #     val_worker_num *= 7
        # Count all matching files in the validation folder (recursively)
        val_root = os.path.join(dataset_path, val_folder)
        num_val_files = 0
        for root, _, files in os.walk(val_root):
            num_val_files += sum(1 for f in files if f.startswith(f"{data_folder}_case") and f.endswith(".pkl"))
        # Build a minimal lookup that covers the entire validation folder
        val_sample_batch_lookup = [
            {"cumulative_sample_cnt": 0},
            {"data_folder": val_folder, "cumulative_sample_cnt": num_val_files},
        ]
        val_dataset = HDGTDataset(dataset_path=dataset_path, data_folder=args.data_folder, is_train=False, num_of_data=num_val_files, train_sample_batch_lookup=val_sample_batch_lookup)
        val_dataloader = DataLoader(val_dataset, pin_memory=True, collate_fn=partial(HDGT_collate_fn, setting_dic=setting_dic, args=args, is_train=False), batch_sampler=val_sampler, num_workers=val_worker_num)
        val_sample_num = len(val_dataset)
    if gpu == 0:
        print('data loaded', flush=True)
    return train_dataloader, val_dataloader, train_sample_num, val_sample_num

def normal_polygon_feature(all_polygon_coor, all_polygon_type, ref_coor, cos_theta, sin_theta):
    now_polygon_coor = all_polygon_coor - ref_coor
    rotate(now_polygon_coor, cos_theta, sin_theta)
    return now_polygon_coor,  all_polygon_type

def normal_agent_feature(feature, ref_coor, ref_psi,  cos_theta, sin_theta):
    feature[..., :3] -= ref_coor[:, np.newaxis, :]
    feature[..., 0], feature[..., 1] = feature[..., 0]*cos_theta - feature[..., 1]*sin_theta, feature[..., 1]*cos_theta + feature[..., 0]*sin_theta
    feature[..., 3], feature[..., 4] = feature[..., 3]*cos_theta - feature[..., 4]*sin_theta, feature[..., 4]*cos_theta + feature[..., 3]*sin_theta
    feature[..., 5] -= ref_psi
    cos_psi = np.cos(feature[..., 5])
    sin_psi = np.sin(feature[..., 5])
    feature = np.concatenate([feature[..., :5], cos_psi[...,np.newaxis], sin_psi[...,np.newaxis], feature[..., 6:]], axis=-1)
    return feature

def normal_lane_feature(now_polyline_coor, now_polyline_type, now_polyline_speed_limit, now_polyline_stop, now_polyline_signal, polyline_index, ref_coor, cos_theta, sin_theta):
    output_polyline_coor = now_polyline_coor[polyline_index] - ref_coor[:, np.newaxis, :]
    rotate(output_polyline_coor, cos_theta, sin_theta)
    output_stop_fea = {i:np.array(now_polyline_stop[_][0]) for i, _ in enumerate(polyline_index) if len(now_polyline_stop[_]) != 0}
    output_signal_fea = {i:np.array(now_polyline_signal[_][0]) for i, _ in enumerate(polyline_index) if len(now_polyline_signal[_]) != 0}
    output_stop_index, output_stop_fea = list(output_stop_fea.keys()), list(output_stop_fea.values())

    if len(output_stop_fea) != 0:
        output_stop_fea = np.stack(output_stop_fea, axis=0)
        output_stop_fea -= ref_coor[output_stop_index]
        if type(cos_theta) == np.float64:
            rotate(output_stop_fea, cos_theta, sin_theta)
        else:
            rotate(output_stop_fea, cos_theta[output_stop_index].flatten(), sin_theta[output_stop_index].flatten())

    output_signal_index, output_signal_fea = list(output_signal_fea.keys()), list(output_signal_fea.values())
    if len(output_signal_fea) != 0:
        output_signal_fea = np.stack(output_signal_fea, axis=0)
        output_signal_fea[..., :3] -= ref_coor[output_signal_index]
        if type(cos_theta) == np.float64:
            rotate(output_signal_fea, cos_theta, sin_theta)
        else:
            rotate(output_signal_fea, cos_theta[output_signal_index].flatten(), sin_theta[output_signal_index].flatten())
    return output_polyline_coor, now_polyline_type[polyline_index], now_polyline_speed_limit[polyline_index], output_stop_fea, output_stop_index, output_signal_fea, output_signal_index

# from visualize_hetero_graph import visualize_batched_hetero_graph
if __name__ == '__main__':
    print(" Load Dataset")
    args = parser.parse_args()
    train_dataloader, val_dataloader, train_sample_num, val_sample_num = obtain_dataset(0,1, 2, args)
    # for batch_index, data in enumerate(train_dataloader, 0):
        # visualize_batched_hetero_graph(data, 0)
        
#         print(batch_index, data)
#         print(data.keys())
#         print(data["a_e_type_dict"].keys())
# # → dict_keys(['self', 'other'])
#         print(data["a_e_type_dict"]["self"])
#         for k, v in data.items():
#             if torch.is_tensor(v):
#                 print(f"{k}: tensor of shape {tuple(v.shape)}")
#             elif isinstance(v, list):
#                 print(f"{k}: list of length {len(v)}")
#             elif isinstance(v, dict):
#                 print(f"{k}: dict with keys {list(v.keys())}")
#             else:
#                 print(f"{k}: type {type(v)}")
    
    
#             break

    # print("load dataset done")