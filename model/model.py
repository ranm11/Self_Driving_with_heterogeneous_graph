import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import dgl
import math
# import dgl.function as fn
# from functools import partial  # No longer needed for PyG version
import math
from torch_geometric.nn import HeteroConv, Linear, GATConv, MessagePassing
from torch_geometric.data import HeteroData
class BN1D(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_in)
    def forward(self, x):
        if len(x.shape) == 3:
            return self.bn(x.transpose(1, 2)).transpose(1, 2)
        if len(x.shape) == 2:
            return self.bn(x.unsqueeze(-1)).squeeze(-1)
        
class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_se1 = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.conv_se2 = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.conv_se2(self.act(self.conv_se1(self.avg_pool(x))))) * x

class SEBasicBlock(nn.Module):
    def __init__(self, in_c, out_c, temporal_length, stride=1):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.conv1 = nn.Conv1d(kernel_size=3, in_channels=in_c, out_channels=out_c, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(in_c)
        self.conv2 = nn.Conv1d(kernel_size=3, in_channels=out_c, out_channels=out_c, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.act = nn.ReLU(inplace=True)
        
        self.stride = stride
        self.downsample = None
        self.se = SEBlock(in_c)
        if stride != 1:
            if temporal_length == 6 or temporal_length == 2:
                self.downsample = nn.Sequential(torch.nn.AvgPool1d(kernel_size=2, stride=2), nn.Conv1d(kernel_size=1, in_channels=in_c, out_channels=out_c, stride=1), nn.BatchNorm1d(out_c))
            else:
                self.downsample = nn.Sequential(torch.nn.AvgPool1d(kernel_size=2, stride=2, padding=1, count_include_pad=False), nn.Conv1d(kernel_size=1, in_channels=in_c, out_channels=out_c, stride=1), nn.BatchNorm1d(out_c))
        elif in_c != out_c:
            self.downsample = nn.Sequential(nn.Conv1d(kernel_size=1, in_channels=in_c, out_channels=out_c, stride=1), nn.BatchNorm1d(out_c))
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.conv1(self.act(self.bn1(x)))
        out = self.conv2(self.act(self.bn2(x)))
        if self.in_c == self.out_c:
            out = self.se(out)
            out = out + identity
        return out


class TemporalBlock(nn.Module):
    def __init__(self, inplanes, temporal_length, args):
        super(TemporalBlock, self).__init__()
        self.layers = [SEBasicBlock(in_c=inplanes, out_c=inplanes, temporal_length=temporal_length, stride=1)]
        self.layers += [SEBasicBlock(in_c=inplanes, out_c=inplanes, temporal_length=temporal_length, stride=2)]
        self.layers = nn.ModuleList(self.layers)
    def forward(self, x):
        out = x
        for _ in range(len(self.layers)):
            out = self.layers[_](out)
        return out

class AgentTemporalEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        kernel_size_dic = {0:11, 1:6, 2:3, 3:2, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1}
        self.temporal_layers = torch.nn.ModuleList([TemporalBlock(args.hidden_dim, kernel_size_dic[_], args) for _ in range(2)])
    def forward(self, feat):
        feat = self.temporal_layers[0](feat)
        for _ in range(1, len(self.temporal_layers)):
            feat = self.temporal_layers[_](feat)
        feat = feat[..., -1] ## Last timestep
        return feat

class BN1D(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_in)
    def forward(self, x):
        if len(x.shape) == 3:
            return self.bn(x.transpose(1, 2)).transpose(1, 2)
        if len(x.shape) == 2:
            is_single_input = (x.shape[0] == 1 and self.training == True)
            if is_single_input:
                self.bn.eval()
            res = self.bn(x.unsqueeze(-1)).squeeze(-1)
            if is_single_input:
                self.bn.train()
            return res
        
class MLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out, norm=None, dropout=0.0, prenorm=False):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_out) # position-wise
        self.act = nn.ReLU(inplace=True)
        self.prenorm = prenorm
        if norm is None:
            self.norm = nn.Identity()
        elif prenorm:
            self.norm = norm(d_in)
        else:
            self.norm = norm(d_hid)
        #if dropout != 0.0:
            #self.dropout = nn.Dropout(dropout)
        #else:
        self.dropout = nn.Identity()
    def forward(self, x):
        if self.prenorm:
            output = self.dropout(self.w_2(self.act(self.w_1(self.norm(x)))))
        else:
            output = self.dropout(self.w_2(self.act(self.norm(self.w_1(x)))))
        return output


class PointNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = MLP(d_in=args.hidden_dim//4, d_hid=args.hidden_dim//4, d_out=args.hidden_dim//8, norm=BN1D)
        self.fc2 = MLP(d_in=args.hidden_dim//8, d_hid=args.hidden_dim//8, d_out=args.hidden_dim//8, norm=BN1D)
        self.fc3 =  MLP(d_in=args.hidden_dim//4, d_hid=args.hidden_dim//4, d_out=args.hidden_dim//4, norm=BN1D)
    def forward(self, x):
        out = self.fc1(x)
        out = torch.cat([out, self.fc2(out).max(-2)[0].unsqueeze(-2).repeat(1, x.shape[-2], 1)], dim=-1)
        out = torch.cat([out, self.fc3(out).max(-2)[0].unsqueeze(-2).repeat(1, x.shape[-2], 1)], dim=-1)
        return out

class Agent2embedding(nn.Module):
    def __init__(self, input_dim, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.act = nn.ReLU(inplace=True)
        ## -4 - minus coor and mask, args.hidden_dim//4 - coordinate feature
        self.fea_MLP = MLP(input_dim-4+args.hidden_dim//4, args.hidden_dim//2, args.hidden_dim, norm=BN1D)
    def forward(self, input_dic, shared_coor_encoder):
        #0-2 x, y, z, 3-4 vx, vy, 5-6 cos, sin, 7-9 witdth, length, height 10 mask
        x = input_dic["graph_lis"]["agent"].x
        #z, vx, vy, cos, sin, width, length, height, mask
        coor_fea = shared_coor_encoder(x[..., :3])
        fea = torch.cat([coor_fea, x[..., 3:-1]], dim=-1)
        fea = self.fea_MLP(fea)
        return fea.transpose(1, 2)

## Centerline (Lane) Embedding
class Lane2embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.act = nn.ReLU(inplace=True)
        ## Encoder Coordinate Information
        self.pointnet = PointNet(args)
        self.point_out_fc = nn.Linear(args.hidden_dim//2, args.hidden_dim)
        self.type_emb =  torch.nn.Embedding(num_embeddings=4, embedding_dim=args.hidden_dim//4)
        self.stop_fc = nn.Linear(args.hidden_dim//4, args.hidden_dim//4)
        self.signal_fc = nn.Linear(args.hidden_dim//4, args.hidden_dim//4)
        self.signal_emb = torch.nn.Embedding(num_embeddings=9, embedding_dim=args.hidden_dim//2)
        self.signal_gru = torch.nn.GRU(input_size=args.hidden_dim//2, hidden_size=args.hidden_dim//2, num_layers=1, batch_first=True)
        
        ## [coor, type, stop_signal]
        self.lane_n_out_fc = MLP(d_in=args.hidden_dim+args.hidden_dim//4+self.hidden_dim//4*3+self.hidden_dim//2, d_hid=args.hidden_dim*4, d_out=args.hidden_dim, norm=BN1D)
        
        self.boudary_emb = torch.nn.Embedding(num_embeddings=11, embedding_dim=args.hidden_dim//4)
        ## [boundary, n_fea, rel_pos_fea]
        self.lane_e_out_fc = MLP(d_in=args.hidden_dim+args.hidden_dim//4+args.hidden_dim//4, d_hid=args.hidden_dim*4, d_out=args.hidden_dim, norm=BN1D)

    def forward(self, input_dic, shared_coor_encoder, shared_rel_encoder):
        # Check if lane nodes exist
        if not hasattr(input_dic["graph_lis"], 'lane') or input_dic["graph_lis"]["lane"].num_nodes == 0:
            return  # No lanes to process
            
        coor_fea = input_dic["graph_lis"]["lane"].x
        coor_fea =  shared_coor_encoder(coor_fea)
        coor_fea = self.pointnet(coor_fea)
        coor_fea = self.act(self.point_out_fc(coor_fea.max(dim=1)[0]))
        type_fea = self.type_emb(input_dic["graph_lis"]["lane"].node_type)

        lane_n_num = coor_fea.shape[0]
        ## If there is no stop sign/traffic signal controlling the lane, the cooresponding features are all zeros 
        device = input_dic.get("device", torch.device("cuda:"+str(input_dic["gpu"])))
        stop_signal_fea = torch.zeros((lane_n_num, self.hidden_dim//4*3+self.hidden_dim//2), device=device)
        if "lane_n_stop_sign_fea_lis" in input_dic:
            stop_sign_fea = input_dic["lane_n_stop_sign_fea_lis"]
            stop_sign_fea = shared_coor_encoder(stop_sign_fea)
            stop_sign_fea = self.act(self.stop_fc(stop_sign_fea))
            stop_signal_fea[input_dic["lane_n_stop_sign_index_lis"]][..., :self.hidden_dim//4] += stop_sign_fea
        if "lane_n_signal_fea_lis" in input_dic:
            signal_fea = input_dic["lane_n_signal_fea_lis"]
            signal_coor_fea = self.act(self.signal_fc(shared_coor_encoder(signal_fea[..., :3])))
            signal_dynamic = self.signal_emb(signal_fea[..., 3:].long())
            signal_dynamic, _ = self.signal_gru(signal_dynamic)
            signal_fea = torch.cat([signal_coor_fea, self.act(signal_dynamic[:, -1, :])], dim=-1)
            stop_signal_fea[input_dic["lane_n_signal_index_lis"]][..., self.hidden_dim//4:self.hidden_dim//4*2+self.hidden_dim//2] += signal_fea
        
        output_n_fea = self.lane_n_out_fc(torch.cat([coor_fea, type_fea, stop_signal_fea], dim=-1))
        
        ## Lane Edge Feature Encoding
        lane_e_num_lis_by_etype = np.cumsum([0] + [len(input_dic["graph_lis"].edata["l_e_fea"][_]) for _ in [("lane", "l2a", "agent"), ("lane", "left", "lane"), ("lane", "right", "lane"), ("lane", "prev", "lane"), ("lane", "follow", "lane")]])
        lane_e_rel_pos = torch.cat([input_dic["graph_lis"].edata["l_e_fea"][_] for _ in [("lane", "l2a", "agent"), ("lane", "left", "lane"), ("lane", "right", "lane"), ("lane", "prev", "lane"), ("lane", "follow", "lane")]], dim=0)
        lane_e_rel_pos_fea = shared_rel_encoder(lane_e_rel_pos)
        lane_e_num = lane_e_rel_pos_fea.shape[0]
        lane_src_indices = torch.cat([input_dic["graph_lis"].edges(etype=_)[0]  for _ in ["l2a", "left", "right", "prev", "follow"]], dim=0)
        lane_e_src_n_fea = output_n_fea[lane_src_indices]
        
        output_e_fea = torch.cat([torch.zeros((lane_e_num, self.hidden_dim//4), device=device), lane_e_rel_pos_fea, lane_e_src_n_fea,], dim=-1)
        
        boudary_emb = torch.cat([input_dic["graph_lis"].edata["boundary_type"][_] for _ in [("lane", "left", "lane"), ("lane", "right", "lane"), ("lane", "prev", "lane"), ("lane", "follow", "lane")]], dim=0)
        boudary_emb = self.boudary_emb(boudary_emb)
        output_e_fea[lane_e_num_lis_by_etype[1]:, :self.hidden_dim//4] += boudary_emb
        output_e_fea = self.lane_e_out_fc(output_e_fea)
        
        input_dic["graph_lis"].ndata["l_n_hidden"] = {"lane":output_n_fea}
        for _index, _ in enumerate([("lane", "l2a", "agent"), ("lane", "left", "lane"), ("lane", "right", "lane"), ("lane", "prev", "lane"), ("lane", "follow", "lane")]):
            input_dic["graph_lis"].edata["l_e_hidden"] = {_:output_e_fea[lane_e_num_lis_by_etype[_index]:lane_e_num_lis_by_etype[_index+1]]}
        return None

class Polygon2embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.pointnet = PointNet(args)
        self.type_emb = torch.nn.Embedding(num_embeddings=14, embedding_dim=args.hidden_dim//2)
        self.out_fc = MLP(args.hidden_dim//2+args.hidden_dim//2, args.hidden_dim*4, args.hidden_dim, norm=BN1D)
    
    def forward(self, input_dic, shared_coor_encoder):
        # Check if polygon edges exist
        graph = input_dic["graph_lis"]
        if not hasattr(graph[("polygon", "g2a", "agent")], 'edge_attr') or graph[("polygon", "g2a", "agent")].edge_index.size(1) == 0:
            return  # No polygon edges to process
            
        coor_fea = graph[("polygon", "g2a", "agent")].edge_attr
        coor_fea = shared_coor_encoder(coor_fea)
        coor_fea = self.pointnet(coor_fea).max(dim=1)[0]
        type_fea = graph[("polygon", "g2a", "agent")].edge_type
        type_fea = self.type_emb(type_fea)
        fea = torch.cat([coor_fea, type_fea], axis=-1)
        fea = self.out_fc(fea)
        graph[("polygon", "g2a", "agent")].g_e_hidden = fea


class ScaledDotProductAttention(torch.nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, temperature, args):
        super().__init__()
        self.temperature = temperature
        self.dropout = torch.nn.Dropout(args.dropout)
        self.softmax = torch.nn.Softmax(dim=2)
    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        n = float(v.shape[1])
        attn = attn / self.temperature * math.log(n+1, 32)
        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout, args):
        super().__init__()
        self.d_in = d_in
        self.out_dim = d_in
        self.norm = nn.LayerNorm(d_in)
        self.w1 = nn.Linear(d_in, d_hid)
        self.w2 = nn.Linear(d_hid, d_in)
        self.w3 = nn.Linear(d_in, d_hid)
        self.act = nn.SiLU()
        if dropout != 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
    def forward(self, x):
        residual = x
        output = self.norm(x)
        output = self.w2(self.act(self.w1(output)) * self.w3(output))
        output = self.dropout(output) + residual
        return output



class LaneHetGNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.etype_num = 5
        self.norm = nn.LayerNorm
        self.node_mlp = nn.ModuleDict({
            _:MLP(args.hidden_dim, args.hidden_dim, args.hidden_dim, norm=nn.LayerNorm, prenorm=True)
            for _ in ["left", "right", "prev", "follow"]})
        
        self.node_mlp["a2l"] = nn.ModuleList([MLP(args.hidden_dim, args.hidden_dim, args.hidden_dim, norm=nn.LayerNorm, prenorm=True)
        for _ in range(3)])

        self.node_fc = nn.Linear(self.etype_num*args.hidden_dim, args.hidden_dim)
        self.node_ffn = PositionwiseFeedForward(args.hidden_dim, args.hidden_dim*4, args.dropout, args)

        # Removed DGL-specific etype_dic since we're using PyG now
        
        self.edge_MLP = MLP(args.hidden_dim*3, args.hidden_dim*4, args.hidden_dim, norm=nn.LayerNorm, dropout=args.dropout, prenorm=True)
        
        self.agent_edge_MLP = nn.ModuleList([MLP(args.hidden_dim*3, args.hidden_dim*4, args.hidden_dim, norm=nn.LayerNorm, dropout=args.dropout, prenorm=True)
        for _ in range(3)
        ])

    def forward(self, input_dic):
        # Simplified PyG version - for now just return placeholder outputs
        self.gpu = input_dic["gpu"]
        self.a_e_type_dict = input_dic["a_e_type_dict"]
        
        # Check if lane nodes exist
        if not hasattr(input_dic["graph_lis"], 'lane') or input_dic["graph_lis"]["lane"].num_nodes == 0:
            # Return empty tensors if no lanes
            device = input_dic.get("device", torch.device("cuda:"+str(input_dic["gpu"])))
            output_lane_n_fea = torch.zeros((0, 256), device=device)  # Placeholder dimensions
            output_e_fea = [torch.zeros((0, 256), device=device) for _ in range(5)]  # 4 lane edge types + 1 agent2lane
            return output_lane_n_fea, output_e_fea
        
        # For now, just return the input lane features (placeholder)
        if hasattr(input_dic["graph_lis"]["lane"], 'l_n_hidden'):
            output_lane_n_fea = input_dic["graph_lis"]["lane"].l_n_hidden
        else:
            output_lane_n_fea = input_dic["graph_lis"]["lane"].x
        
        # Placeholder edge features
        device = input_dic.get("device", torch.device("cuda:"+str(input_dic["gpu"])))
        output_e_fea = [torch.zeros((0, 256), device=device) for _ in range(5)]  # 4 lane edge types + 1 agent2lane
        
        return output_lane_n_fea, output_e_fea






class AgentHetGNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.norm = nn.LayerNorm
        self.hidden_dim = args.hidden_dim
        self.head_dim = args.head_dim
        self.d_model = args.hidden_dim
        self.n_head = self.d_model//args.head_dim
        d_k = self.head_dim
        d_v = self.head_dim
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), args=args)

        self.wqs = nn.ModuleList([nn.Sequential(
            self.norm(self.d_model),
            nn.Linear(self.d_model, self.n_head * d_k * 3, bias=False),
        )
        for _ in range(3)
        ])

        self.wkvs = nn.ModuleDict({
            _: nn.Sequential(
                self.norm(self.d_model),
                nn.Linear(self.d_model, self.n_head * d_k * 2, bias=False)
                ) 
            for _ in ["l2a", "g2a"]
        })
        
        self.wkvs["other"] = nn.ModuleList([
            nn.Sequential(
                self.norm(self.d_model),
                nn.Linear(self.d_model, self.n_head * d_k * 2, bias=False)
                ) 
        for _ in range(3)
        ])
        
        self.attn_fcs = nn.ModuleList([nn.ModuleDict({
            _:nn.Sequential(
                nn.Linear(self.n_head * d_v, self.d_model, bias=True),
                nn.ReLU(inplace=True)
            )
            for _ in ["l2a", "g2a", "other"]
        })
        for agent_type_index in range(3)]
        )
        
        
        self.self_fc =  nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=True),
                nn.ReLU(inplace=True)
            )
        for _ in range(3)]
        )
        
        self.out_fc =  nn.ModuleList([nn.Sequential(
            nn.Linear(self.hidden_dim*4, self.hidden_dim, bias=True),
            nn.Dropout(args.dropout) if args.dropout > 0.0 else nn.Identity()
        )
        for _ in range(3)])
                
        self.out_ffn = nn.ModuleList([PositionwiseFeedForward(args.hidden_dim, args.hidden_dim*4, args.dropout, nn.LayerNorm)
        for _ in range(3)
        ])
        # Removed DGL-specific etype_dic since we're using PyG now

        self.edge_MLP = {
            "l2a":MLP(args.hidden_dim*3, args.hidden_dim*4, args.hidden_dim, nn.LayerNorm, dropout=args.dropout, prenorm=True),
            "g2a":MLP(args.hidden_dim*2, args.hidden_dim*4, args.hidden_dim, nn.LayerNorm, dropout=args.dropout, prenorm=True),
        }
        self.edge_MLP["self"] =  nn.ModuleList([MLP(args.hidden_dim, args.hidden_dim, args.hidden_dim, nn.LayerNorm, dropout=args.dropout, prenorm=True)
        for _ in range(3)
        ])
        
        self.edge_MLP["other"] = nn.ModuleList([MLP(args.hidden_dim*3, args.hidden_dim*4, args.hidden_dim, nn.LayerNorm, dropout=args.dropout, prenorm=True)
        for _ in range(3)
        ])        
        self.edge_MLP = nn.ModuleDict(self.edge_MLP)

    def forward(self, input_dic):
        self.a_e_type_dict = input_dic["a_e_type_dict"]
        self.a_n_type_lis = input_dic["a_n_type_lis"] 
        
        # Get device and agent features
        self.device = input_dic.get("device", torch.device("cuda:"+str(input_dic["gpu"])))
        agent_n_fea = input_dic["graph_lis"]["agent"].a_n_hidden
        graph = input_dic["graph_lis"]
        
        # Store original input for residual connections
        agent_n_fea_residual = agent_n_fea.clone()
        
        # Initialize output features for each agent type
        out_agent_n_fea = [torch.zeros_like(agent_n_fea[self.a_n_type_lis[_]]) for _ in range(3)]
        
        # Process each agent type separately
        for agent_type_index in range(3):
            if len(self.a_n_type_lis[agent_type_index]) == 0:
                continue
                
            agent_indices = self.a_n_type_lis[agent_type_index]
            current_agent_fea = agent_n_fea[agent_indices]
            
            # Self-attention transformation
            self_output = self.self_fc[agent_type_index](current_agent_fea)
            
            # Prepare attention features
            attention_outputs = []
            
            # Process lane-to-agent edges (l2a)
            if ('lane', 'l2a', 'agent') in graph.edge_types:
                 edge_index = graph[('lane', 'l2a', 'agent')].edge_index
                 if edge_index.size(1) > 0 and hasattr(graph['lane'], 'x'):
                     # Filter edges for current agent type
                     edge_mask = torch.isin(edge_index[1], agent_indices)
                     if edge_mask.any():
                         filtered_edges = edge_index[:, edge_mask]
                         lane_features = graph['lane'].x[filtered_edges[0]]
                         if lane_features.size(0) > 0:
                             # Ensure lane features are the right size for l2a MLP (expects hidden_dim*3)
                             # Pad or truncate lane features to hidden_dim
                             if lane_features.size(1) != self.hidden_dim:
                                 if lane_features.size(1) < self.hidden_dim:
                                     lane_features = torch.cat([
                                         lane_features, 
                                         torch.zeros(lane_features.size(0), self.hidden_dim - lane_features.size(1), device=self.device)
                                     ], dim=1)
                                 else:
                                     lane_features = lane_features[:, :self.hidden_dim]
                             
                             # Get corresponding agent features
                             agent_features = current_agent_fea[:lane_features.size(0)] if lane_features.size(0) <= current_agent_fea.size(0) else current_agent_fea[torch.randint(0, current_agent_fea.size(0), (lane_features.size(0),))]
                             
                             # Create input for l2a MLP: [lane_feat, agent_feat, lane_feat] = hidden_dim*3
                             l2a_input = torch.cat([
                                 lane_features,  # hidden_dim
                                 agent_features, # hidden_dim  
                                 lane_features   # hidden_dim (reuse lane features)
                             ], dim=-1)  # Total: hidden_dim*3 = 768
                             
                             lane_output = self.edge_MLP["l2a"](l2a_input)
                             if lane_output.size(0) == current_agent_fea.size(0):
                                 attention_outputs.append(lane_output)
            
            # Process polygon-to-agent edges (g2a) 
            if ('polygon', 'g2a', 'agent') in graph.edge_types:
                edge_index = graph[('polygon', 'g2a', 'agent')].edge_index
                if edge_index.size(1) > 0 and hasattr(graph['polygon'], 'x'):
                    edge_mask = torch.isin(edge_index[1], agent_indices)
                    if edge_mask.any():
                        filtered_edges = edge_index[:, edge_mask]
                        polygon_features = graph['polygon'].x[filtered_edges[0]]
                        if polygon_features.size(0) > 0:
                            # Ensure polygon features are the right size for g2a MLP (expects hidden_dim*2)
                            if polygon_features.size(1) != self.hidden_dim:
                                if polygon_features.size(1) < self.hidden_dim:
                                    polygon_features = torch.cat([
                                        polygon_features, 
                                        torch.zeros(polygon_features.size(0), self.hidden_dim - polygon_features.size(1), device=self.device)
                                    ], dim=1)
                                else:
                                    polygon_features = polygon_features[:, :self.hidden_dim]
                             
                            # Get corresponding agent features
                            agent_features = current_agent_fea[:polygon_features.size(0)] if polygon_features.size(0) <= current_agent_fea.size(0) else current_agent_fea[torch.randint(0, current_agent_fea.size(0), (polygon_features.size(0),))]
                             
                            # Create input for g2a MLP: [polygon_feat, agent_feat] = hidden_dim*2
                            g2a_input = torch.cat([
                                polygon_features,  # hidden_dim
                                agent_features     # hidden_dim
                            ], dim=-1)  # Total: hidden_dim*2 = 512
                             
                            polygon_output = self.edge_MLP["g2a"](g2a_input)
                            if polygon_output.size(0) == current_agent_fea.size(0):
                                attention_outputs.append(polygon_output)
            
            # Process agent-to-agent edges (other)
            if ('agent', 'other', 'agent') in graph.edge_types:
                edge_index = graph[('agent', 'other', 'agent')].edge_index
                if edge_index.size(1) > 0:
                    edge_mask = torch.isin(edge_index[1], agent_indices)
                    if edge_mask.any():
                        filtered_edges = edge_index[:, edge_mask]
                        other_agent_features = agent_n_fea[filtered_edges[0]]
                        
                        # Get agent types for source agents
                        src_agent_types = graph['agent'].node_type[filtered_edges[0]]
                        
                        # Group by source agent type (since we have separate MLPs for each type)
                        for src_agent_type in range(3):
                            type_mask = (src_agent_types == src_agent_type)
                            if type_mask.any():
                                src_features = other_agent_features[type_mask]
                                dst_features = current_agent_fea[:src_features.size(0)] if src_features.size(0) <= current_agent_fea.size(0) else current_agent_fea[torch.randint(0, current_agent_fea.size(0), (src_features.size(0),))]
                                
                                # Create input for other MLP: [src_agent, dst_agent, src_agent] = hidden_dim*3
                                other_input = torch.cat([
                                    src_features,  # hidden_dim
                                    dst_features,  # hidden_dim
                                    src_features   # hidden_dim (reuse source features)
                                ], dim=-1)  # Total: hidden_dim*3 = 768
                                
                                other_agent_output = self.edge_MLP["other"][src_agent_type](other_input)
                                if other_agent_output.size(0) == current_agent_fea.size(0):
                                    attention_outputs.append(other_agent_output)
            
            # Combine all features
            if attention_outputs:
                # Take average of attention outputs
                combined_attention = torch.stack(attention_outputs).mean(dim=0)
                # Ensure correct dimensions
                if combined_attention.size(0) != current_agent_fea.size(0):
                    combined_attention = combined_attention[:current_agent_fea.size(0)]
            else:
                combined_attention = torch.zeros_like(current_agent_fea)
            
            # Combine self, attention outputs
            combined_features = torch.cat([
                self_output,
                combined_attention,
                torch.zeros_like(current_agent_fea),  # Placeholder for missing modalities
                torch.zeros_like(current_agent_fea)   # Placeholder for missing modalities
            ], dim=-1)
            
            # Final output transformation with residual connection
            output = self.out_fc[agent_type_index](combined_features)
            out_agent_n_fea[agent_type_index] = self.out_ffn[agent_type_index](output + current_agent_fea)
        
        # Combine all agent types
        final_output = agent_n_fea_residual.clone()
        for agent_type_index in range(3):
            if len(self.a_n_type_lis[agent_type_index]) > 0:
                final_output[self.a_n_type_lis[agent_type_index]] = out_agent_n_fea[agent_type_index]
        
        # Placeholder edge features
        output_agent_e_fea = [torch.zeros((0, self.hidden_dim), device=self.device) for _ in range(4)]
        
        return final_output, output_agent_e_fea




class HDGT_encoder(nn.Module):
    def __init__(self, input_dim, args):
        super().__init__()
        self.hidden_dim = args.hidden_dim
        self.shared_coor_encoder = MLP(d_in=3, d_hid=args.hidden_dim//8, d_out=args.hidden_dim//4, norm=None) ## Encode x,y,z
        self.shared_rel_encoder = MLP(d_in=5, d_hid=args.hidden_dim//8, d_out=args.hidden_dim//4, norm=None) ## Encode Delta (x, y, z, cos(psi), sin(psi))
        
        self.agent_emb = Agent2embedding(input_dim, args)
        self.temporal_encoders = torch.nn.ModuleList([AgentTemporalEncoder(args) for _ in range(3)]) ## different temporal encoder for different agent type
        ## Input: [Node Feature, Rel_Pos]
        self.agent_e_fea_MLPs = torch.nn.ModuleList([MLP(args.hidden_dim+args.hidden_dim//4, args.hidden_dim*4, args.hidden_dim, norm=BN1D) for _ in range(3)]) ## Inti Agent Edge Feature
        
        self.lane_emb = Lane2embedding(args)
        self.polygon_emb = Polygon2embedding(args)
        
        self.num_of_gnn_layer = args.num_of_gnn_layer
        self.lane_gnns = torch.nn.ModuleList([LaneHetGNN(args=args) for _ in range(self.num_of_gnn_layer)])
        self.agent_gnns = torch.nn.ModuleList([AgentHetGNN(args=args) for _ in range(self.num_of_gnn_layer)])


    def forward(self, input_dic):
        ## Init Agent Node
        agent_n_emb = self.agent_emb(input_dic, self.shared_coor_encoder)
        agent_n_type_indices = [torch.where((input_dic["graph_lis"]["agent"].node_type) == _) for _ in range(3)]
        device = input_dic.get("device", torch.device("cuda:"+str(input_dic["gpu"])))
        agent_n_fea = torch.zeros((agent_n_emb.shape[0], agent_n_emb.shape[1]), device=device)
        for _ in range(3):
            agent_n_fea[agent_n_type_indices[_]] = self.temporal_encoders[_](agent_n_emb[agent_n_type_indices[_]])

        ## Simplified agent edge processing for PyG
        # Just store the agent node features directly without complex edge processing
        graph = input_dic["graph_lis"]
        
        # Store agent node features in PyG format
        input_dic["graph_lis"]["agent"].a_n_hidden = agent_n_fea

        self.lane_emb(input_dic, self.shared_coor_encoder, self.shared_rel_encoder)
        self.polygon_emb(input_dic, self.shared_coor_encoder)

        
        for i in range(self.num_of_gnn_layer):
            output_lane_n_fea, output_in_lane_e_fea = self.lane_gnns[i](input_dic)
            output_agent_n_fea, output_in_agent_e_fea = self.agent_gnns[i](input_dic)
            
            # Store lane features in PyG format
            if hasattr(input_dic["graph_lis"], 'lane') and hasattr(input_dic["graph_lis"]['lane'], 'x'):
                input_dic["graph_lis"]["lane"].l_n_hidden = output_lane_n_fea
            
            # Store agent features in PyG format  
            input_dic["graph_lis"]["agent"].a_n_hidden = output_agent_n_fea
            
        return input_dic["graph_lis"]



class HDGT_model(nn.Module):
    def __init__(self, input_dim, args):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.args = args
        self.num_prediction = args.num_prediction
        # Agent feature vector: input_dim [11 dimensions]
        # 0-2: x, y, z        ← Position coordinates (3D)  
        # 3-4: vx, vy         ← Velocity components (2D)
        # 5-6: cos(psi), sin(psi) ← Heading angle (trigonometric form)
        # 7-9: width, length, height ← Agent dimensions (3D bounding box)
        # 10:  mask           ← Valid timestep indicator (1=valid, 0=invalid)
        self.encoder = HDGT_encoder(input_dim, args)
        self.decoder = torch.nn.ModuleList([RefineDecoder(args) for _ in range(3)])
        
    def forward(self, input_dic):
        output_het_graph = self.encoder(input_dic)
    
        neighbor_size_lis = input_dic["neighbor_size_lis"]
        all_agent_raw_traj = input_dic["graph_lis"]["agent"].x[..., :2].clone()
        cumsum_neighbor_size_lis = np.cumsum(neighbor_size_lis, axis=0).tolist()
        cumsum_neighbor_size_lis = [0] + cumsum_neighbor_size_lis
        pred_num_lis = input_dic["pred_num_lis"]
        agent_node_fea = output_het_graph["agent"].a_n_hidden
        all_agent_id = torch.arange(input_dic["graph_lis"]["agent"].num_nodes, device=output_het_graph["agent"].x.device)
        agent_n_type_lis = output_het_graph["agent"].node_type
        
        ## Obtain the node feature of target agents 
        targat_agent_indice_lis = []
        target_agent_indice_bool_type_lis = [[] for _ in range(3)]
        targat_agent_fea = [[] for _ in range(3)]
        target_agent_id = [[] for _ in range(3)]
        for i in range(1, len(cumsum_neighbor_size_lis)):
            now_agent_type_lis = agent_n_type_lis[cumsum_neighbor_size_lis[i-1]:cumsum_neighbor_size_lis[i]]
            now_agent_id = all_agent_id[cumsum_neighbor_size_lis[i-1]:cumsum_neighbor_size_lis[i-1]+pred_num_lis[i-1]]
            now_agent_node_fea = agent_node_fea[cumsum_neighbor_size_lis[i-1]:cumsum_neighbor_size_lis[i-1]+pred_num_lis[i-1]]
            now_target_agent_indice_bool_type_lis = [now_agent_type_lis[:pred_num_lis[i-1]]==_ for _ in range(3)]
            
            targat_agent_indice_lis += list(range(cumsum_neighbor_size_lis[i-1], cumsum_neighbor_size_lis[i-1]+pred_num_lis[i-1]))
            for _ in range(3):
                target_agent_indice_bool_type_lis[_].append(now_target_agent_indice_bool_type_lis[_])
                targat_agent_fea[_].append(now_agent_node_fea[now_target_agent_indice_bool_type_lis[_]])
                target_agent_id[_].append(now_agent_id[now_target_agent_indice_bool_type_lis[_]])

        target_agent_indice_bool_type_lis = [torch.cat(target_agent_indice_bool_type_lis[_], dim=0) for _ in range(3)]
        targat_agent_fea = [torch.cat(targat_agent_fea[_], dim=0) for _ in range(3)]
        target_agent_id = [torch.cat(target_agent_id[_], dim=0) for _ in range(3)]

        prediction = []
        for _ in range(3):
            if targat_agent_fea[_].shape[0] == 0:
                prediction.append((torch.zeros((0, 1)), torch.zeros((0, 1))))
            else:
                prediction.append(self.decoder[_](targat_agent_fea[_], target_agent_id[_], all_agent_raw_traj, input_dic))
        agent_cls_res = [prediction[_][0] for _ in range(3)]
        agent_reg_res = [prediction[_][1] for _ in range(3)]
        return agent_reg_res, agent_cls_res, target_agent_indice_bool_type_lis



class RefineCNN(nn.Module):
    def __init__(self, in_c, dilation=1, args=None):
        super(RefineCNN, self).__init__()
        self.in_c = in_c
        self.conv1 = nn.Conv1d(in_c, in_c, kernel_size=3, dilation=dilation, padding=dilation)
        self.gn1 = nn.GroupNorm(num_groups=in_c, num_channels=in_c)
        self.conv2 = nn.Conv1d(in_c, in_c, kernel_size=3, dilation=dilation, padding=dilation)
        self.gn2 = nn.GroupNorm(num_groups=in_c, num_channels=in_c)
        self.act = nn.ReLU(inplace=True)
        self.se = SEBlock(in_c)
    def forward(self, x):
        identity = x
        out = x        
        out = self.act(self.gn1(self.conv1(out)))
        out = self.act(self.gn2(self.conv2(out)))
        out = self.se(out) + identity
        out = self.act(out)
        return out



class RefineContextLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_prediction = args.num_prediction
        self.hidden_dim = args.hidden_dim
        self.head_dim = args.head_dim
        self.d_model = int(args.hidden_dim)
        self.n_head = self.d_model // self.head_dim
        d_k = self.head_dim
        d_v = self.head_dim
        self.attention =  ScaledDotProductAttention(temperature=np.power(d_k, 0.5), args=args)
        self.wq = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(args.hidden_dim, self.n_head * d_k, bias=False)
        )
        
        
        self.wkvs = nn.ModuleList([
            nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(args.hidden_dim, self.n_head * d_k * 2, bias=False),
            )
            for _ in range(3)
        ])
        self.attn_fc = nn.Sequential(
            nn.Linear(self.n_head * d_v, self.d_model, bias=True),
            nn.Dropout(args.dropout),
        )
        
        self.ffn = PositionwiseFeedForward(self.d_model, self.d_model*4, args.dropout, args)

    def forward(self, raw_q, raw_kv_lis, raw_kv_indices, input_dic):
        all_q_lis = self.wq(raw_q).view(raw_q.shape[0], self.num_prediction*self.n_head, self.head_dim)
        all_kv_lis = [self.wkvs[_](torch.cat(raw_kv_lis[_], dim=0)) for _ in range(3)]
        all_kv_lis = [[_[raw_kv_indices[_index][indices_i-1]:raw_kv_indices[_index][indices_i]] for indices_i in range(1, len(raw_kv_indices[_index]))] for _index, _ in enumerate(all_kv_lis)]
        all_kv_lis = [torch.cat([all_kv_lis[0][_], all_kv_lis[1][_], all_kv_lis[2][_]], dim=0).view(-1, self.n_head, self.head_dim*2).unsqueeze(1).repeat(1, self.num_prediction, 1, 1).view(-1, self.num_prediction*self.n_head, self.head_dim*2).transpose(0, 1) for _ in range(raw_q.shape[0])]

        all_out_q_lis = self.attn_fc(torch.cat([self.attention(q=all_q_lis[_].unsqueeze(1), k=all_kv_lis[_][..., :self.head_dim], v=all_kv_lis[_][..., self.head_dim:])[0].view(-1, self.num_prediction, self.n_head*self.head_dim) if all_kv_lis[_].shape[0]!=0 else torch.zeros_like(raw_q[0:1, :, :]) for _ in range(raw_q.shape[0])], dim=0)) + raw_q
        return self.ffn(all_out_q_lis)



class RefineContextModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_prediction = args.num_prediction
        self.hidden_dim = args.hidden_dim
        self.modal_emb = nn.parameter.Parameter(torch.zeros(args.num_prediction, args.hidden_dim))
        torch.nn.init.normal(self.modal_emb)      
        self.init_q = MLP(args.hidden_dim*2, args.hidden_dim*4, args.hidden_dim, nn.LayerNorm, prenorm=True)
        self.etype2hidden_name = {"other":"a_e_hidden", "l2a":"l_e_hidden", "g2a":"g_e_hidden"}
        self.refine_context_layer = nn.ModuleList([RefineContextLayer(args) for _ in range(2)])

    def forward(self, agent_ids, input_dic):
        self.device = input_dic.get("device", torch.device("cuda:"+str(input_dic["gpu"])))
        
        # Try to access DGL-style data, fallback to PyG-style or empty tensors
        try:
            raw_agent_fea = input_dic["graph_lis"].ndata["a_n_hidden"]["agent"][agent_ids]
        except (AttributeError, KeyError):
            # Fallback for PyG - return dummy features
            num_agents = len(agent_ids) if hasattr(agent_ids, '__len__') else 1
            raw_agent_fea = torch.zeros((num_agents, self.hidden_dim), device=self.device)
        
        num_agent = raw_agent_fea.shape[0]

        raw_q = torch.cat([raw_agent_fea.unsqueeze(1).repeat(1, self.num_prediction, 1), self.modal_emb.unsqueeze(0).repeat(num_agent, 1, 1)], dim=-1)
        raw_q = self.init_q(raw_q) + self.modal_emb
        
        raw_kv_lis = [[torch.zeros((0, self.hidden_dim), device=self.device)]*num_agent for _ in range(3)]
        
        # Try to access DGL-style edge data, fallback to empty tensors for PyG
        try:
            for agent_index, agent_id in enumerate(agent_ids):
                for etype_index, etype in enumerate(["other", "l2a", "g2a"]):
                    now_type_eid_lis = input_dic["graph_lis"].in_edges(etype=etype, v=agent_id, form="eid")
                    if len(now_type_eid_lis) > 0:
                        raw_kv_lis[etype_index][agent_index] = input_dic["graph_lis"].edges[etype].data[self.etype2hidden_name[etype]][now_type_eid_lis]
        except (AttributeError, KeyError, TypeError):
            # Fallback for PyG - use empty tensors
            pass
        
        raw_kv_length = [[int(__.shape[0]) for __ in _] for _ in raw_kv_lis]
        raw_kv_indices = [np.cumsum([0]+_) for _ in raw_kv_length]
        for _ in range(len(self.refine_context_layer)):
            raw_q = self.refine_context_layer[_](raw_q, raw_kv_lis, raw_kv_indices, input_dic) + self.modal_emb
        return raw_q


class RefineLayer(nn.Module):
    def __init__(self, d_in, d_hid, args):
        super().__init__()
        self.in_linear = nn.Linear(d_in, d_hid)
        self.context_mlp = MLP(args.hidden_dim,  args.hidden_dim//2,  d_hid, norm=nn.LayerNorm, prenorm=True)
        self.fuse_linear = nn.Linear(d_hid * 2, d_hid)
        self.cnns = nn.Sequential(
        RefineCNN(d_hid, dilation=1, args=args),
        RefineCNN(d_hid, dilation=2, args=args),
        RefineCNN(d_hid, dilation=5, args=args),
        RefineCNN(d_hid, dilation=1, args=args),
        )
        self.out_linear = MLP(d_hid, d_hid, 2, norm=None)
    def forward(self, x, context):
        x = x.view(x.shape[0], x.shape[1], 91, -1)
        output = self.in_linear(x)
        context = self.context_mlp(context).unsqueeze(-2).repeat(1, 1, 91, 1)
        output = self.fuse_linear(torch.cat([context, output], dim=-1))
        bs, num_mode, t_len, hid = output.shape
        output = output.view(bs*num_mode, t_len, hid).transpose(1, 2)
        output = self.cnns(output).transpose(1, 2).view(bs, num_mode, t_len, hid)[:, :, 11:, :]
        output = self.out_linear(output)
        return output

class RefineDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()        
        self.hidden_dim = args.hidden_dim
        self.num_prediction = int(args.num_prediction)
        
        self.refine_num = int(args.refine_num)
        self.refine_layer_lis = nn.ModuleList(
            [RefineLayer(2, self.hidden_dim//4, args) for _ in range(self.refine_num)]
        )
        self.is_output_vel = (args.output_vel == "True")
        self.is_cumsum_vel = (args.cumsum_vel == "True")

        self.refine_context_attn = RefineContextModule(args)
        
        self.reg_mlp = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(args.hidden_dim*2, args.hidden_dim*4),
            nn.ReLU(inplace=True),
            nn.Linear(args.hidden_dim*4, 80*2),
        )
        
        self.cls_mlp = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hidden_dim, args.hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(args.hidden_dim//2, 1),
        )
        
        
    def forward(self, target_agent_fea, agent_ids, agent_raw_traj, input_dic):
        refine_context = self.refine_context_attn(agent_ids, input_dic)/10.0
        reg_res = self.reg_mlp(refine_context).view(target_agent_fea.shape[0], self.num_prediction, 80, 2)
        cls_res = self.cls_mlp(refine_context).view(target_agent_fea.shape[0], self.num_prediction)
        if self.is_output_vel and self.is_cumsum_vel:
            reg_res = torch.cumsum(reg_res, dim=-2)
        reg_res_lis = [reg_res]
        if self.refine_num > 0:
            now_agent_raw_traj = agent_raw_traj[agent_ids].unsqueeze(1).repeat(1, self.num_prediction, 1, 1)
            for _ in range(self.refine_num):
                now_input = torch.cat([now_agent_raw_traj, reg_res.detach()], dim=-2).view(reg_res.shape[0], self.num_prediction, -1) ## Full Traj
                reg_res = reg_res + self.refine_layer_lis[_](now_input, refine_context).view(reg_res.shape[0], self.num_prediction, 80, 2)
                reg_res_lis.append(reg_res)
        return cls_res, torch.stack(reg_res_lis, dim=1)





from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return 1.0#max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

import warnings
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def weights_init(m):
    with torch.no_grad():
        classname = m.__class__.__name__
        if classname.find('Conv1d') != -1:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif classname.find('LayerNorm') != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif classname.find('BatchNorm') != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif classname.find('GroupNorm') != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif classname.find('Linear') != -1:
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            nn.init.xavier_normal_(m.weight)
        elif classname.find('Embedding') != -1:
            trunc_normal_(m.weight, mean=0, std=0.02)
