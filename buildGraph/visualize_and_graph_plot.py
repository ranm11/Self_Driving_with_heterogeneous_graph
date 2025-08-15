import pickle
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.transforms import Affine2D
import networkx as nx
import numpy as np
import argparse
import os

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
# Default values - will be overridden by command line args
PKL_PATH = "scenario.pkl"
OUTPUT_IMG = "scenario_with_graph.png"

# Node type colors
NODE_COLORS = {
    "lane": "green",
    "vehicle": "blue",
    "pedestrian": "orange",
    "cyclist": "purple",
    "other": "brown",
    "traffic_light": "red",
    "crosswalk": "pink",
    "stop_sign": "darkred",
    "ego": "black"
}

# Lane segmentation config
SEGMENT_LENGTH_METERS = 10.0

# Default vehicle rectangle dimensions (meters)
DEFAULT_VEHICLE_LENGTH = 4.5
DEFAULT_VEHICLE_WIDTH = 1.9
EGO_LENGTH = 4.8
EGO_WIDTH = 2.0

# Curve densification factor (higher -> denser nodes in curves)
CURVE_DENSIFY_FACTOR = 3.0

# ------------------------------------------------------------
# COMMAND LINE ARGUMENTS
# ------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize driving scenario with graph overlay")
    parser.add_argument("pkl_file", nargs='?', default="/localdrive/users/ranmi/HDGT/dataset/waymo/training/hdgt_waymo_dev_tmp0/scenario0_t044.pkl",
                        help="Path to the PKL file to visualize")
    parser.add_argument("--output_dir", "-o", default=".", 
                       help="Directory to save the output PNG file (default: current directory)")
    parser.add_argument("--output_name", default=None,
                       help="Custom output filename (default: based on input PKL name)")
    # Forward-view filtering options
    parser.add_argument("--filter_forward", action="store_true", default=True,
                       help="Filter graph nodes/edges to a forward-looking FOV ahead of ego")
    parser.add_argument("--fov_deg", type=float, default=120.0,
                       help="Forward field-of-view in degrees centered on +X axis (ego heading)")
    parser.add_argument("--max_range", type=float, default=80.0,
                       help="Max range (meters) for including nodes in graph")
    parser.add_argument("--segment_length", type=float, default=SEGMENT_LENGTH_METERS,
                       help="Lane segment length in meters for lane node creation")
    return parser.parse_args()

# ------------------------------------------------------------
# STEP 1: Load PKL data
# ------------------------------------------------------------
def load_pkl_data(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data

# ------------------------------------------------------------
# STEP 2: Visualization function for raw scenario
# ------------------------------------------------------------
def plot_scenario(ax, data):
    # Extract data from new PKL structure
    map_features = data.get("map_features", [])
    traffic_lights = data.get("traffic_lights", [])
    agents = data.get("agents", [])
    
    # Group agents by type
    vehicles = [a for a in agents if a["type"] == "VEHICLE"]
    pedestrians = [a for a in agents if a["type"] == "PEDESTRIAN"]
    cyclists = [a for a in agents if a["type"] == "CYCLIST"]
    others = [a for a in agents if a["type"] in ["OTHER", "UNKNOWN"]]
    
    # Group map features by type
    lanes = [mf for mf in map_features if mf["type"] == "lane"]
    crosswalks = [mf for mf in map_features if mf["type"] == "crosswalk"]
    road_lines = [mf for mf in map_features if mf["type"] == "road_line"]
    road_edges = [mf for mf in map_features if mf["type"] == "road_edge"]
    stop_signs = [mf for mf in map_features if mf["type"] == "stop_sign"]
    
    # Plot map features
    for lane in lanes:
        if lane["vertices"]:
            coords = np.array(lane["vertices"])
            ax.plot(coords[:,0], coords[:,1], color="green", linewidth=1, alpha=0.7)
    
    for road_line in road_lines:
        if road_line["vertices"]:
            coords = np.array(road_line["vertices"])
            ax.plot(coords[:,0], coords[:,1], color="yellow", linewidth=1, alpha=0.5)
            
    for road_edge in road_edges:
        if road_edge["vertices"]:
            coords = np.array(road_edge["vertices"])
            ax.plot(coords[:,0], coords[:,1], color="gray", linewidth=1, alpha=0.5)

    for crosswalk in crosswalks:
        if crosswalk["vertices"]:
            coords = np.array(crosswalk["vertices"])
            ax.plot(coords[:,0], coords[:,1], color="pink", linewidth=2, alpha=0.7)

    # Plot stop signs
    for stop_sign in stop_signs:
        if stop_sign["vertices"]:
            coords = np.array(stop_sign["vertices"])
            ax.plot(coords[:,0], coords[:,1], "ro", markersize=8, label="Stop Sign" if stop_sign == stop_signs[0] else "")

    # Plot agents
    for veh in vehicles:
        x, y = veh["position"]
        length = float(veh.get("length", DEFAULT_VEHICLE_LENGTH))
        width = float(veh.get("width", DEFAULT_VEHICLE_WIDTH))
        heading = float(veh.get("heading", 0.0))
        rect = patches.Rectangle((x - length / 2.0, y - width / 2.0),
                                 length, width,
                                 linewidth=1.5, edgecolor="blue", facecolor="none")
        rect.set_transform(Affine2D().rotate_around(x, y, heading) + ax.transData)
        ax.add_patch(rect)

    for ped in pedestrians:
        x, y = ped["position"]
        ax.plot(x, y, "o", color="orange", markersize=4)

    for cyc in cyclists:
        x, y = cyc["position"]
        ax.plot(x, y, "^", color="purple", markersize=4)
        
    for other in others:
        x, y = other["position"]
        ax.plot(x, y, "d", color="brown", markersize=4)

    # Plot traffic lights
    for tl in traffic_lights:
        x, y = tl["position"]
        ax.plot(x, y, "s", color="red", markersize=7)

    # Plot ego vehicle as oriented rectangle (always centered at origin in ego-centric coordinates)
    ego_pose = data.get("ego_pose", (0.0, 0.0, 0.0))
    ego_rect = patches.Rectangle((-EGO_LENGTH / 2.0, -EGO_WIDTH / 2.0),
                                 EGO_LENGTH, EGO_WIDTH,
                                 linewidth=2.0, edgecolor="black", facecolor="none")
    ego_rect.set_transform(Affine2D().rotate_around(0.0, 0.0, 0.0) + ax.transData)
    ax.add_patch(ego_rect)

    ax.set_aspect("equal")
    ax.set_title("Scenario Visualization with Graph Overlay")
    
    # Add legend for visualization elements
    legend_elements = []
    
    # Map features
    if lanes:
        legend_elements.append(plt.Line2D([0], [0], color='green', linewidth=2, label='Lanes'))
    if road_lines:
        legend_elements.append(plt.Line2D([0], [0], color='yellow', linewidth=2, label='Road Lines'))
    if road_edges:
        legend_elements.append(plt.Line2D([0], [0], color='gray', linewidth=2, label='Road Edges'))
    if crosswalks:
        legend_elements.append(plt.Line2D([0], [0], color='pink', linewidth=3, label='Crosswalks'))
    
    # Agents
    if vehicles:
        legend_elements.append(patches.Patch(edgecolor='blue', facecolor='none', label='Vehicles'))
    if pedestrians:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                                        markersize=8, label='Pedestrians', linestyle='None'))
    if cyclists:
        legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='purple', 
                                        markersize=8, label='Cyclists', linestyle='None'))
    if others:
        legend_elements.append(plt.Line2D([0], [0], marker='d', color='w', markerfacecolor='brown', 
                                        markersize=8, label='Others', linestyle='None'))
    
    # Traffic elements
    if traffic_lights:
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                                        markersize=8, label='Traffic Lights', linestyle='None'))
    if stop_signs:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                        markersize=10, label='Stop Signs', linestyle='None'))
    
    # Ego vehicle (always present)
    legend_elements.append(patches.Patch(edgecolor='black', facecolor='none', label='Ego Vehicle'))
    
    # Add legend if we have elements
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0), 
                 frameon=True, fancybox=True, shadow=True, fontsize=10)

# ------------------------------------------------------------
# STEP 3: Convert polylines to graph nodes
# ------------------------------------------------------------
# Helper to filter nodes to forward FOV
default_half_fov_rad = np.deg2rad(120.0 / 2.0)

def _is_in_forward_view(pos_xy, half_fov_rad, max_range):
    x, y = pos_xy
    if max_range is not None and np.hypot(x, y) > max_range:
        return False
    # Forward direction assumed to be +X in ego-centric frame
    if x <= 0:
        return False
    angle = np.arctan2(y, x)
    return abs(angle) <= half_fov_rad

# Geometry helpers for segmentation

def _polyline_lengths(coords):
    if len(coords) < 2:
        return np.array([0.0])
    diffs = np.diff(coords, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_lens = np.concatenate(([0.0], np.cumsum(seg_lens)))
    return cum_lens

def _interpolate_point(p0, p1, t):
    return p0 + t * (p1 - p0)

# Curvature-aware helpers

def _compute_turn_angles(coords):
    if len(coords) < 3:
        return np.zeros(len(coords))
    v_prev = coords[1:-1] - coords[:-2]
    v_next = coords[2:] - coords[1:-1]
    # Normalize
    prev_norm = np.linalg.norm(v_prev, axis=1)
    next_norm = np.linalg.norm(v_next, axis=1)
    # Avoid division by zero
    valid = (prev_norm > 1e-8) & (next_norm > 1e-8)
    angles_interior = np.zeros(len(coords) - 2)
    if np.any(valid):
        v_prev_u = v_prev[valid] / prev_norm[valid][:, None]
        v_next_u = v_next[valid] / next_norm[valid][:, None]
        # Angle between vectors using stable atan2 formulation
        cross_z = v_prev_u[:, 0] * v_next_u[:, 1] - v_prev_u[:, 1] * v_next_u[:, 0]
        dot = (v_prev_u * v_next_u).sum(axis=1)
        angles_masked = np.arctan2(np.abs(cross_z), np.clip(dot, -1.0, 1.0))
        angles_interior[valid] = angles_masked
    # Pad with zeros for endpoints
    full_angles = np.zeros(len(coords))
    full_angles[1:-1] = angles_interior
    return full_angles

def _adaptive_sample_positions_on_polyline(coords, base_step, curve_factor=3.0, min_step=None, start_s=None):
    coords = np.asarray(coords, dtype=float)
    cum_lens = _polyline_lengths(coords)
    total_len = cum_lens[-1]
    if total_len == 0.0:
        return [coords.mean(axis=0)]
    if min_step is None:
        min_step = max(0.5, float(base_step) * 0.33)
    base_step = max(1e-3, float(base_step))

    # Compute per-vertex turn angles
    angles = _compute_turn_angles(coords)

    # Helper: get local step based on curvature at arc-length s
    def _local_step(s):
        # find segment index
        idx = np.searchsorted(cum_lens, s, side='right') - 1
        idx = int(np.clip(idx, 0, len(coords) - 2))
        # Normalize angle: 0..pi mapped against 60 deg reference
        angle_here = float(angles[idx])
        angle_norm = min(1.0, angle_here / np.deg2rad(60.0))
        return max(min_step, base_step / (1.0 + curve_factor * angle_norm))

    # March along arc-length with variable step
    positions = []
    s0 = 0.0 if start_s is None else float(np.clip(start_s, 0.0, total_len))
    s = min(s0 + 0.5 * base_step, total_len)
    while s <= total_len:
        # locate segment for s
        idx = np.searchsorted(cum_lens, s, side='right') - 1
        idx = int(np.clip(idx, 0, len(coords) - 2))
        d0 = cum_lens[idx]
        d1 = cum_lens[idx + 1]
        if d1 == d0:
            p = coords[idx].copy()
        else:
            t = (s - d0) / (d1 - d0)
            p = _interpolate_point(coords[idx], coords[idx + 1], t)
        positions.append(p)
        s += _local_step(s)

    # Ensure last position at the end if not already very close
    if np.linalg.norm(positions[-1] - coords[-1]) > 1e-3:
        positions.append(coords[-1])
    return positions

def _nearest_arclength_to_point(coords, point_xy):
    coords = np.asarray(coords, dtype=float)
    cum_lens = _polyline_lengths(coords)
    px, py = float(point_xy[0]), float(point_xy[1])
    best_d2 = float('inf')
    best_s = 0.0
    for i in range(len(coords) - 1):
        ax, ay = coords[i]
        bx, by = coords[i + 1]
        vx, vy = bx - ax, by - ay
        seg_len2 = vx * vx + vy * vy
        if seg_len2 == 0.0:
            cx, cy = ax, ay
            t = 0.0
        else:
            t = ((px - ax) * vx + (py - ay) * vy) / seg_len2
            t = max(0.0, min(1.0, t))
            cx, cy = ax + t * vx, ay + t * vy
        dx, dy = px - cx, py - cy
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            # arc-length up to this projected point
            best_s = float(cum_lens[i] + t * np.sqrt(seg_len2))
    return best_s

def build_heterogeneous_graph(data, filter_forward=True, fov_deg=120.0, max_range=80.0, segment_length=SEGMENT_LENGTH_METERS):
    # Extract data from new PKL structure
    map_features = data.get("map_features", [])
    traffic_lights = data.get("traffic_lights", [])
    agents = data.get("agents", [])
    
    # Group agents by type
    vehicles = [a for a in agents if a["type"] == "VEHICLE"]
    pedestrians = [a for a in agents if a["type"] == "PEDESTRIAN"]
    cyclists = [a for a in agents if a["type"] == "CYCLIST"]
    others = [a for a in agents if a["type"] in ["OTHER", "UNKNOWN"]]
    
    # Group map features by type
    lanes_all = [mf for mf in map_features if mf["type"] == "lane"]
    ego_lane_id = data.get("ego_lane_id", None)
    def _lane_heading(coords, at_end=False):
        if len(coords) < 2:
            return 0.0
        if at_end:
            dy = coords[-1][1] - coords[-2][1]
            dx = coords[-1][0] - coords[-2][0]
        else:
            dy = coords[1][1] - coords[0][1]
            dx = coords[1][0] - coords[0][0]
        return float(np.arctan2(dy, dx))
    def _wrap_angle_diff(a, b):
        d = a - b
        while d > np.pi:
            d -= 2 * np.pi
        while d < -np.pi:
            d += 2 * np.pi
        return abs(d)
    def _find_successor_chain(lanes_all_list, start_lane_id, max_hops=20, dist_thresh=15.0, heading_thresh_rad=np.deg2rad(30.0)):
        id_to_lane = {}
        for lf in lanes_all_list:
            if "id" in lf:
                try:
                    id_to_lane[int(lf["id"])] = lf
                except Exception:
                    continue
        try:
            sid = int(start_lane_id) if start_lane_id is not None else None
        except Exception:
            sid = None
        chain = []
        if sid is None or sid not in id_to_lane:
            return chain
        current = id_to_lane[sid]
        chain.append(current)
        visited = {sid}
        for _ in range(max_hops):
            end_pt = np.asarray(current["vertices"][-1], dtype=float)
            end_heading = _lane_heading(current["vertices"], at_end=True)
            best = None
            best_d = float("inf")
            for cand in lanes_all_list:
                cid = int(cand.get("id", -1)) if "id" in cand else -1
                if cid in visited or cid == -1:
                    continue
                start_pt = np.asarray(cand["vertices"][0], dtype=float)
                d = float(np.linalg.norm(start_pt - end_pt))
                if d > dist_thresh:
                    continue
                h0 = _lane_heading(cand["vertices"], at_end=False)
                if _wrap_angle_diff(h0, end_heading) > heading_thresh_rad:
                    continue
                if d < best_d:
                    best = cand
                    best_d = d
            if best is None:
                break
            best_id = int(best.get("id", -1))
            visited.add(best_id)
            chain.append(best)
            current = best
        return chain
    if ego_lane_id is not None:
        lanes_chain = _find_successor_chain(lanes_all, ego_lane_id)
        lanes = lanes_chain if lanes_chain else [lf for lf in lanes_all if lf.get("id") == ego_lane_id]
    else:
        lanes = lanes_all
    crosswalks = [mf for mf in map_features if mf["type"] == "crosswalk"]
    stop_signs = [mf for mf in map_features if mf["type"] == "stop_sign"]
    
    G = nx.Graph()

    intra_lane_edges = []  # store edges between sequential lane segments (added after filtering)

    # Lane nodes: segment each lane polyline and add a node for each segment midpoint
    for lane_idx, lane in enumerate(lanes):
        if not lane["vertices"]:
            continue
        coords = np.array(lane["vertices"], dtype=float)
        step = max(1e-3, float(segment_length))
        # If this is the first lane and we know ego lane id, start from the nearest point to ego (0,0)
        start_s = None
        if lane_idx == 0 and data.get("ego_lane_id", None) is not None:
            start_s = _nearest_arclength_to_point(coords, (0.0, 0.0))
        positions = _adaptive_sample_positions_on_polyline(
            coords, base_step=step, curve_factor=CURVE_DENSIFY_FACTOR,
            min_step=max(0.5, step * 0.33), start_s=start_s
        )
        prev_node = None
        for seg_idx, p in enumerate(positions):
            node_name = f"lane_{lane_idx}_{seg_idx}"
            G.add_node(node_name, type="lane", pos=(float(p[0]), float(p[1])))
            if prev_node is not None:
                intra_lane_edges.append((prev_node, node_name))
            prev_node = node_name

    # Vehicle nodes
    for i, veh in enumerate(vehicles):
        x, y = veh["position"]
        #TODO add position and orientation to the node
        G.add_node(f"vehicle_{i}", type="vehicle", pos=(x, y))

    # Pedestrian nodes
    for i, ped in enumerate(pedestrians):
        x, y = ped["position"]
        #TODO add position and orientation to the node
        G.add_node(f"ped_{i}", type="pedestrian", pos=(x, y))

    # Cyclist nodes
    for i, cyc in enumerate(cyclists):
        x, y = cyc["position"]
        G.add_node(f"cyc_{i}", type="cyclist", pos=(x, y))
        
    # Other agent nodes
    for i, other in enumerate(others):
        x, y = other["position"]
        G.add_node(f"other_{i}", type="other", pos=(x, y))

    # Traffic light nodes
    for i, tl in enumerate(traffic_lights):
        x, y = tl["position"]
        G.add_node(f"tl_{i}", type="traffic_light", pos=(x, y))

    # Crosswalk nodes (use centroid of vertices)
    for i, cw in enumerate(crosswalks):
        if cw["vertices"]:
            coords = np.array(cw["vertices"])
            centroid = coords.mean(axis=0)
            G.add_node(f"cw_{i}", type="crosswalk", pos=(float(centroid[0]), float(centroid[1])))

    # Stop sign nodes
    for i, ss in enumerate(stop_signs):
        if ss["vertices"]:
            coords = np.array(ss["vertices"])
            if len(coords) > 0:
                # For stop signs, use first coordinate (position)
                pos = coords[0] if len(coords.shape) == 2 else coords
                G.add_node(f"stop_{i}", type="stop_sign", pos=(float(pos[0]), float(pos[1])))

    # Add ego vehicle as central node (always at origin in ego-centric coordinates)
    G.add_node("ego", type="ego", pos=(0, 0))

    # Optionally filter nodes to forward view
    if filter_forward:
        half_fov_rad = np.deg2rad(fov_deg / 2.0)
        node_positions_all = nx.get_node_attributes(G, "pos")
        nodes_to_keep = set()
        for n, p in node_positions_all.items():
            if n == "ego" or _is_in_forward_view(p, half_fov_rad, max_range):
                nodes_to_keep.add(n)
        for n in list(G.nodes):
            if n not in nodes_to_keep:
                G.remove_node(n)

    # Add intra-lane edges (sequential segments) if both nodes remain after filtering
    for n1, n2 in intra_lane_edges:
        if n1 in G and n2 in G:
            G.add_edge(n1, n2)

    # Add semantically meaningful edges instead of simple proximity
    node_positions = nx.get_node_attributes(G, "pos")
    node_types = nx.get_node_attributes(G, "type")
    
    # Define semantic connection rules with different thresholds
    connection_rules = {
        # Agent connections (smaller threshold, selective)
        ("ego", "lane"): 20.0,
        ("vehicle", "lane"): 12.0,
        ("pedestrian", "crosswalk"): 8.0,
        ("cyclist", "lane"): 10.0,
        
        # Traffic infrastructure connections
        ("traffic_light", "lane"): 15.0,
        ("stop_sign", "lane"): 12.0,
        ("crosswalk", "lane"): 10.0,
        
        # Lane-to-lane proximity connections (in addition to sequential ones)
        ("lane", "lane"): 6.0,
        
        # Ego connections (central hub)
        ("ego", "vehicle"): 30.0,
        ("ego", "pedestrian"): 25.0,
        ("ego", "traffic_light"): 35.0,
    }
    
    nodes_list = list(G.nodes)
    
    for i, n1 in enumerate(nodes_list):
        for j, n2 in enumerate(nodes_list):
            if i >= j:
                continue
            
            type1 = node_types.get(n1, "unknown")
            type2 = node_types.get(n2, "unknown")
            
            # Check both directions for connection rules
            threshold = None
            rule_key1 = (type1, type2)
            rule_key2 = (type2, type1)
            
            if rule_key1 in connection_rules:
                threshold = connection_rules[rule_key1]
            elif rule_key2 in connection_rules:
                threshold = connection_rules[rule_key2]
            
            # Only connect if there's a semantic rule
            if threshold is not None:
                p1 = np.array(node_positions[n1])
                p2 = np.array(node_positions[n2])
                distance = np.linalg.norm(p1 - p2)
                
                if distance <= threshold:
                    G.add_edge(n1, n2)

    return G

# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Generate output filename
    if args.output_name:
        output_filename = args.output_name
        if not output_filename.endswith('.png'):
            output_filename += '.png'
    else:
        # Use input filename with _graph suffix
        base_name = os.path.splitext(os.path.basename(args.pkl_file))[0]
        output_filename = f"{base_name}_graph.png"
    
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading PKL data from: {args.pkl_file}")
    data = load_pkl_data(args.pkl_file)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 12))
    plot_scenario(ax, data)

    # Build and overlay graph
    G = build_heterogeneous_graph(
        data,
        filter_forward=args.filter_forward,
        fov_deg=args.fov_deg,
        max_range=args.max_range,
        segment_length=args.segment_length,
    )
    pos = nx.get_node_attributes(G, "pos")
    node_types = nx.get_node_attributes(G, "type")
    # Exclude ego and vehicles from circular node drawing (they are rendered as rectangles above)
    display_nodes = [n for n in G.nodes if node_types.get(n) not in ("ego", "vehicle")]
    node_colors = [NODE_COLORS.get(node_types[n], "gray") for n in display_nodes]
    
    # Draw graph with different edge styles for different connection types
    if G.number_of_edges() > 0:
        # Get edge information for styling
        edges = list(G.edges())
        edge_colors = []
        edge_styles = []
        edge_widths = []
        
        for edge in edges:
            n1, n2 = edge
            type1 = node_types.get(n1, "unknown")
            type2 = node_types.get(n2, "unknown")
            
            # Style edges based on connection type
            if "ego" in [type1, type2]:
                edge_colors.append("red")
                edge_styles.append("-")
                edge_widths.append(2.0)
            elif type1 == "lane" and type2 == "lane":
                edge_colors.append("green")
                edge_styles.append("--")
                edge_widths.append(1.5)
            elif "traffic_light" in [type1, type2] or "stop_sign" in [type1, type2]:
                edge_colors.append("orange")
                edge_styles.append("-.")
                edge_widths.append(1.0)
            else:
                edge_colors.append("gray")
                edge_styles.append("-")
                edge_widths.append(0.8)
        
        # Draw nodes (excluding ego and vehicles)
        if display_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=display_nodes, ax=ax, node_color=node_colors,
                                   node_size=50, alpha=0.7)
        
        # Draw edges with different styles
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, 
                              width=edge_widths, alpha=0.6, style=edge_styles)
    else:
        # Fallback if no edges: draw only non-agent nodes
        if display_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=display_nodes, ax=ax, node_color=node_colors,
                                   node_size=50, alpha=0.6)
    
    # Add graph legend (separate from visualization legend)
    if G.number_of_nodes() > 0:
        # Create a second legend for graph elements
        graph_legend_elements = []
        
        # Get unique node types in the graph
        node_types = nx.get_node_attributes(G, "type")
        unique_types = {t for t in node_types.values() if t not in ("ego", "vehicle")}
        
        for node_type in sorted(unique_types):
            color = NODE_COLORS.get(node_type, "gray")
            graph_legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                   markerfacecolor=color, markersize=6, 
                                                   label=f'Graph: {node_type.title()}', 
                                                   linestyle='None', alpha=0.6))
        
        # Add edge legends with different styles
        if G.number_of_edges() > 0:
            graph_legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=2, 
                                                   label='Ego Connections', alpha=0.6))
            graph_legend_elements.append(plt.Line2D([0], [0], color='green', linewidth=1.5, 
                                                   linestyle='--', label='Lane Flow', alpha=0.6))
            graph_legend_elements.append(plt.Line2D([0], [0], color='orange', linewidth=1, 
                                                   linestyle='-.', label='Traffic Control', alpha=0.6))
            graph_legend_elements.append(plt.Line2D([0], [0], color='gray', linewidth=0.8, 
                                                   label='Other Connections', alpha=0.6))
        
        # Position graph legend at upper left
        if graph_legend_elements:
            graph_legend = ax.legend(handles=graph_legend_elements, loc='upper left', 
                                   bbox_to_anchor=(0.0, 1.0), frameon=True, 
                                   fancybox=True, shadow=True, fontsize=9,
                                   title="Graph Network")
            ax.add_artist(graph_legend)  # Add as separate legend
    
    # Save output
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization with graph saved to: {output_path}")
    
    # Print graph statistics
    print(f"Graph statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    if node_types:
        type_counts = {}
        for node_type in node_types.values():
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        for node_type, count in type_counts.items():
            print(f"  {node_type}: {count}")

if __name__ == "__main__":
    main()
