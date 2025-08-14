import os
import math
import pickle
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
from google.protobuf.json_format import MessageToDict

def rotate_point(x, y, heading):
    cos_h = math.cos(-heading)
    sin_h = math.sin(-heading)
    return x * cos_h - y * sin_h, x * sin_h + y * cos_h

def global_to_ego(x, y, ego_x, ego_y, ego_heading):
    dx = x - ego_x
    dy = y - ego_y
    return rotate_point(dx, dy, ego_heading)

def get_lane_center_position(scenario, lane_id):
    for mf in scenario.map_features:
        if mf.WhichOneof("feature_data") == "lane" and mf.lane.id == lane_id:
            poly = mf.lane.polyline
            if poly:
                xs = [p.x for p in poly]
                ys = [p.y for p in poly]
                center_x = sum(xs)/len(xs)
                center_y = sum(ys)/len(ys)
                return (center_x, center_y)
    return None

def extract_agents_for_timestep(scenario, timestep, sdc_index):
    agents = []
    
    # Get ego state for this timestep
    ego_track = scenario.tracks[sdc_index]
    if timestep >= len(ego_track.states) or not ego_track.states[timestep].valid:
        return agents
    
    ego_state = ego_track.states[timestep]
    ego_x, ego_y, ego_heading = ego_state.center_x, ego_state.center_y, ego_state.heading
    
    for i, track in enumerate(scenario.tracks):
        if i == sdc_index:
            continue  # Skip ego vehicle
        if timestep >= len(track.states):
            continue
        state = track.states[timestep]
        if not state.valid:
            continue
            
        # Transform agent position to ego frame
        pos_ego = global_to_ego(state.center_x, state.center_y, ego_x, ego_y, ego_heading)
        
        # Map object types to readable strings
        object_type_map = {1: 'VEHICLE', 2: 'PEDESTRIAN', 3: 'CYCLIST', 4: 'OTHER'}
        agent_type = object_type_map.get(track.object_type, 'UNKNOWN')
        
        agents.append({
            "id": track.id,
            "type": agent_type,
            "position": pos_ego,
            "heading": state.heading - ego_heading,
            "length": state.length,
            "width": state.width,
            "height": state.height,
            "velocity_x": state.velocity_x,
            "velocity_y": state.velocity_y
        })
    return agents

def extract_traffic_lights_for_timestep(scenario, timestep):
    traffic_lights = []
    if timestep >= len(scenario.dynamic_map_states):
        return traffic_lights
    
    # Traffic light state mapping
    state_map = {
        0: "UNKNOWN", 1: "ARROW_STOP", 2: "ARROW_CAUTION", 3: "ARROW_GO",
        4: "STOP", 5: "CAUTION", 6: "GO", 7: "FLASHING_STOP", 8: "FLASHING_CAUTION"
    }
    
    dms = scenario.dynamic_map_states[timestep]
    for lane_state in dms.lane_states:
        # Use stop_point coordinates directly (more reliable than lane center)
        stop_point = lane_state.stop_point
        state_name = state_map.get(lane_state.state, "UNKNOWN")
        
        traffic_lights.append({
            "lane_id": lane_state.lane,
            "state": state_name,
            "position_global": (stop_point.x, stop_point.y)
        })
    
    return traffic_lights

def process_map_features(scenario, ego_pose):
    ego_x, ego_y, ego_heading = ego_pose
    map_features = []
    for mf in scenario.map_features:
        mf_type = mf.WhichOneof("feature_data")
        coords = []
        
        if mf_type and hasattr(mf, mf_type):
            feature_obj = getattr(mf, mf_type)
            
            # Handle different coordinate field types
            if hasattr(feature_obj, 'polyline') and feature_obj.polyline:
                # Lane, road_line, road_edge use polyline
                for p in feature_obj.polyline:
                    x_ego, y_ego = global_to_ego(p.x, p.y, ego_x, ego_y, ego_heading)
                    coords.append((x_ego, y_ego))
            elif hasattr(feature_obj, 'polygon') and feature_obj.polygon:
                # Crosswalk, speed_bump use polygon
                for p in feature_obj.polygon:
                    x_ego, y_ego = global_to_ego(p.x, p.y, ego_x, ego_y, ego_heading)
                    coords.append((x_ego, y_ego))
            elif hasattr(feature_obj, 'position') and feature_obj.position:
                # Stop_sign uses position
                p = feature_obj.position
                x_ego, y_ego = global_to_ego(p.x, p.y, ego_x, ego_y, ego_heading)
                coords.append((x_ego, y_ego))
            
            if coords:  # Only add if we found coordinates
                map_features.append({"type": mf_type, "vertices": coords})
    
    return map_features

def tfrecord_to_comprehensive_pkl(tfrecord_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')

    for record_idx, raw_record in enumerate(dataset):
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(raw_record.numpy())

        sdc_index = scenario.sdc_track_index

        # Pre-extract ego poses per timestep
        ego_poses = []
        for state in scenario.tracks[sdc_index].states:
            if not state.valid:
                ego_poses.append(None)
            else:
                ego_poses.append((state.center_x, state.center_y, state.heading))

        for t, ego_pose in enumerate(ego_poses):
            if ego_pose is None:
                continue

            map_features = process_map_features(scenario, ego_pose)
            traffic_lights_global = extract_traffic_lights_for_timestep(scenario, t)

            # Convert traffic lights positions to ego frame
            traffic_lights = []
            for tl in traffic_lights_global:
                x_ego, y_ego = global_to_ego(tl["position_global"][0], tl["position_global"][1],
                                             ego_pose[0], ego_pose[1], ego_pose[2])
                traffic_lights.append({
                    "lane_id": tl["lane_id"],
                    "state": tl["state"],
                    "position": (x_ego, y_ego)
                })

            agents = extract_agents_for_timestep(scenario, t, sdc_index)

            out_path = os.path.join(output_dir, f"scenario{record_idx}_t{t:03d}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump({
                    "timestep": t,
                    "ego_pose": ego_pose,
                    "map_features": map_features,
                    "traffic_lights": traffic_lights,
                    "agents": agents
                }, f)

        print(f"Processed scenario {record_idx}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord", help="Path to Scenario TFRecord",
                       default="dataset/waymo/training/uncompressed_scenario_training_training.tfrecord-00014-of-01000")
    parser.add_argument("--out_dir", help="Directory to store per-timestep PKL files",
                       default="dataset/waymo/training/hdgt_waymo_dev_tmp0")
    args = parser.parse_args()

    tfrecord_to_comprehensive_pkl(args.tfrecord, args.out_dir)
