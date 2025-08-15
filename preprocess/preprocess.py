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

# ---------------- Ego-lane detection helpers ----------------

def _wrap_angle(angle):
	while angle > math.pi:
		angle -= 2 * math.pi
	while angle < -math.pi:
		angle += 2 * math.pi
	return angle

def _nearest_point_on_polyline(poly_points, px, py):
	"""Return (closest_x, closest_y, segment_heading, distance) to point (px,py)."""
	best_d2 = float('inf')
	best_pt = (None, None)
	best_heading = None
	def _as_xy(pt):
		return (pt.x, pt.y) if hasattr(pt, 'x') else (pt[0], pt[1])
	for i in range(len(poly_points) - 1):
		ax, ay = _as_xy(poly_points[i])
		bx, by = _as_xy(poly_points[i + 1])
		vx, vy = bx - ax, by - ay
		seg_len2 = vx * vx + vy * vy
		if seg_len2 == 0:
			cx, cy = ax, ay
		else:
			t = ((px - ax) * vx + (py - ay) * vy) / seg_len2
			t = max(0.0, min(1.0, t))
			cx, cy = ax + t * vx, ay + t * vy
		dx, dy = px - cx, py - cy
		d2 = dx * dx + dy * dy
		if d2 < best_d2:
			best_d2 = d2
			best_pt = (cx, cy)
			best_heading = math.atan2(vy, vx) if seg_len2 > 0 else best_heading
	return best_pt[0], best_pt[1], best_heading, math.sqrt(best_d2)

def find_ego_lane_id(scenario, ego_pose, max_centerline_distance_m=4.0, max_heading_diff_rad=math.radians(45.0)):
	"""Find lane id best matching ego pose (nearest centerline with aligned heading)."""
	ego_x, ego_y, ego_heading = ego_pose
	best_lane_id = None
	best_score = float('inf')
	for mf in scenario.map_features:
		if mf.WhichOneof("feature_data") != "lane":
			continue
		lane = mf.lane
		if not lane.polyline:
			continue
		cx, cy, lane_heading, dist = _nearest_point_on_polyline(lane.polyline, ego_x, ego_y)
		if dist > max_centerline_distance_m:
			continue
		if lane_heading is None:
			continue
		heading_diff = abs(_wrap_angle(ego_heading - lane_heading))
		if heading_diff > max_heading_diff_rad:
			continue
		score = dist + 0.5 * heading_diff
		if score < best_score:
			best_score = score
			best_lane_id = mf.id
	return best_lane_id

# ------------------------------------------------------------

def get_lane_center_position(scenario, lane_id):
	for mf in scenario.map_features:
		if mf.WhichOneof("feature_data") == "lane" and mf.id == lane_id:
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
				entry = {"type": mf_type, "vertices": coords}
				if mf_type == "lane":
					lane_id = getattr(feature_obj, "id", None)
					if lane_id is None or lane_id == 0:
						lane_id = getattr(mf, "id", None)
					if lane_id is not None:
						entry["id"] = int(lane_id)
				map_features.append(entry)
	
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

			# Ego kinematics and orientation in global frame
			ego_state = scenario.tracks[sdc_index].states[t]
			ego_vx, ego_vy = ego_state.velocity_x, ego_state.velocity_y
			ego_speed = math.hypot(ego_vx, ego_vy)
			ego_heading_abs = ego_state.heading

			# Determine ego lane id in global frame for this timestep
			ego_lane_id = find_ego_lane_id(scenario, ego_pose)

			out_path = os.path.join(output_dir, f"scenario{record_idx}_t{t:03d}.pkl")
			with open(out_path, "wb") as f:
				pickle.dump({
					"timestep": t,
					"ego_pose": ego_pose,
					"ego_velocity": (ego_vx, ego_vy),
					"ego_speed": ego_speed,
					"ego_orientation": ego_heading_abs,
					"ego_lane_id": ego_lane_id,
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
