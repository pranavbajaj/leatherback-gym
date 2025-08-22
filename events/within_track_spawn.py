import torch
from pxr import UsdGeom, Gf, Usd
import numpy as np

class SpawnOnTrackEvent:
    """
    Randomly spawns the robot between left and right track boundaries.
    Properly handles coordinate transformations from USD local space to world space.
    Orients robots along the track direction using multi-point tangent estimation.
    """
    def __init__(self, threshold=0.1, num_samples=200, add_yaw_noise=True):
        self.left_line_local = None  # Track boundaries in env_0 local coordinates
        self.right_line_local = None
        self.threshold = threshold
        self.num_samples = num_samples
        self.left_resampled = None
        self.right_resampled = None
        self.add_yaw_noise = add_yaw_noise  # Whether to add small random variations to orientation
    
    def resample_curve(self, points, num_samples):
        """
        Resample a curve to have evenly spaced points using linear interpolation.
        Handles both open and closed (looped) curves.
        """
        if len(points) < 2:
            raise ValueError("Need at least 2 points to resample curve")
        
        # Calculate cumulative distances along the curve
        diffs = points[1:] - points[:-1]
        segment_lengths = torch.norm(diffs, dim=1)
        cumulative_lengths = torch.cat([torch.zeros(1, device=points.device), 
                                      torch.cumsum(segment_lengths, dim=0)])
        
        # Check if this is a closed loop
        loop_threshold = 0.1
        is_loop = torch.norm(points[-1] - points[0]) < loop_threshold
        
        if is_loop:
            points_extended = torch.cat([points, points[0:1]], dim=0)
            total_length = cumulative_lengths[-1] + torch.norm(points[-1] - points[0])
            cumulative_lengths = torch.cat([cumulative_lengths, 
                                          torch.tensor([total_length], device=points.device)])
        else:
            points_extended = points
            total_length = cumulative_lengths[-1]
        
        # Create evenly spaced parameter values
        if is_loop:
            sample_distances = torch.linspace(0, total_length, num_samples + 1, device=points.device)[:-1]
        else:
            sample_distances = torch.linspace(0, total_length, num_samples, device=points.device)
        
        # Interpolate points at the evenly spaced distances
        resampled_points = []
        for dist in sample_distances:
            idx = torch.searchsorted(cumulative_lengths, dist, right=False)
            idx = torch.clamp(idx, 0, len(points_extended) - 2)
            
            t = (dist - cumulative_lengths[idx]) / (cumulative_lengths[idx + 1] - cumulative_lengths[idx] + 1e-8)
            t = torch.clamp(t, 0, 1)
            
            interpolated_point = points_extended[idx] * (1 - t) + points_extended[idx + 1] * t
            resampled_points.append(interpolated_point)
        
        return torch.stack(resampled_points)
    
    def load_and_transform_boundary(self, stage, path, device):
        """
        Load boundary mesh and transform points from USD local space to world space,
        then compute relative to env_0 origin for replication.
        """
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Track boundary not found at {path}")
        
        mesh = UsdGeom.Mesh(prim)
        local_points = mesh.GetPointsAttr().Get()
        if local_points is None or len(local_points) == 0:
            raise RuntimeError(f"No points found in track boundary at {path}")
        
        # Get transformation from local to world coordinates
        xformable = UsdGeom.Xformable(prim)
        local_to_world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        
        # Transform points to world coordinates
        world_points = []
        for point in local_points:
            world_point = local_to_world.Transform(point)
            world_points.append([world_point[0], world_point[1]])
        
        world_points_tensor = torch.tensor(world_points, dtype=torch.float32, device=device)
        
        # Get env_0 origin to compute relative positions
        env0_prim = stage.GetPrimAtPath("/World/envs/env_0")
        if env0_prim.IsValid():
            env0_xform = UsdGeom.Xformable(env0_prim)
            env0_to_world = env0_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            env0_pos = env0_to_world.ExtractTranslation()
            env0_xy = torch.tensor([env0_pos[0], env0_pos[1]], dtype=torch.float32, device=device)
        else:
            env0_xy = torch.zeros(2, dtype=torch.float32, device=device)
        
        # Return points relative to env_0 origin
        return world_points_tensor - env0_xy
    
    def __call__(self, env, env_ids):
        device = env.device
        
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=device)
        
        num_envs = len(env_ids)
        
        # Load and transform track boundaries once
        if self.left_resampled is None or self.right_resampled is None:
            stage = env.sim.stage
            
            try:
                # Try to load track boundaries with proper transformation
                self.left_line_local = self.load_and_transform_boundary(
                    stage, "/World/envs/env_0/Track/TrackCenterLine/ID11", device
                )
                self.right_line_local = self.load_and_transform_boundary(
                    stage, "/World/envs/env_0/Track/TrackCenterLine/ID19", device
                )
                
                print(f"[SpawnOnTrackEvent] Loaded boundaries ID11 and ID19 (relative to env_0)")
                
            except RuntimeError:
                # Try alternative IDs
                potential_ids = [
                    ("ID10", "ID18"), ("ID12", "ID20"), ("ID9", "ID17"),
                    ("ID8", "ID16"), ("ID13", "ID21"), ("ID7", "ID15")
                ]
                
                loaded = False
                for left_id, right_id in potential_ids:
                    try:
                        left_path = f"/World/envs/env_0/Track/TrackCenterLine/{left_id}"
                        right_path = f"/World/envs/env_0/Track/TrackCenterLine/{right_id}"
                        self.left_line_local = self.load_and_transform_boundary(stage, left_path, device)
                        self.right_line_local = self.load_and_transform_boundary(stage, right_path, device)
                        print(f"[SpawnOnTrackEvent] Loaded boundaries {left_id} and {right_id}")
                        loaded = True
                        break
                    except RuntimeError:
                        continue
                
                if not loaded:
                    print("[SpawnOnTrackEvent] Creating fallback oval track")
                    # Fallback oval track (relative to env_0 origin)
                    t = torch.linspace(0, 2 * torch.pi, self.num_samples, device=device)
                    a, b = 10.0, 6.0
                    track_width = 3.0
                    
                    center_x = a * torch.cos(t)
                    center_y = b * torch.sin(t)
                    
                    dx_dt = -a * torch.sin(t)
                    dy_dt = b * torch.cos(t)
                    tangent_norm = torch.sqrt(dx_dt**2 + dy_dt**2)
                    
                    perp_x = -dy_dt / tangent_norm
                    perp_y = dx_dt / tangent_norm
                    
                    self.left_line_local = torch.stack([
                        center_x + perp_x * track_width/2,
                        center_y + perp_y * track_width/2
                    ], dim=1)
                    self.right_line_local = torch.stack([
                        center_x - perp_x * track_width/2,
                        center_y - perp_y * track_width/2
                    ], dim=1)
            
            # Resample boundaries
            self.left_resampled = self.resample_curve(self.left_line_local, self.num_samples)
            self.right_resampled = self.resample_curve(self.right_line_local, self.num_samples)
        
        # Get environment origins for world positioning
        env_origins = env.scene.env_origins[env_ids.long()][:, :2]  # XY only
        
        # Transform boundaries to world coordinates for each environment
        left_world = self.left_resampled[None, :, :] + env_origins[:, None, :]
        right_world = self.right_resampled[None, :, :] + env_origins[:, None, :]
        
        # Pick random indices along track
        rand_idx = torch.randint(0, self.num_samples, (num_envs,), device=device)
        
        # Get corresponding points on boundaries
        idx_range = torch.arange(num_envs, device=device)
        left_pts = left_world[idx_range, rand_idx]
        right_pts = right_world[idx_range, rand_idx]
        
        # Random interpolation between boundaries
        alpha = torch.rand(num_envs, device=device) * (1 - 2*self.threshold) + self.threshold
        spawn_positions = left_pts * (1 - alpha[:, None]) + right_pts * alpha[:, None]
        
        # Calculate orientation from track direction using finite differences
        # We'll use the track centerline to determine forward direction
        
        # Get a small step forward and backward for finite difference
        step_size = 5  # Use points a few steps away for better tangent estimation
        
        # Forward and backward indices with wrapping for closed tracks
        forward_idx = (rand_idx + step_size) % self.num_samples
        backward_idx = (rand_idx - step_size) % self.num_samples
        
        # Get the centerline points at current, forward, and backward positions
        current_center = (left_pts + right_pts) / 2.0
        
        # Get forward centerline point
        left_forward = left_world[idx_range, forward_idx]
        right_forward = right_world[idx_range, forward_idx]
        forward_center = (left_forward + right_forward) / 2.0
        
        # Get backward centerline point
        left_backward = left_world[idx_range, backward_idx]
        right_backward = right_world[idx_range, backward_idx]
        backward_center = (left_backward + right_backward) / 2.0
        
        # Calculate tangent using central finite difference
        # This gives us the direction of the track at the spawn point
        track_tangent = forward_center - backward_center
        
        # Normalize the tangent vector
        tangent_norm = track_tangent.norm(dim=-1, keepdim=True)
        track_tangent = track_tangent / (tangent_norm + 1e-8)
        
        # For very small tangent norms (straight sections or errors), use simple forward difference
        small_tangent_mask = tangent_norm.squeeze() < 1e-4
        if small_tangent_mask.any():
            # Fallback: use immediate next point
            next_idx = (rand_idx + 1) % self.num_samples
            left_next = left_world[idx_range, next_idx]
            right_next = right_world[idx_range, next_idx]
            next_center = (left_next + right_next) / 2.0
            
            fallback_tangent = next_center - current_center
            fallback_tangent = fallback_tangent / (fallback_tangent.norm(dim=-1, keepdim=True) + 1e-8)
            
            # Replace invalid tangents with fallback
            track_tangent = torch.where(
                small_tangent_mask.unsqueeze(-1),
                fallback_tangent,
                track_tangent
            )
        
        # Debug: Check if track tangent makes sense
        # You can uncomment these lines to debug orientation issues
        # print(f"[SpawnOnTrackEvent] Track tangents: {track_tangent[:5]}")  # First 5 for debugging
        # print(f"[SpawnOnTrackEvent] Tangent norms: {tangent_norm.squeeze()[:5]}")
        
        # Compute yaw angle from tangent vector
        # atan2(y, x) gives angle from positive x-axis
        yaws = torch.atan2(track_tangent[:, 1], track_tangent[:, 0])
        
        # Optional: Add small random yaw variation if configured
        if self.add_yaw_noise:
            yaw_noise_std = 0.05  # Reduced to ~3 degrees for better track following
            yaw_noise = torch.randn(num_envs, device=device) * yaw_noise_std
            yaws = yaws + yaw_noise
        
        # Create robot poses
        robot_entity = env.scene["robot"]
        z_height = 0.05
        
        # Position
        new_positions = torch.zeros((num_envs, 3), device=device)
        new_positions[:, :2] = spawn_positions
        new_positions[:, 2] = z_height
        
        # Orientation (quaternion from yaw)
        cos_half_yaw = torch.cos(yaws / 2)
        sin_half_yaw = torch.sin(yaws / 2)
        new_orientations = torch.zeros((num_envs, 4), device=device)
        new_orientations[:, 0] = cos_half_yaw  # w
        new_orientations[:, 3] = sin_half_yaw  # z
        
        # Set robot poses
        pose_data = torch.cat([new_positions, new_orientations], dim=1)
        robot_entity.write_root_pose_to_sim(pose_data, env_ids=env_ids)
        
        # Reset velocities
        zero_velocities = torch.zeros((num_envs, 6), device=device)
        robot_entity.write_root_velocity_to_sim(zero_velocities, env_ids=env_ids)
        
        print(f"[SpawnOnTrackEvent] Spawned {num_envs} robots on track")