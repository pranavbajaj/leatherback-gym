import torch
from pxr import UsdGeom, Usd

# Global storage for curve data to avoid reloading
_curve_cache = {}

def outside_track_bounds_termination(env) -> torch.Tensor:
    """
    Terminates the episode if the robot moves outside the corridor
    using vector directions to left/right closest points.
    
    Args:
        env: The environment instance.
        
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    device = env.device
    num_envs = env.num_envs
    
    # Load curves once and cache them (with proper coordinate transformation)
    cache_key = "track_curves"
    if cache_key not in _curve_cache:
        stage = env.sim.stage
        
        def load_and_transform_curve(path):
            """Load curve and transform from USD local space to world space, 
            then make relative to env_0 origin."""
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"Curve mesh not found at {path}")
            
            mesh = UsdGeom.Mesh(prim)
            local_points = mesh.GetPointsAttr().Get()
            
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
        
        try:
            # Try primary IDs
            left_line = load_and_transform_curve("/World/envs/env_0/Track/TrackCenterLine/ID11")
            right_line = load_and_transform_curve("/World/envs/env_0/Track/TrackCenterLine/ID19")
        except RuntimeError:
            # Try alternative IDs if primary ones don't exist
            alternative_ids = [
                ("ID10", "ID18"), ("ID12", "ID20"), ("ID9", "ID17"),
                ("ID8", "ID16"), ("ID13", "ID21"), ("ID7", "ID15")
            ]
            
            loaded = False
            for left_id, right_id in alternative_ids:
                try:
                    left_path = f"/World/envs/env_0/Track/TrackCenterLine/{left_id}"
                    right_path = f"/World/envs/env_0/Track/TrackCenterLine/{right_id}"
                    left_line = load_and_transform_curve(left_path)
                    right_line = load_and_transform_curve(right_path)
                    loaded = True
                    break
                except RuntimeError:
                    continue
            
            if not loaded:
                # Fallback: create simple oval track boundaries
                t = torch.linspace(0, 2 * torch.pi, 100, device=device)
                a, b = 10.0, 6.0  # Semi-major and semi-minor axes
                track_width = 3.0
                
                center_x = a * torch.cos(t)
                center_y = b * torch.sin(t)
                
                dx_dt = -a * torch.sin(t)
                dy_dt = b * torch.cos(t)
                tangent_norm = torch.sqrt(dx_dt**2 + dy_dt**2)
                
                perp_x = -dy_dt / tangent_norm
                perp_y = dx_dt / tangent_norm
                
                left_line = torch.stack([
                    center_x + perp_x * track_width/2,
                    center_y + perp_y * track_width/2
                ], dim=1)
                right_line = torch.stack([
                    center_x - perp_x * track_width/2,
                    center_y - perp_y * track_width/2
                ], dim=1)
        
        _curve_cache[cache_key] = {
            'left_line': left_line,
            'right_line': right_line
        }
    
    left_line = _curve_cache[cache_key]['left_line']
    right_line = _curve_cache[cache_key]['right_line']
    
    # Robot positions (XY) in world coordinates
    robot_entity = env.scene["robot"]
    robot_positions = robot_entity.data.root_pos_w[:, :2]
    
    # Get environment origins for proper world positioning
    env_origins = env.scene.env_origins[:, :2]
    
    # Transform boundaries to world coordinates for each environment
    left_world = left_line[None, :, :] + env_origins[:, None, :]
    right_world = right_line[None, :, :] + env_origins[:, None, :]
    
    # Find closest points
    left_vecs = left_world - robot_positions[:, None, :]
    right_vecs = right_world - robot_positions[:, None, :]
    
    min_left_idx = torch.argmin(torch.norm(left_vecs, dim=-1), dim=-1)
    min_right_idx = torch.argmin(torch.norm(right_vecs, dim=-1), dim=-1)
    
    closest_left_vec = left_vecs[torch.arange(num_envs), min_left_idx]
    closest_right_vec = right_vecs[torch.arange(num_envs), min_right_idx]
    
    # Normalize vectors
    left_norm = closest_left_vec / (closest_left_vec.norm(dim=-1, keepdim=True) + 1e-6)
    right_norm = closest_right_vec / (closest_right_vec.norm(dim=-1, keepdim=True) + 1e-6)
    
    # Dot product to check angle
    dot = (left_norm * right_norm).sum(dim=-1)
    
    # Angle < 90 degrees => dot > 0 => vectors point in same general direction => robot outside
    terminated = dot > 0
    
    return terminated