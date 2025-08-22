from pxr import UsdGeom, Usd
import torch
import numpy as np

class DistanceToCenterlineReward:
    def __init__(self):
        self.centerline_xy_world = None
        self.centerline_xy_local = None
        
    def __call__(self, env):
        if self.centerline_xy_world is None:
            stage = env.sim.stage
            device = env.device
            
            # Get mesh prim and its points (in local coordinates)
            mesh_prim = stage.GetPrimAtPath("/World/envs/env_0/Track/TrackCenterLine/ID3")
            if not mesh_prim.IsValid():
                raise RuntimeError("Centerline mesh not found!")
            
            mesh = UsdGeom.Mesh(mesh_prim)
            local_points = mesh.GetPointsAttr().Get()
            
            # Get transformation from ID3 local space to world space
            xformable = UsdGeom.Xformable(mesh_prim)
            # Use default time code (usually Usd.TimeCode.Default())
            local_to_world_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            
            # Transform points to world coordinates
            world_points = []
            for point in local_points:
                world_point = local_to_world_matrix.Transform(point)
                world_points.append([world_point[0], world_point[1]])  # Extract XY
            
            # Store the world-space centerline for env_0
            self.centerline_xy_world = torch.tensor(world_points, dtype=torch.float32, device=device)
            
            # Also compute the centerline relative to env_0 origin for replication
            # Get env_0's transform
            env0_prim = stage.GetPrimAtPath("/World/envs/env_0")
            env0_xform = UsdGeom.Xformable(env0_prim)
            env0_to_world = env0_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            
            # Get env_0 position in world
            env0_world_pos = env0_to_world.ExtractTranslation()
            
            # Store centerline relative to env_0 origin
            self.centerline_xy_local = self.centerline_xy_world - torch.tensor(
                [env0_world_pos[0], env0_world_pos[1]], 
                dtype=torch.float32, 
                device=device
            )
        
        device = env.device
        
        # Get robot positions in world coordinates
        robot_entity = env.scene["robot"]
        robot_positions = robot_entity.data.root_pos_w[:, :2]  # World XY positions for all robots
        
        # Get environment origins (where each env_i is positioned in world)
        env_origins = env.scene.env_origins[:, :2]  # World XY positions of each environment
        
        # Replicate centerline for each environment
        # The centerline pattern is the same for each env, just offset by env origin
        world_centerlines = self.centerline_xy_local[None, :, :] + env_origins[:, None, :]
        
        # Calculate distances from each robot to its corresponding centerline
        diffs = robot_positions[:, None, :] - world_centerlines
        dists = torch.norm(diffs, dim=-1)
        min_dists, _ = torch.min(dists, dim=-1)
        
        # Calculate reward: closer to centerline = higher reward
        # Using exponential decay with adjustable scale factor
        scale_factor = 2.0  # Adjust this to control reward sensitivity
        rewards = torch.exp(-scale_factor * min_dists)
        
        return rewards


# Alternative implementation with more features
class AdvancedDistanceToCenterlineReward:
    def __init__(self, scale_factor=2.0, max_distance=5.0, use_squared_distance=False):
        """
        Advanced centerline reward with configurable parameters.
        
        Args:
            scale_factor: Controls the decay rate of the reward
            max_distance: Maximum distance beyond which reward is zero
            use_squared_distance: If True, uses squared distance for smoother gradients
        """
        self.centerline_xy_world = None
        self.centerline_xy_local = None
        self.scale_factor = scale_factor
        self.max_distance = max_distance
        self.use_squared_distance = use_squared_distance
        
    def _initialize_centerline(self, env):
        """Initialize centerline data with proper coordinate transformations."""
        stage = env.sim.stage
        device = env.device
        
        # Get mesh prim
        mesh_prim = stage.GetPrimAtPath("/World/envs/env_0/Track/TrackCenterLine/ID3")
        if not mesh_prim.IsValid():
            # Try alternative paths
            alt_paths = [
                "/World/envs/env_0/Track/TrackCenterLine",
                "/World/envs/env_0/TrackCenterLine/ID3",
                "/World/envs/env_0/track/centerline"
            ]
            for path in alt_paths:
                mesh_prim = stage.GetPrimAtPath(path)
                if mesh_prim.IsValid():
                    break
            
            if not mesh_prim.IsValid():
                raise RuntimeError(f"Centerline mesh not found at any expected path!")
        
        mesh = UsdGeom.Mesh(mesh_prim)
        local_points = mesh.GetPointsAttr().Get()
        
        # Transform to world coordinates
        xformable = UsdGeom.Xformable(mesh_prim)
        local_to_world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        
        # Efficient numpy-based transformation
        np_matrix = np.array(local_to_world)
        np_points = np.array([[p[0], p[1], p[2]] for p in local_points])
        
        # Apply transformation: points @ rotation.T + translation
        world_points_3d = np_points @ np_matrix[:3, :3].T + np_matrix[3, :3]
        world_points_xy = world_points_3d[:, :2]
        
        self.centerline_xy_world = torch.tensor(world_points_xy, dtype=torch.float32, device=device)
        
        # Get env_0 origin for relative positioning
        env0_prim = stage.GetPrimAtPath("/World/envs/env_0")
        if env0_prim.IsValid():
            env0_xform = UsdGeom.Xformable(env0_prim)
            env0_to_world = env0_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            env0_pos = env0_to_world.ExtractTranslation()
            env0_xy = torch.tensor([env0_pos[0], env0_pos[1]], dtype=torch.float32, device=device)
        else:
            # Fallback: assume env_0 is at origin
            env0_xy = torch.zeros(2, dtype=torch.float32, device=device)
        
        self.centerline_xy_local = self.centerline_xy_world - env0_xy
        
    def __call__(self, env):
        if self.centerline_xy_local is None:
            self._initialize_centerline(env)
        
        # Get robot world positions
        robot_entity = env.scene["robot"]
        robot_positions = robot_entity.data.root_pos_w[:, :2]
        
        # Get environment origins
        env_origins = env.scene.env_origins[:, :2]
        
        # Replicate centerline for each environment
        world_centerlines = self.centerline_xy_local[None, :, :] + env_origins[:, None, :]
        
        # Compute distances
        diffs = robot_positions[:, None, :] - world_centerlines
        
        if self.use_squared_distance:
            # Squared distance for smoother gradients
            dists_squared = torch.sum(diffs ** 2, dim=-1)
            min_dists_squared, _ = torch.min(dists_squared, dim=-1)
            
            # Reward based on squared distance
            rewards = torch.exp(-self.scale_factor * min_dists_squared)
        else:
            # Euclidean distance
            dists = torch.norm(diffs, dim=-1)
            min_dists, closest_indices = torch.min(dists, dim=-1)
            
            # Clamp distances to max_distance
            clamped_dists = torch.clamp(min_dists, max=self.max_distance)
            
            # Normalized reward (1 at centerline, 0 at max_distance)
            normalized_dists = clamped_dists / self.max_distance
            rewards = 1.0 - normalized_dists
            
            # Optional: Apply exponential shaping
            rewards = torch.exp(-self.scale_factor * clamped_dists)
        
        return rewards
    
    def get_debug_info(self, env):
        """Get debug information for visualization."""
        if self.centerline_xy_local is None:
            self._initialize_centerline(env)
        
        robot_entity = env.scene["robot"]
        robot_positions = robot_entity.data.root_pos_w[:, :2]
        env_origins = env.scene.env_origins[:, :2]
        
        world_centerlines = self.centerline_xy_local[None, :, :] + env_origins[:, None, :]
        
        diffs = robot_positions[:, None, :] - world_centerlines
        dists = torch.norm(diffs, dim=-1)
        min_dists, closest_indices = torch.min(dists, dim=-1)
        
        # Get closest points on centerline for each robot
        batch_indices = torch.arange(len(robot_positions), device=robot_positions.device)
        closest_points = world_centerlines[batch_indices, closest_indices]
        
        return {
            "min_distances": min_dists,
            "closest_points": closest_points,
            "closest_indices": closest_indices,
            "robot_positions": robot_positions
        }

# from pxr import UsdGeom
# import torch

# class DistanceToCenterlineReward:
#     def __init__(self):
#         self.centerline_xy = None
    
#     def __call__(self, env):
#         if self.centerline_xy is None:
#             stage = env.sim.stage
#             device = env.device
#             mesh_prim = stage.GetPrimAtPath("/World/envs/env_0/Track/TrackCenterLine/ID3")
#             if not mesh_prim.IsValid():
#                 raise RuntimeError("Centerline mesh not found!")
#             mesh = UsdGeom.Mesh(mesh_prim)
#             points = mesh.GetPointsAttr().Get()
#             self.centerline_xy = torch.tensor([(p[0], p[1]) for p in points], dtype=torch.float32, device=device)
        
#         centerline_xy = self.centerline_xy
#         device = env.device
        
#         # Get robot positions using proper Isaac Lab scene access
#         robot_entity = env.scene["robot"]
#         robot_positions = robot_entity.data.root_pos_w[:, :2]  # Get XY positions for all robots
        
#         # Get track positions using environment origins
#         track_positions = env.scene.env_origins[:, :2]  # Use environment origins since tracks are positioned there
        
#         # Transform centerlines to world coordinates for each environment
#         world_centerlines = centerline_xy[None, :, :] + track_positions[:, None, :]  # (num_envs, N_points, 2)
        
#         # Calculate distances from robot to centerline points
#         diffs = robot_positions[:, None, :] - world_centerlines
#         dists = torch.norm(diffs, dim=-1)
#         min_dists, _ = torch.min(dists, dim=-1)
        
#         # Calculate exponential reward based on distance to centerline
#         rewards = torch.exp(-min_dists)
        
#         return rewards