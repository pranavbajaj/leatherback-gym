from pxr import UsdGeom
import torch
# import torch.nn.functional as F
# from isaaclab.managers import BaseRewardTerm
# from isaaclab.utils import configclass



class DistanceToCenterlineReward:
    def __init__(self):
        self.centerline_xy = None

    def __call__(self, env, **kwargs):
        if self.centerline_xy is None:
            stage = env.sim.stage
            device = env.device
            mesh_prim = stage.GetPrimAtPath("/World/envs/env_0/Track/TrackCenterLine/ID3")
            if not mesh_prim.IsValid():
                raise RuntimeError("Centerline mesh not found!")
            mesh = UsdGeom.Mesh(mesh_prim)
            points = mesh.GetPointsAttr().Get()
            self.centerline_xy = torch.tensor([(p[0], p[1]) for p in points], dtype=torch.float32, device=device)

        centerline_xy = self.centerline_xy
        device = env.device

        robot_positions = torch.stack([
            torch.tensor(env.get_entity_position("robot", env_index=i)[:2], device=device)
            for i in range(env.num_envs)
        ], dim=0)

        track_roots = torch.stack([
            torch.tensor(env.get_entity_position("track", env_index=i)[:2], device=device)
            for i in range(env.num_envs)
        ], dim=0)

        world_centerlines = centerline_xy[None, :, :] + track_roots[:, None, :]  # (num_envs, N_points, 2)
        diffs = robot_positions[:, None, :] - world_centerlines
        dists = torch.norm(diffs, dim=-1)
        min_dists, _ = torch.min(dists, dim=-1)
        rewards = torch.exp(-min_dists)

        return rewards



#### BaseRewardTerm implementation 

# @configclass
# class DistanceToCenterlineReward(BaseRewardTerm):
#     """
#     Reward based on distance of robot to the track centerline.
#     Fully vectorized for multiple environments. 
#     Assumes the track mesh is identical across envs; only root positions differ.
#     """

#     robot_name: str = "robot"
#     mesh_path: str = "{ENV_REGEX_NS}/Track/TrackCenterLine/ID3"

#     def initialize(self, env):
#         """
#         Load the track mesh once and store it.
#         """
#         stage = env.sim.stage
#         device = env.device

#         # Pick the first environment to load the mesh
#         mesh_path_env0 = self.mesh_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
#         mesh_prim = stage.GetPrimAtPath(mesh_path_env0)
#         if not mesh_prim.IsValid():
#             raise RuntimeError(f"[ERROR] Mesh prim at {mesh_path_env0} is not valid!")

#         mesh = UsdGeom.Mesh(mesh_prim)
#         points = mesh.GetPointsAttr().Get()  # list of Gf.Vec3f
#         centerline_xy = torch.tensor([(p[0], p[1]) for p in points], dtype=torch.float32, device=device)
#         self.centerline_xy = centerline_xy  # (N_points, 2)

#     def compute(self, env, **kwargs):
#         """
#         Compute rewards for all environments in a batched, vectorized manner.
#         """
#         num_envs = env.num_envs
#         device = env.device

#         # Get robot positions (num_envs, 3) -> take XY
#         robot_positions = torch.stack([
#             torch.tensor(env.get_entity_position(self.robot_name, env_index=i)[:2], device=device)
#             for i in range(num_envs)
#         ], dim=0)  # (num_envs, 2)

#         # Get track root positions (num_envs, 3) -> XY
#         track_roots = torch.stack([
#             torch.tensor(env.get_entity_position("track", env_index=i)[:2], device=device)
#             for i in range(num_envs)
#         ], dim=0)  # (num_envs, 2)

#         # Broadcast centerline to each env based on root position
#         world_centerlines = self.centerline_xy[None, :, :] + track_roots[:, None, :]  # (num_envs, N_points, 2)

#         # Compute distances to centerline
#         diffs = robot_positions[:, None, :] - world_centerlines  # (num_envs, N_points, 2)
#         dists = torch.norm(diffs, dim=-1)  # (num_envs, N_points)
#         min_dists, _ = torch.min(dists, dim=-1)  # (num_envs,)

#         # Convert distance to reward (closer = higher)
#         rewards = torch.exp(-min_dists)  # (num_envs,)
#         return rewards




#### Function based implementation
# def distance_to_centerline_fn(env, **kwargs):
#     """
#     Compute distance-to-centerline reward from robot to static track centerline.
#     """
#     stage = env.sim.stage
#     device = env.device

#     # Load mesh from env_0 only once
#     if not hasattr(env, "_centerline_xy"):
#         mesh_prim = stage.GetPrimAtPath("/World/envs/env_0/Track/TrackCenterLine/ID3")
#         if not mesh_prim.IsValid():
#             raise RuntimeError("Centerline mesh not found!")
#         mesh = UsdGeom.Mesh(mesh_prim)
#         points = mesh.GetPointsAttr().Get()
#         env._centerline_xy = torch.tensor([(p[0], p[1]) for p in points], dtype=torch.float32, device=device)

#     centerline_xy = env._centerline_xy  # (N_points, 2)

#     robot_positions = torch.stack([
#         torch.tensor(env.get_entity_position("robot", env_index=i)[:2], device=device)
#         for i in range(env.num_envs)
#     ], dim=0)

#     track_roots = torch.stack([
#         torch.tensor(env.get_entity_position("track", env_index=i)[:2], device=device)
#         for i in range(env.num_envs)
#     ], dim=0)

#     world_centerlines = centerline_xy[None, :, :] + track_roots[:, None, :]  # (num_envs, N_points, 2)
#     diffs = robot_positions[:, None, :] - world_centerlines
#     dists = torch.norm(diffs, dim=-1)
#     min_dists, _ = torch.min(dists, dim=-1)
#     rewards = torch.exp(-min_dists)

#     return rewards