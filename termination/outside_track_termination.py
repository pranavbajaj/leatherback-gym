import torch
from pxr import UsdGeom

class OutsideTrackBoundsTermination:
    """
    Terminates the episode if the robot moves outside the corridor
    using vector directions to left/right closest points.
    """
    def __init__(self):
        self.left_line = None
        self.right_line = None

    def __call__(self, env, **kwargs):
        device = env.device
        num_envs = env.num_envs

        # Load curves once
        if self.left_line is None or self.right_line is None:
            stage = env.sim.stage

            def load_curve(path):
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    raise RuntimeError(f"Curve mesh not found at {path}")
                points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                return torch.tensor([(p[0], p[1]) for p in points], dtype=torch.float32, device=device)

            self.left_line = load_curve("/World/envs/env_0/Track/TrackLeftLine/ID11")
            self.right_line = load_curve("/World/envs/env_0/Track/TrackRightLine/ID19")

        # Robot positions (XY)
        robot_xy = torch.stack([
            torch.tensor(env.get_entity_position("robot", env_index=i)[:2], device=device)
            for i in range(num_envs)
        ], dim=0)

        # Track roots
        track_roots = torch.stack([
            torch.tensor(env.get_entity_position("track", env_index=i)[:2], device=device)
            for i in range(num_envs)
        ], dim=0)

        # World coordinates of lines
        left_world = self.left_line[None, :, :] + track_roots[:, None, :]
        right_world = self.right_line[None, :, :] + track_roots[:, None, :]

        # Find closest points
        left_vecs = left_world - robot_xy[:, None, :]  # vectors from robot to all left points
        right_vecs = right_world - robot_xy[:, None, :]  # vectors from robot to all right points

        min_left_idx = torch.argmin(torch.norm(left_vecs, dim=-1), dim=-1)
        min_right_idx = torch.argmin(torch.norm(right_vecs, dim=-1), dim=-1)

        closest_left_vec = left_vecs[torch.arange(num_envs), min_left_idx]  # (num_envs, 2)
        closest_right_vec = right_vecs[torch.arange(num_envs), min_right_idx]  # (num_envs, 2)

        # Normalize vectors
        left_norm = closest_left_vec / (closest_left_vec.norm(dim=-1, keepdim=True) + 1e-6)
        right_norm = closest_right_vec / (closest_right_vec.norm(dim=-1, keepdim=True) + 1e-6)

        # Dot product to check angle
        dot = (left_norm * right_norm).sum(dim=-1)  # cosine of angle between vectors

        # Angle < 90 degrees => dot > 0 => vectors point in same general direction => robot outside
        terminated = dot > 0

        return terminated