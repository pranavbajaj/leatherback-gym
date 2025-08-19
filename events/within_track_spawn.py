import torch
from pxr import UsdGeom

class SpawnOnTrackEvent:
    """
    Randomly spawns the robot between left and right track lines.
    Orientation aligned to mean tangent direction.
    Vectorized across environments.
    """
    def __init__(self, threshold=0.1):
        self.left_line = None
        self.right_line = None
        self.threshold = threshold

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

        # Track roots
        track_roots = torch.stack([
            torch.tensor(env.get_entity_position("track", env_index=i)[:2], device=device)
            for i in range(num_envs)
        ], dim=0)

        left_world = self.left_line[None, :, :] + track_roots[:, None, :]
        right_world = self.right_line[None, :, :] + track_roots[:, None, :]

        # Pick random indices along track
        max_idx = min(left_world.shape[1], right_world.shape[1]) - 2
        rand_idx = torch.randint(0, max_idx + 1, (num_envs,), device=device)

        idx_range = torch.arange(num_envs, device=device)
        left_pts = left_world[idx_range, rand_idx]
        right_pts = right_world[idx_range, rand_idx]

        # Random interpolation between left and right
        alpha = torch.rand(num_envs, device=device) * (1 - 2*self.threshold) + self.threshold
        spawn_positions = left_pts * (1 - alpha[:, None]) + right_pts * alpha[:, None]

        # Tangent vectors for orientation
        left_tangent = left_world[idx_range, rand_idx + 1] - left_pts
        right_tangent = right_world[idx_range, rand_idx + 1] - right_pts
        mean_tangent = (left_tangent + right_tangent) / 2.0
        mean_tangent = mean_tangent / (mean_tangent.norm(dim=-1, keepdim=True) + 1e-6)

        # Compute yaw angle
        yaws = torch.atan2(mean_tangent[:, 1], mean_tangent[:, 0])

        # Set robot positions and orientation
        z_height = 0.05
        for i in range(num_envs):
            pos = spawn_positions[i]
            yaw = yaws[i]
            env.set_entity_position("robot", env_index=i, pos=(pos[0].item(), pos[1].item(), z_height))
            env.set_entity_orientation("robot", env_index=i, yaw=yaw.item())