import argparse

from isaaclab.app import AppLauncher 

parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math 
import torch
import random 
 
import isaaclab.envs.mdp as mdp 
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass 

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import AssetBaseCfg 
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils


from pxr import UsdGeom, Gf

### Leatherback articulation config ###
RC_CONFIG = ArticulationCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Leatherback/leatherback.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.6352, 0.0, 0.05),
        joint_pos={
            "Wheel__Knuckle__Front_Left": 0.0,
            "Wheel__Knuckle__Front_Right": 0.0,
            "Wheel__Upright__Rear_Right": 0.0,
            "Wheel__Upright__Rear_Left": 0.0,
            "Knuckle__Upright__Front_Right": 0.0,
            "Knuckle__Upright__Front_Left": 0.0,
            "Shock__Rear_Right": -0.045, 
            "Shock__Rear_Left": -0.045,
            "Shock__Front_Right": 0.045,
            "Shock__Front_Left": 0.045, 
        },
    ),
    actuators={
        "throttle": ImplicitActuatorCfg(
            joint_names_expr=["Wheel.*"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=100000.0,
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["Knuckle__Upright__Front.*"],
            effort_limit=40000.0,
            velocity_limit=100.0,
            stiffness=1000.0,
            damping=0.0,
        ),
    },
)

### Leatherback articulation config end ###
@configclass 
class LeatherbackSceneCfg(InteractiveSceneCfg): 
    gound = AssetBaseCfg(prim_path="/World/gridroom_curved", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    track = AssetBaseCfg(
        prim_path = "{ENV_REGEX_NS}/Track",
        spawn = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Jetracer/jetracer_track_solid.usd"
        ),
    )

    robot = RC_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass 
class ActionsCfg: 
    # throttle_dof_name = [
    #     "Wheel__Knuckle__Front_Left",
    #     "Wheel__Knuckle__Front_Right",
    #     "Wheel__Upright__Rear_Right",
    #     "Wheel__Upright__Rear_Left",
    # ]
    # steering_dof_name = [
    #     "Knuckle__Upright__Front_Right",
    #     "Knuckle__Upright__Front_Left",
    # ]

    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["Knuckle__Upright__Front.*"], 
        scale=0.1, 
        clip={
            "Knuckle__Upright__Front_Right": (float(-0.75),float(0.75)),
            "Knuckle__Upright__Front_Left": (float(-0.75),float(0.75)),
        }
    )

    joint_velocity = mdp.JointVelocityActionCfg(
        asset_name="robot", 
        joint_names=["Wheel.*"], 
        scale = 10.0,
        clip={
            "Wheel__Knuckle__Front_Left": (float(-50),float(50)),
            "Wheel__Knuckle__Front_Right": (float(-50),float(50)),
            "Wheel__Upright__Rear_Right": (float(-50),float(50)),
            "Wheel__Upright__Rear_Left": (float(-50),float(50)),
        }, 
    )
    


@configclass
class ObservationsCfg: 


    @configclass 
    class PolicyCfg(ObsGroup): 


        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params = {
                "asset_cfg": SceneEntityCfg(
                    name = 'robot', 
                    joint_names = [
                        "Knuckle__Upright__Front_Right",
                        "Knuckle__Upright__Front_Left",
                    ]
                )
            },
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params = {
                "asset_cfg": SceneEntityCfg(
                    name = 'robot', 
                    joint_names = [
                        "Wheel__Knuckle__Front_Left",
                        "Wheel__Knuckle__Front_Right",
                        "Wheel__Upright__Rear_Right",
                        "Wheel__Upright__Rear_Left",
                    ]
                )
            },
        )

        def __post_init__(self) -> None: 
            self.enable_corruption = False
            self.concatenate_terms = True 

    policy: PolicyCfg = PolicyCfg() 


@configclass 
class EventCfg: 

    reset_robot = EventTerm(
        func = mdp.reset_scene_to_default, 
        mode = "reset",
    )
    
@configclass 
class LeatherbackEnvCfg(ManagerBasedEnvCfg): 

    scene = LeatherbackSceneCfg(num_envs = 4, env_spacing = 40)

    observations = ObservationsCfg()
    actions = ActionsCfg() 
    events = EventCfg() 

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200H


def main(): 

    env_cfg = LeatherbackEnvCfg() 
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedEnv(cfg = env_cfg)

    # print("\n[INFO] Listing all prim paths in stage:")
    # for prim in env.sim.stage.Traverse():
    #     print(prim.GetPath())


    # -----------------------------------------------
    # Extract centerline from USD stage
    # -----------------------------------------------
    from pxr import UsdGeom
    import numpy as np
    import torch

    mesh_path = "/World/envs/env_0/Track/TrackCenterLine/ID19"
    mesh_prim = env.sim.stage.GetPrimAtPath(mesh_path)

    if not mesh_prim.IsValid():
        raise RuntimeError(f"[ERROR] Mesh prim at {mesh_path} is not valid!")

    # Access the mesh
    mesh = UsdGeom.Mesh(mesh_prim)

    # Get vertices
    points = mesh.GetPointsAttr().Get()  # returns a list of Gf.Vec3f
    centerline_np = np.array([(p[0], p[1]) for p in points], dtype=np.float32)  # take x, y only
    centerline = torch.tensor(centerline_np, dtype=torch.float32)

    print("[INFO] Centerline loaded from mesh! Shape:", centerline.shape)
    print(centerline[:10, :])
    # # ----------------------------------------------

    count = 0

    curr_action = torch.zeros_like(env.action_manager.action, dtype=torch.float32)
    print(curr_action.size())
    
    num_envs = curr_action.shape[0]
    
    curr_action[:, :2] = torch.randint(low=-4, high=5, size=(num_envs, 2), dtype=torch.float32)    # Position, consider scaling factor of 0.1
    curr_action[:, 2:] = torch.randint(low=-4, high=5, size=(num_envs, 4), dtype=torch.float32)

    while simulation_app.is_running(): 
        with torch.inference_mode(): 

            if count % 300 == 0: 
                count = 0 
                env.reset() 
                print("-" * 80)
                print("[INFO]: Resetting environment...")
             # Velocity, consider scaling factor of 10
            obs, _ = env.step(curr_action)
            # print(obs["policy"][0])
            count += 1

    env.close() 

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()