import math 
import torch

import isaaclab.envs.mdp as mdp 
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass 

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import AssetBaseCfg 
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

from rewards.track_follow_reward import DistanceToCenterlineReward
from termination.outside_track_termination import outside_track_bounds_termination
from events.within_track_spawn import SpawnOnTrackEvent


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
        pos=(0.0, 0.0, 0.05),
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


@configclass 
class LeatherbackSceneCfg(InteractiveSceneCfg): 
    ground = AssetBaseCfg(prim_path="/World/gridroom_curved", spawn=sim_utils.GroundPlaneCfg())
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


    chase_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Rigid_Bodies/Chassis/Camera_Chase",  # Path to existing camera
        spawn=None,  # Camera already exists in USD
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
    )


@configclass 
class ActionsCfg: 

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
            "Wheel__Knuckle__Front_Right": (float(-50),float(50)) ,
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

        camera_obs = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("chase_camera")},
        )        

        def __post_init__(self) -> None: 
            self.enable_corruption = False
            self.concatenate_terms = False 

    policy: PolicyCfg = PolicyCfg() 


@configclass
class EventCfg: 
    """Configuration for events."""

    spawn_robot = EventTerm(
        func=SpawnOnTrackEvent(threshold=0.1),
        mode="reset",
        params={},  # any extra parameters
    )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    ## Follow center line reward 
    # distance_to_centerline: DistanceToCenterlineReward = DistanceToCenterlineReward()


    distance_to_centerline = RewTerm(
        func=DistanceToCenterlineReward(), 
        weight=1.0
        )

    ## Positive reward for cars speed 
    car_vel = RewTerm(
        func = mdp.joint_vel_l1, 
        weight = 0.1, 
        params={"asset_cfg": SceneEntityCfg("robot", 
                                            joint_names=[
                                                "Wheel__Knuckle__Front_Left",
                                                "Wheel__Knuckle__Front_Right",
                                                "Wheel__Upright__Rear_Right",
                                                "Wheel__Upright__Rear_Left",
                                            ])}, 
        )


@configclass
class TerminationsCfg:
    
    time_out = DoneTerm(
        func=mdp.time_out, 
        time_out=True
        )

    outside_track_bounds = DoneTerm(
        func=outside_track_bounds_termination, 
        params={}
    )




@configclass 
class LeatherbackEnvCfg(ManagerBasedRLEnvCfg): 
    """Configuration for the LeatherbackEnvCfg environment."""


    scene: LeatherbackSceneCfg = LeatherbackSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

