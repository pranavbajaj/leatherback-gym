
import math 
import torch
import numpy as np 
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
from events.within_centerline_spawn import SpawnOnTrackEvent
from rewards.speed_reward import speed_magnitude_reward_local


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
        update_period=0.0, # 0.0 make sure it updates every sim step.
        height=240,
        width=320,
        data_types=["rgb"],
    )

@configclass 
class ActionsCfg: 

    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["Knuckle__Upright__Front.*"], 
        scale=0.6, 
        clip={
            "Knuckle__Upright__Front_Right": (float(-0.6),float(0.6)),
            "Knuckle__Upright__Front_Left": (float(-0.6),float(0.6)),
        }
    )

    joint_velocity = mdp.JointVelocityActionCfg(
        asset_name="robot", 
        joint_names=["Wheel.*"], 
        scale = 30.0, 
        clip={
            "Wheel__Knuckle__Front_Left": (float(-30),float(30)),   ## Can set to [-30, 30] 
            "Wheel__Knuckle__Front_Right": (float(-30),float(30)),
            "Wheel__Upright__Rear_Right": (float(-30),float(30)),
            "Wheel__Upright__Rear_Left": (float(-30),float(30)),
        }, 
    )
    
@configclass
class ObservationsCfg:

    @configclass 
    class PolicyCfg(ObsGroup): 

        # Steering angle
        steering_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", 
                                              joint_names=["Knuckle__Upright__Front.*"])},
            scale=1.0/0.6
        )

        # Wheel velocities
        # wheel_vel = ObsTerm(
        #     func=mdp.joint_vel_rel,
        #     params={"asset_cfg": SceneEntityCfg("robot", 
        #                                       joint_names=["Wheel.*"])},
        #     scale=1.0/30.0
        # )

        camera_obs = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("chase_camera")},
        )        

       # Local frame velocities (what the robot "feels")
        root_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,  # Local frame
            params={"asset_cfg": SceneEntityCfg("robot")},
            scale=0.1
        )
        # Robot learns: [forward, lateral, vertical] speeds
        
        # root_ang_vel = ObsTerm(
        #     func=mdp.base_ang_vel,  # Local frame
        #     params={"asset_cfg": SceneEntityCfg("robot")},
        #     scale=0.1
        # )
        # # Robot learns: [roll, pitch, yaw] rates
        
        # Previous actions (action history)
        # last_action = ObsTerm(
        #     func=mdp.last_action,
        # )

        # generated_commands = ObsTerm(
        #     func=mdp.generated_commands,
        # )

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

    
    # Reward for staying alive 
    # Testing done: Trying to keep reward in 0.01 magnitude
    alive = RewTerm(
        func=mdp.is_alive,  # Returns 1 if not terminated
        weight=1
    )

    ## Follow center line reward 
    # distance_to_centerline: DistanceToCenterlineReward = DistanceToCenterlineReward()
    # Testing done: Tyring to keep reward in 0.1 magnitude 
    distance_to_centerline = RewTerm(
        func=DistanceToCenterlineReward(), 
        weight=10.0
        )
    
    # # Primary: Forward speed in robot's local frame
    # # Rewards to Forward speed 
    # # Penalty to Backward speed
    # Testing done: Trying to keep reward in 0.1 magnitude level  
    forward_velocity = RewTerm(
        func=speed_magnitude_reward_local,  
        weight=10.0
    )


    # # ========== PENALTIES ==========
    
    # # Joint acceleration penalty (smoothness)
    ## Tested for how change is different scales of acceleration affects the penalty. 
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-0.00001,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )


    # # Steering limits penalty
    ## Not able to find the joint limits.
    ### Not adding beacuse joint max clip is lower than joint limit. 
    # joint_limits = RewTerm(
    #     func=mdp.joint_pos_limits,
    #     weight=-0.15,
    #     params={"asset_cfg": SceneEntityCfg("robot",
    #                                        joint_names=["Knuckle__Upright__Front_Right",
    #                                                    "Knuckle__Upright__Front_Left"])}
    # )

    # # Penalty for sudden steering change 
    ## Testing done. 
    steering_change = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-100,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["Knuckle__Upright__Front.*"]
            )
        }
    )

    termination = RewTerm(
        func=mdp.is_terminated,  # Returns 1 when terminated
        weight=-1200.0  # Large negative weight
    )

@configclass
class TerminationsCfg:
    
    # If you are adding termination penality in reward, don't add time_out ?
    # time_out = DoneTerm(
    #     func=mdp.time_out, 
    #     time_out=True
    # )

    # Outside track bounds
    outside_track_bounds = DoneTerm(
        func=outside_track_bounds_termination, 
        params={}
    )

    # # Car flipped over 
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Chassis"),
    #     }
    # )

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
        # self.sim.dt = 1 / 120
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation



# 4. Curriculum learning configuration 
# @configclass
# class CurriculumCfg:
#     """Curriculum learning settings."""
    
#     # Start with easier settings
#     initial_max_speed = 5.0
#     final_max_speed = 50.0
    
#     initial_track_difficulty = 0.2  # Start with wide track bounds
#     final_track_difficulty = 1.0    # Full difficulty
    
#     curriculum_steps = 1000  # Increase difficulty every N episodes


# @configclass
# class DomainRandomizationCfg:
#     """Domain randomization settings."""
    
#     # Randomize friction
#     friction_range = (0.5, 1.5)
    
#     # Randomize mass
#     mass_range = (0.9, 1.1)  # ±10% mass variation
    
#     # Randomize actuator strength
#     actuator_strength_range = (0.9, 1.1)
    
#     # Visual randomization (if using camera)
#     light_intensity_range = (2500, 3500)
    