    front_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",  # Attached to robot's base link
        update_period=0.1,  # Update at 10 Hz (every 0.1 seconds)
        height=480,  # Image height in pixels
        width=640,   # Image width in pixels
        data_types=["rgb", "distance_to_image_plane"],  # RGB and depth data
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,  # Focal length in mm
            focus_distance=400.0,  # Focus distance in stage units
            horizontal_aperture=20.955,  # Horizontal aperture in mm
            clipping_range=(0.1, 1.0e5)  # Near and far clipping planes
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.3, 0.0, 0.1),  # 30cm forward, 10cm up from base
            rot=(0.5, -0.5, 0.5, -0.5),  # Quaternion for forward-facing (ROS convention)
            convention="ros"  # Use ROS coordinate convention
        ),
    )
    
    # Optional: Top-down camera for debugging/visualization
    # top_camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base/top_camera",
    #     update_period=0.2,  # Update at 5 Hz
    #     height=480,
    #     width=640,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=6.0,  # Wide angle for broader view
    #         focus_distance=500.0,
    #         horizontal_aperture=8.0,
    #         clipping_range=(0.1, 1000.0)
    #     ),
    #     offset=CameraCfg.OffsetCfg(
    #         pos=(0.0, 0.0, 2.0),  # 2 meters above robot
    #         rot=(0.0, 0.7071068, 0.0, -0.7071068),  # Looking down
    #         convention="ros"
    #     ),
    # )


# Alternative camera configurations you might want to use:

# Configuration 1: Racing game style (behind and above)
RACING_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/chase_camera",
    update_period=0.05,  # 20 Hz for smoother following
    height=720,
    width=1280,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=35.0,
        focus_distance=500.0,
        horizontal_aperture=36.0,
        clipping_range=(0.1, 1000.0)
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(-0.5, 0.0, 0.3),  # Behind and above
        rot=(0.5, -0.5, 0.5, -0.5),
        convention="ros"
    ),
)

# Configuration 2: First-person view (driver's perspective)
DRIVER_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/driver_camera",
    update_period=0.033,  # 30 Hz for immersive view
    height=720,
    width=1280,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=28.0,
        focus_distance=300.0,
        horizontal_aperture=24.0,
        clipping_range=(0.05, 500.0)
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.1, 0.0, 0.2),  # Slightly forward and up (driver seat position)
        rot=(0.5, -0.5, 0.5, -0.5),
        convention="ros"
    ),
)

# Configuration 3: Wide-angle front camera (for better track visibility)
WIDE_ANGLE_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/wide_camera",
    update_period=0.1,
    height=480,
    width=848,  # Wider aspect ratio
    data_types=["rgb", "semantic_segmentation"],  # Include semantic segmentation
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=14.0,  # Shorter focal length for wider field of view
        focus_distance=200.0,
        horizontal_aperture=16.0,
        clipping_range=(0.05, 100.0)
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.25, 0.0, 0.15),
        rot=(0.5, -0.5, 0.5, -0.5),
        convention="ros"
    ),
)