SimulationContext
│
├── Physics Backend (GPU PhysX)
│     ├── Time Stepping
│     ├── Solvers
│     ├── Contact/Collision Pipeline
│     └── GPU Tensors (PhysX States)
│
├── Renderer (RTX / USD Viewport)
│
└── Scene
      │
      ├── World Root Prim  (e.g., /World)
      │
      ├── Vectorized Environment Manager
      │       │
      │       ├── Environment Instances (Env #0 ... Env #N)
      │       │      │
      │       │      ├── Environment Prim (e.g., /World/envs/env_001)
      │       │      │
      │       │      ├── Assets
      │       │      │      ├── Rigid Bodies (prims)
      │       │      │      ├── Articulations (e.g., robots)
      │       │      │      │        ├── USD Articulation Root Prim
      │       │      │      │        ├── Links (Rigid Prims)
      │       │      │      │        └── Joints (Joint Prims)
      │       │      │      │
      │       │      │      ├── Static Geometry
      │       │      │      │      ├── Ground / Terrain prim
      │       │      │      │      └── Walls / Obstacles
      │       │      │      │
      │       │      │      └── Objects / Props
      │       │      │
      │       │      ├── Sensors
      │       │      │      ├── Camera Sensors
      │       │      │      │      ├── Color Buffer
      │       │      │      │      ├── Depth Buffer
      │       │      │      │      ├── Segmentation
      │       │      │      │      └── Point Cloud
      │       │      │      ├── LiDAR Sensors
      │       │      │      ├── IMU Sensors
      │       │      │      └── Contact Sensors
      │       │      │
      │       │      ├── Controllers
      │       │      │      ├── PD Controllers
      │       │      │      ├── OSC / Whole Body
      │       │      │      └── RL Policy Interface
      │       │      │
      │       │      ├── Domain Randomization
      │       │      │      ├── Physics Randomization
      │       │      │      ├── Material Randomization
      │       │      │      └── Visual Randomization
      │       │      │
      │       │      ├── Rewards
      │       │      │
      │       │      ├── Observations
      │       │      │      ├── State Observations
      │       │      │      ├── Sensor Observations
      │       │      │      └── Privileged Observations
      │       │      │
      │       │      └── Reset / Termination Logic
      │       │
      │       └── Env Space Layout
      │              ├── Grid Placement (e.g., NxM tiling)
      │              ├── Offsets per env
      │              └── Prim Naming per env
      │
      └── Scene Graph (USD)
             │
             ├── All Prims
             │     ├── Xform Prim
             │     ├── Mesh Prim
             │     ├── Rigid Body Prim
             │     ├── Joint Prim
             │     ├── Collision Prim
             │     ├── Material Prim
             │     ├── Camera Prim
             │     ├── Light Prim
             │     ├── Sensor Extension Prims
             │     └── Custom Prims
             │
             └── USD Stage