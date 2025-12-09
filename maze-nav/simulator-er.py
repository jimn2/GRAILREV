import pybullet as p
import pybullet_data
import time
import os
import zmq
import numpy as np

def wrap_angle(angle):
    """Normalize angle to [-pi, pi]."""
    return (angle + np.pi) % (2*np.pi) - np.pi


# =====================================================
#                 ZeroMQ Server
# =====================================================
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

# =====================================================
#               PyBullet Setup
# =====================================================
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

plane_id = p.loadURDF("plane.urdf")

# =====================================================
#               Load Husky Mobile Robot
# =====================================================
husky_start_pos = [0, 0, 0.1]
husky_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

husky_id = p.loadURDF(
    "husky/husky.urdf",
    husky_start_pos,
    husky_start_orientation,
    useFixedBase=False
)

wheel_joints = {
    "front_left": 2,
    "front_right": 3,
    "rear_left": 4,
    "rear_right": 5
}

# Tune wheel friction to reduce slipping
FORWARD_FRICTION = 0.0
SIDEWAYS_FRICTION = 2.0
SPINNING_FRICTION = 0.0

for j in wheel_joints.values():
    p.changeDynamics(
        husky_id,
        j,
        lateralFriction=SIDEWAYS_FRICTION,
        rollingFriction=FORWARD_FRICTION,
        spinningFriction=SPINNING_FRICTION
    )


# =====================================================
#               ADD MAZE BLOCKS HERE (1.5m CELLS)
# =====================================================

maze = [
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0]
]

CELL_SIZE = 1.5  # *** expanded from 1.0m to 1.5m grid spacing ***

# Block dimensions for 1.5m cells
block_half_extents = [CELL_SIZE/2, CELL_SIZE/2, 0.125]
block_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=block_half_extents)
block_visual = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=block_half_extents,
    rgbaColor=[0.6, 0.2, 0.2, 1]
)

# Offset so that cell1102 is world (0,0)
origin_row = 10   # row index of cell1102
origin_col = 1    # column index of cell1102

for r in range(12):
    for c in range(12):
        if maze[r][c] == 1:
            x = (c - origin_col) * CELL_SIZE
            y = -(r - origin_row) * CELL_SIZE
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=block_shape,
                baseVisualShapeIndex=block_visual,
                basePosition=[x, y, 0.125]
            )


# =====================================================
#         Waypoint Navigation Controller Params
# =====================================================
# Rotation controller
Kp_rot = 2.0
MAX_ROT_SPEED = 1.0
MIN_ROT_SPEED = 0.4
ANGLE_TOL = 0.01      # rad ≈ 3 degrees

# Drive controller
Kp_drive = 4.0
MAX_FWD_SPEED = 8.0
DIST_TOL = 0.05       # 5 cm


# =====================================================================
#         INTERNAL STATE: current waypoint and control phase
# =====================================================================
current_waypoint = None
control_phase = "idle"      # "rotate_to_target", "drive_to_target", "final_rotate"
print("Simulator ready.")


# =====================================================================
#                   Helper: Apply wheel speeds
# =====================================================================
def set_wheels(left, right):
    p.setJointMotorControl2(husky_id, wheel_joints["front_left"],
                            p.VELOCITY_CONTROL, targetVelocity=left)
    p.setJointMotorControl2(husky_id, wheel_joints["rear_left"],
                            p.VELOCITY_CONTROL, targetVelocity=left)
    p.setJointMotorControl2(husky_id, wheel_joints["front_right"],
                            p.VELOCITY_CONTROL, targetVelocity=right)
    p.setJointMotorControl2(husky_id, wheel_joints["rear_right"],
                            p.VELOCITY_CONTROL, targetVelocity=right)


# =====================================================================
#   Navigation Step: Computes wheel commands and whether waypoint is done
# =====================================================================
def navigation_step():
    global current_waypoint, control_phase

    if current_waypoint is None:
        set_wheels(0, 0)
        return False  # No waypoint active

    # Read robot state
    pos, orn = p.getBasePositionAndOrientation(husky_id)
    roll, pitch, yaw = p.getEulerFromQuaternion(orn)

    x, y = pos[0], pos[1]
    wx, wy = current_waypoint["x"], current_waypoint["y"]
    final_heading = current_waypoint["heading"]

    dx = wx - x
    dy = wy - y
    dist = np.sqrt(dx*dx + dy*dy)
    target_angle = np.arctan2(dy, dx)
    angle_error = wrap_angle(target_angle - yaw)
    final_heading_error = wrap_angle(final_heading - yaw)

    # ===========================================
    # Phase 1: Rotate to face waypoint
    # ===========================================
    if control_phase == "rotate_to_target":

        # FIX A: if we are basically *at* the waypoint position,
        # skip Phase 1 (rotate_to_target) and go directly to final_rotate.
        if dist < DIST_TOL:
            control_phase = "final_rotate"
            set_wheels(0, 0)
            return False

        # Normal rotate_to_target behavior
        if abs(angle_error) < ANGLE_TOL:
            control_phase = "drive_to_target"
            set_wheels(0, 0)
        else:
            rot = Kp_rot * angle_error
            rot = np.clip(rot, -MAX_ROT_SPEED, MAX_ROT_SPEED)
            if abs(rot) < MIN_ROT_SPEED:
                rot = MIN_ROT_SPEED * np.sign(rot)
            set_wheels(-rot, rot)
        return False

    # ===========================================
    # Phase 2: Drive straight toward waypoint
    # ===========================================
    if control_phase == "drive_to_target":
        if dist < DIST_TOL:
            control_phase = "final_rotate"
            set_wheels(0, 0)
        else:
            fwd = Kp_drive * dist
            fwd = min(fwd, MAX_FWD_SPEED)
            set_wheels(fwd, fwd)
        return False

    # ===========================================
    # Phase 3: Final rotation to desired heading
    # ===========================================
    if control_phase == "final_rotate":
        if abs(final_heading_error) < ANGLE_TOL:
            set_wheels(0, 0)
            current_waypoint = None
            control_phase = "idle"
            return True  # SUCCESS
        else:
            rot = Kp_rot * final_heading_error
            rot = np.clip(rot, -MAX_ROT_SPEED, MAX_ROT_SPEED)
            if abs(rot) < MIN_ROT_SPEED:
                rot = MIN_ROT_SPEED * np.sign(rot)
            set_wheels(-rot, rot)
        return False


# =====================================================
#                     MAIN LOOP
# =====================================================
while True:

    # Read message from agent
    msg = socket.recv_json()

    # Check for new waypoint
    if "waypoint" in msg:
        current_waypoint = msg["waypoint"]
        control_phase = "rotate_to_target"
        socket.send_json({"ack": True})
        continue

    # Step navigation controller
    done = navigation_step()

    # Debug prints
    if current_waypoint is not None:
        x = pos[0]
        y = pos[1]
        yaw = yaw
        wx = current_waypoint["x"]
        wy = current_waypoint["y"]
        wh = current_waypoint["heading"]

        dx = wx - x
        dy = wy - y
        dheading = wrap_angle(wh - yaw)

        print(f"[SIM] ACTUAL:  x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
        print(f"[SIM] DESIRED: x={wx:.2f}, y={wy:.2f}, yaw={wh:.2f}")
        print(f"[SIM] ERROR:   dx={dx:.2f}, dy={dy:.2f}, dyaw={dheading:.2f}")
        print()

    # Build robot state for agent
    pos, orn = p.getBasePositionAndOrientation(husky_id)
    roll, pitch, yaw = p.getEulerFromQuaternion(orn)
    lin_vel, ang_vel = p.getBaseVelocity(husky_id)

    state = {
        "x": pos[0],
        "y": pos[1],
        "yaw": yaw,
        "vx": lin_vel[0],
        "vy": lin_vel[1],
        "wz": ang_vel[2]
    }

    socket.send_json({
        "done": done,
        "robot_state": state
    })

    p.stepSimulation()
    time.sleep(1.0 / 60.0)
