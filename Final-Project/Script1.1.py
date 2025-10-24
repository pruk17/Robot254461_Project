#luaExec wrapper='pythonWrapper'
# Threaded Python controller: drive IkTarget in WORLD, follower Lua does IK.
# All comments in English.

import math

def sysCall_init():
    global sim, TGT
    sim = require('sim')
    # Absolute path to your target dummy:
    TGT = sim.getObject('/IRB4600/IkTarget')
    sim.addLog(sim.verbosity_scriptinfos, '[PyIKTarget] init (threaded)')

def _set_target_world(pos_m, eul_deg):
    """Set IkTarget in WORLD frame (legacy pythonWrapper signature)."""
    sim.setObjectPosition(TGT, [pos_m[0], pos_m[1], pos_m[2]], -1)
    sim.setObjectOrientation(
        TGT,
        [math.radians(eul_deg[0]), math.radians(eul_deg[1]), math.radians(eul_deg[2])],
        -1
    )

def _lerp_pose_world(p0, e0_deg, p1, e1_deg, duration_s):
    """Linear interpolation with one sim.step() per frame."""
    dt = max(sim.getSimulationTimeStep(), 0.01)
    steps = max(int(duration_s / dt), 40)
    for k in range(1, steps + 1):
        s = k / steps
        p = [p0[i] + s * (p1[i] - p0[i]) for i in range(3)]
        e = [e0_deg[i] + s * (e1_deg[i] - e0_deg[i]) for i in range(3)]
        _set_target_world(p, e)
        sim.step()

def _grip_open():
    """Signal-based open for PGripRightAngle."""
    sim.setInt32Signal('PGripRightAngle_cmd', 0)

def _grip_close():
    """Signal-based close for PGripRightAngle."""
    sim.setInt32Signal('PGripRightAngle_cmd', 1)

def sysCall_thread():
    # Sanity swirl so you can immediately see target motion
    base_pos = [0.000, 0.000, 0.40]
    base_eul = [0.0, 0.0, 90.0]
    for i in range(60):
        ang = (i / 60.0) * 2.0 * math.pi
        p = [base_pos[0] + 0.08 * math.cos(ang),
             base_pos[1] + 0.08 * math.sin(ang),
             base_pos[2]]
        _set_target_world(p, base_eul)
        sim.step()

    # Task waypoints (meters, degrees)
    start_pos = [0.000,  0.000, 0.0975]; start_eul = [0.0, 0.0, 90.0]
    pick_pos  = [2.000,  0.000, 1.600 ]; pick_eul  = [0.0, 0.0, 90.0]
    lift_pos  = [2.000,  0.000, 2.600 ]; lift_eul  = [0.0, 0.0, 90.0]  # +1.0 m
    place_pos = [0.000, -1.600, 1.200 ]; place_eul = [0.0, 0.0, 90.0]

    # Sequence: start ? pick(close) ? lift ? place(open) ? start
    _lerp_pose_world(start_pos, start_eul, start_pos, start_eul, 0.2); _grip_open()
    _lerp_pose_world(start_pos, start_eul, pick_pos,  pick_eul,  1.2); _grip_close()
    _lerp_pose_world(pick_pos,  pick_eul,  lift_pos,  lift_eul,  1.0)
    _lerp_pose_world(lift_pos,  lift_eul,  place_pos, place_eul, 1.4); _grip_open()
    _lerp_pose_world(place_pos, place_eul, start_pos, start_eul, 1.2)

    # Keep thread alive (optional)
    while True:
        sim.step()
