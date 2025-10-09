# UR5 - Final-only print of FK vs Scene (Threaded Python Script)
# --------------------------------------------------------------
# - Captures the last joint angles and EndPoint poses during the run.
# - Prints ONCE at the end (sysCall_cleanup) in the same console window.
# - Shows only the final EndPoint pose (predicted FK and simulated scene).
# - Comments are in English.

import math
import time
import numpy as np

pi = np.pi
d2r = pi/180.0
r2d = 1.0/d2r

# ---- EDIT HERE: UR5 DH table (alpha, a, d, theta_offset) ----
JOINTS = [
    # alpha,      a,        d,        theta_offset (added to measured joint)
    (0.0,         0.0,      0.0892,   -np.pi/2),   # joint 1
    (np.pi/2,     0.0,      0.0,      +np.pi/2),   # joint 2
    (0.0,         0.4251,   0.0,      0.0),        # joint 3
    (0.0,         0.39215,  0.11,     +np.pi/2),   # joint 4
    (np.pi/2,     0.0,      0.09475,  0.0),        # joint 5
    (-np.pi/2,    0.0,      0.26658,  0.0),        # joint 6 (tool flange->TCP length)
]

def dh_transform_matrix(alpha, a, d, theta):
    """Classic DH homogeneous transform that matches your working file."""
    ca = math.cos(alpha); sa = math.sin(alpha)
    ct = math.cos(theta); st = math.sin(theta)
    return np.array([
        [   ct,   -st,   0.0,   a     ],
        [st*ca, ct*ca,  -sa,  -d*sa  ],
        [st*sa, ct*sa,   ca,   d*ca  ],
        [ 0.0 ,  0.0 ,  0.0,   1.0   ]
    ], dtype=float)

def rotmat_to_euler_XYZ(R):
    """Intrinsic X-Y-Z Euler like CoppeliaSim."""
    EPS = 1e-9
    r02 = float(R[0,2])
    r02 = max(-1.0, min(1.0, r02))
    beta = math.asin(r02)  # Y
    if abs(r02) < 1.0 - EPS:
        alpha = math.atan2(-R[1,2], R[2,2])   # X
        gamma = math.atan2(-R[0,1], R[0,0])   # Z
    else:
        alpha = 0.0
        gamma = math.atan2(R[1,0], R[1,1])
    return np.array([alpha, beta, gamma], dtype=float)

def ur5_fk(theta):
    """Forward kinematics from the editable DH table above."""
    T = np.eye(4, dtype=float)
    for i, (alpha, a, d, th_off) in enumerate(JOINTS):
        T = T @ dh_transform_matrix(alpha, a, d, theta[i] + th_off)
    pos = T[:3, 3]
    eul = rotmat_to_euler_XYZ(T[:3, :3])
    return pos, eul, T

# --------- global snapshot (final only) ---------
_last_q = None
_last_pos_sc = None
_last_eul_sc = None
_last_pos_fk = None
_last_eul_fk = None
_last_T0E = None

def sysCall_init():
    # Use CoppeliaSim's embedded Python "require"
    global sim
    sim = require('sim')

def _snapshot(h_j, h_end):
    """Read actual joint angles and scene EndPoint pose, compute FK from those angles."""
    q = [sim.getJointPosition(h_j[i]) for i in range(6)]
    pos_fk, eul_fk, T0E = ur5_fk(q)
    pos_sc = sim.getObjectPosition(h_end, -1)        # world
    eul_sc = sim.getObjectOrientation(h_end, -1)     # world, XYZ
    return q, pos_fk, eul_fk, T0E, pos_sc, eul_sc

def sysCall_thread():
    # Handles
    h_j = {}
    h_j[0] = sim.getObject("/UR5/joint")
    h_j[1] = sim.getObject("/UR5/joint/link/joint")
    h_j[2] = sim.getObject("/UR5/joint/link/joint/link/joint")
    h_j[3] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint")
    h_j[4] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint")
    h_j[5] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint")
    h_end = sim.getObject("/UR5/EndPoint")

    # Loop until the simulation stops; do NOT print per step.
    while sim.getSimulationState() != sim.simulation_stopped:
        q, pos_fk, eul_fk, T0E, pos_sc, eul_sc = _snapshot(h_j, h_end)

        global _last_q, _last_pos_sc, _last_eul_sc, _last_pos_fk, _last_eul_fk, _last_T0E
        _last_q = q
        _last_pos_sc = pos_sc
        _last_eul_sc = eul_sc
        _last_pos_fk = pos_fk
        _last_eul_fk = eul_fk
        _last_T0E = T0E

        sim.switchThread()

def _fmt_vec3(v):
    return f"[{float(v[0]):+0.6f}, {float(v[1]):+0.6f}, {float(v[2]):+0.6f}]"

def sysCall_cleanup():
    if _last_q is None:
        print("No final data captured.")
        return

    q_deg = [round(qi*r2d, 3) for qi in _last_q]
    eul_fk_deg = [round(float(e)*r2d, 3) for e in _last_eul_fk]
    eul_sc_deg = [round(float(e)*r2d, 3) for e in _last_eul_sc]

    print("==================== FINAL ====================")
    print(f"Joint angles (degrees): {q_deg}")

    print("\nForward kinematics (predicted, world frame)")
    print(f"  End point position: {_fmt_vec3(_last_pos_fk)}")
    print(f"  End point orientation XYZ (degrees): {eul_fk_deg}")

    print("\nScene values (simulated, world frame)")
    print(f"  End point position: {_fmt_vec3(_last_pos_sc)}")
    print(f"  End point orientation XYZ (degrees): {eul_sc_deg}")
