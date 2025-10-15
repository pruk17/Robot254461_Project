# UR5 FK + Jacobian (single-mode, matches Test-1 reference)
# - Always uses th = [10,20,30,40,50,60] deg
# - FK from DH (with th_offset) exactly as provided
# - Jacobian: geometric (SPATIAL @ BASE; angular = ?), using z_i and o_i of frame i
# - Prints one final snapshot (no switches, no Euler-rate conversion, no flips)

import math, time
import numpy as np

pi = math.pi
d2r = pi/180.0
r2d = 180.0/pi

# ----------------- Correct DH specification -----------------
# (alpha, a, d, theta_fixed)
DH_Base = np.array([[0.0, 0.0, 0.0892, -pi/2]], dtype=float)

DH = np.array([
    [0.0,       0.0,      0.0,     0.0],
    [pi/2,      0.0,      0.0,     0.0],
    [0.0,       0.4251,   0.0,     0.0],
    [0.0,       0.39215,  0.11,    0.0],
    [-pi/2,     0.0,      0.09475, 0.0],
    [pi/2,      0.0,      0.0,     0.0],
], dtype=float)

DH_EE = np.array([[0.0, 0.0, 0.26658, pi]], dtype=float)

# joint offsets
th_offset = np.array([0.0, pi/2, 0.0, -pi/2, 0.0, 0.0], dtype=float)

# ---------- Helpers ----------
def dh_transform_matrix(alpha, a, d, theta):
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
        [   ct,    -st,    0.0,  a],
        [st*ca,  ct*ca,   -sa, -d*sa],
        [st*sa,  ct*sa,    ca,  d*ca],
        [  0.0,    0.0,   0.0, 1.0]
    ], dtype=float)

def rotmat_to_euler_xyz(R):
    EPS = 1e-9
    r02 = max(-1.0, min(1.0, float(R[0,2])))
    beta = math.asin(r02)
    if abs(r02) < 1.0 - EPS:
        alpha = math.atan2(-R[1,2], R[2,2])
        gamma = math.atan2(-R[0,1], R[0,0])
    else:
        alpha = 0.0
        gamma = math.atan2(R[1,0], R[1,1])
    return np.array([alpha, beta, gamma], dtype=float)

def fk_all_frames(theta):
    """Return list Ts with T0, T1, ..., T6 (after each joint), and T0E (TCP)"""
    Ts = []
    T = dh_transform_matrix(*DH_Base[0])  # T0
    Ts.append(T.copy())                   # Ts[0] = T0
    for i in range(6):
        alpha, a, d, thf = DH[i]
        th = float(theta[i] + th_offset[i] + thf)
        T = T @ dh_transform_matrix(alpha, a, d, th)
        Ts.append(T.copy())               # Ts[i+1] = T_i  (frame i)
    T = T @ dh_transform_matrix(*DH_EE[0])  # TCP
    return Ts, T

def jacobian_spatial_base(theta):
    """Geometric J in BASE frame (rows: v; ?), using z_i and o_i of frame i."""
    Ts, T0E = fk_all_frames(theta)
    oE = T0E[:3,3]
    J = np.zeros((6,6), dtype=float)
    # For joint i (0..5), use frame i: Ts[i+1]
    for i in range(6):
        Ti = Ts[i+1]
        zi = Ti[:3,2]          # z_i in BASE
        oi = Ti[:3,3]          # o_i in BASE
        J[:3,i] = np.cross(zi, (oE - oi))  # linear part
        J[3:,i] = zi                          # angular part = ? = z_i
    return J, T0E

def fmt_T(T):
    return "\n".join("  {:+.6f}  {:+.6f}  {:+.6f}  {:+.6f}".format(T[r,0],T[r,1],T[r,2],T[r,3]) for r in range(4))

# ---------- CoppeliaSim hooks ----------
def sysCall_init():
    global sim, h_j, h_tip
    sim = require("sim")
    h_j = [
        sim.getObject("/UR5/joint"),
        sim.getObject("/UR5/joint/link/joint"),
        sim.getObject("/UR5/joint/link/joint/link/joint"),
        sim.getObject("/UR5/joint/link/joint/link/joint/link/joint"),
        sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint"),
        sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint"),
    ]
    h_tip = sim.getObject("/UR5/EndPoint")

def sysCall_thread():
    # no sampling: single-mode with fixed Test-1 angles
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.02)
        sim.switchThread()

def sysCall_cleanup():
    # ----- Single target pose: Test-1 -----
    theta = np.array([10,20,30,40,50,60], dtype=float) * d2r

    # FK (predicted)
    Ts, T0E = fk_all_frames(theta)
    p_fk = T0E[:3,3]
    eul_fk = rotmat_to_euler_xyz(T0E[:3,:3]) * r2d

    # Scene pose (for info)
    p_sim  = np.array(sim.getObjectPosition(h_tip, -1), dtype=float)
    eul_sim = np.array(sim.getObjectOrientation(h_tip, -1), dtype=float) * r2d

    # Jacobian: geometric (SPATIAL @ BASE; angular = ?)
    J_spatial, _ = jacobian_spatial_base(theta)

    # ----- Print -----
    print("==================== FINAL (Test-1 only) ====================")
    print("Joint angles (degrees): [{}]".format(", ".join("{:.3f}".format(x*r2d) for x in theta)))

    print("\nForward kinematics (predicted, BASE frame)")
    print("  End point position: [{:+.5f}, {:+.5f}, {:+.5f}]".format(p_fk[0], p_fk[1], p_fk[2]))
    print("  End point orientation XYZ (degrees): [{:.2f}, {:.2f}, {:.2f}]".format(eul_fk[0], eul_fk[1], eul_fk[2]))
    print("  Homogeneous transform T0E:\n{}".format(fmt_T(T0E)))

    print("\nScene values (simulated, world frame)")
    print("  End point position: [{:+.5f}, {:+.5f}, {:+.5f}]".format(p_sim[0], p_sim[1], p_sim[2]))
    print("  End point orientation XYZ (degrees): [{:.2f}, {:.2f}, {:.2f}]".format(eul_sim[0], eul_sim[1], eul_sim[2]))

    print("\nJacobian (GEOMETRIC, SPATIAL @ BASE; angular = \u03C9)  [rows: linear x,y,z; angular x,y,z]")
    labels = ["linear x:", "linear y:", "linear z:", "angular x:", "angular y:", "angular z:"]
    for r in range(6):
        print("  {:11s} {}".format(labels[r], "  ".join("{:+.4f}".format(v) for v in J_spatial[r,:])))
    print("[done]")
