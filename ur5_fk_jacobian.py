# UR5 FK using your DH_Base, DH, DH_EE and th_offset (CoppeliaSim Threaded Python)
# - Reads joint angles during the run and remembers the last ones
# - On cleanup, computes forward kinematics with the given DH sets
# - Prints ONLY the final sample: FK (predicted) vs scene (simulated)
# - English comments; no extra window; simple formatting

import math
import time
import numpy as np

# ----------------- User DH specification (exactly as requested) -----------------
pi = math.pi
d2r = pi/180.0
r2d = 180.0/pi

# From Base to Frame 0
DH_Base = np.array([[0.0, 0.0, 0.0892, -pi/2]], dtype=float)   # (alpha, a, d, theta_fixed)

# Main chain 1..6
DH = np.array([
    [0.0,       0.0,      0.0,     0.0],
    [pi/2,      0.0,      0.0,     0.0],
    [0.0,       0.4251,   0.0,     0.0],
    [0.0,       0.39215,  0.11,    0.0],
    [-pi/2,     0.0,      0.09475, 0.0],
    [pi/2,      0.0,      0.0,     0.0],
], dtype=float)

# From Frame 6 to End-Effector (tool flange -> TCP)
DH_EE = np.array([[0.0, 0.0, 0.26658, pi]], dtype=float)

# Joint-angle offsets for the current home position
th_offset = np.array([0.0, pi/2, 0.0, -pi/2, 0.0, 0.0], dtype=float)

# Optional: set this True to run the given "Test 1" pose instead of the last scene pose
RUN_TEST_1 = False
TEST1_DEG  = np.array([10, 20, 30, 40, 50, 60], dtype=float)

# ---------------------------- Helpers ------------------------------------------
def dh_transform_matrix(alpha, a, d, theta):
    """Classic DH transform (alpha, a, d, theta)."""
    ca = math.cos(alpha); sa = math.sin(alpha)
    ct = math.cos(theta); st = math.sin(theta)
    # Same layout as your reference implementation
    A = np.array([
        [   ct,    -st,    0.0,    a],
        [st*ca,  ct*ca,   -sa,  -d*sa],
        [st*sa,  ct*sa,    ca,   d*ca],
        [  0.0,    0.0,   0.0,   1.0]
    ], dtype=float)
    return A

def rotmat_to_euler_xyz(R):
    """CoppeliaSim-like intrinsic X-Y-Z Euler from a rotation matrix."""
    EPS = 1e-9
    r02 = float(R[0, 2])
    r02 = max(-1.0, min(1.0, r02))
    beta = math.asin(r02)  # Y
    if abs(r02) < 1.0 - EPS:
        alpha = math.atan2(-R[1, 2], R[2, 2])  # X
        gamma = math.atan2(-R[0, 1], R[0, 0])  # Z
    else:
        alpha = 0.0
        gamma = math.atan2(R[1, 0], R[1, 1])
    return np.array([alpha, beta, gamma], dtype=float)

def fk_from_dh(theta):
    """Compute T0E = T_base * ?(T_i) * T_ee with the requested DH sets and offsets."""
    # T_base
    alpha, a, d, th_fixed = DH_Base[0]
    T = dh_transform_matrix(alpha, a, d, th_fixed)

    # chain 1..6 (theta_i + offset_i)
    for i in range(6):
        alpha, a, d, th_fixed = DH[i]
        th = float(theta[i] + th_offset[i] + th_fixed)
        T = T @ dh_transform_matrix(alpha, a, d, th)

    # T_ee
    alpha, a, d, th_fixed = DH_EE[0]
    T = T @ dh_transform_matrix(alpha, a, d, th_fixed)
    return T

def fmt_T(T):
    """Format a 4x4 nicely."""
    rows = []
    for r in range(4):
        rows.append("  {:+.6f}  {:+.6f}  {:+.6f}  {:+.6f}".format(T[r,0], T[r,1], T[r,2], T[r,3]))
    return "\n".join(rows)

# ---------------------------- CoppeliaSim hooks --------------------------------
def sysCall_init():
    # Python wrapper v2 provides "sim" via require
    global sim, h_j, h_tip, last_theta
    sim = require("sim")

    # Joint handles (same order as the scene)
    h_j = [
        sim.getObject("/UR5/joint"),
        sim.getObject("/UR5/joint/link/joint"),
        sim.getObject("/UR5/joint/link/joint/link/joint"),
        sim.getObject("/UR5/joint/link/joint/link/joint/link/joint"),
        sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint"),
        sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint"),
    ]
    h_tip = sim.getObject("/UR5/EndPoint")
    last_theta = np.zeros(6, dtype=float)

def sysCall_thread():
    # Just sample joint angles while simulation runs
    global last_theta
    while sim.getSimulationState() != sim.simulation_stopped:
        # read current joints (in radians)
        cur = [sim.getJointPosition(h) for h in h_j]
        last_theta = np.array(cur, dtype=float)
        # small yield
        time.sleep(0.02)
        sim.switchThread()

def sysCall_cleanup():
    # Decide which joint angles to use: last scene pose or TEST 1
    if RUN_TEST_1:
        theta = TEST1_DEG * d2r
    else:
        theta = last_theta

    # Predicted FK from DH sets
    T0E = fk_from_dh(theta)
    p_fk = T0E[:3, 3]
    eul_fk = rotmat_to_euler_xyz(T0E[:3, :3]) * r2d

    # Scene pose (simulated) for the same final instant
    p_sim = np.array(sim.getObjectPosition(h_tip, -1), dtype=float)
    eul_sim = np.array(sim.getObjectOrientation(h_tip, -1), dtype=float) * r2d

    # Print only once at the very end
    print("==================== FINAL ====================")
    print("Joint angles (degrees): [{}]".format(
        ", ".join("{:.3f}".format(x*r2d) for x in theta)
    ))

    print("\nForward kinematics (predicted, world frame)")
    print("  End point position: [{:+.6f}, {:+.6f}, {:+.6f}]".format(p_fk[0], p_fk[1], p_fk[2]))
    print("  End point orientation XYZ (degrees): [{:.3f}, {:.3f}, {:.3f}]".format(eul_fk[0], eul_fk[1], eul_fk[2]))
    print("  Homogeneous transform T0E:\n{}".format(fmt_T(T0E)))

    print("\nScene values (simulated, world frame)")
    print("  End point position: [{:+.6f}, {:+.6f}, {:+.6f}]".format(p_sim[0], p_sim[1], p_sim[2]))
    print("  End point orientation XYZ (degrees): [{:.3f}, {:.3f}, {:.3f}]".format(eul_sim[0], eul_sim[1], eul_sim[2]))
    print("[done]")
