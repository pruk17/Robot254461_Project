# UR5 Jacobian-based IK with your DH (threaded Python script for CoppeliaSim)
# - Uses your confirmed DH: DH_Base, DH, DH_EE, th_offset
# - Target is a 50x50x50 mm cuboid (created if missing)
# - Solves IK (position + orientation) with damped least squares
# - Prints a single summary at the very end (no intermediate prints)

import math
import time
import numpy as np

# ----------------------- Small math helpers -----------------------
pi = math.pi

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def Rx(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]], dtype=float)

def Ry(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]], dtype=float)

def Rz(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]], dtype=float)

def eulerXYZ_from_R(R):
    """CoppeliaSim-like intrinsic X-Y-Z Euler."""
    r02 = clamp(float(R[0,2]), -1.0, 1.0)
    beta = math.asin(r02)  # Y
    if abs(r02) < 1.0 - 1e-9:
        alpha = math.atan2(-R[1,2], R[2,2])  # X
        gamma = math.atan2(-R[0,1], R[0,0])  # Z
    else:
        alpha = 0.0
        gamma = math.atan2(R[1,0], R[1,1])
    return np.array([alpha, beta, gamma], dtype=float)

def T_of(alpha, a, d, theta):
    """Craig-style DH (alpha, a, d, theta)."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
        [   ct,   -st,    0,     a],
        [st*ca, ct*ca, -sa, -d*sa],
        [st*sa, ct*sa,  ca,  d*ca],
        [    0,     0,   0,     1]], dtype=float)

def pose_to_T(pos, eul_xyz):
    R = Rx(eul_xyz[0]) @ Ry(eul_xyz[1]) @ Rz(eul_xyz[2])
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = np.array(pos, float)
    return T

def T_to_pose(T):
    pos = T[:3,3].copy()
    eul = eulerXYZ_from_R(T[:3,:3])
    return pos, eul

def rot_log(R):
    """Returns axis-angle vector omega, where exp(omega^) = R."""
    tr = (np.trace(R) - 1.0) * 0.5
    tr = clamp(tr, -1.0, 1.0)
    th = math.acos(tr)
    if th < 1e-6:
        return np.zeros(3)
    w_hat = (R - R.T) / (2.0 * math.sin(th))
    return th * np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]], dtype=float)

# ----------------------- Your confirmed DH -----------------------
# From Base to Frame 0  (alpha, a, d, theta_fixed)
DH_Base = np.array([[0.0, 0.0, 0.0892, -pi/2]], dtype=float)

# 6 joints (alpha, a, d, theta_fixed)
DH = np.array([
    [0.0,    0.0,      0.0,     0.0],
    [pi/2,   0.0,      0.0,     0.0],
    [0.0,    0.4251,   0.0,     0.0],
    [0.0,    0.39215,  0.11,    0.0],
    [-pi/2,  0.0,      0.09475, 0.0],
    [pi/2,   0.0,      0.0,     0.0],
], dtype=float)

# From Frame 6 to End-Effector TCP
DH_EE = np.array([[0.0, 0.0, 0.26658, pi]], dtype=float)

# Offset angles added to measured joints (to match your scene's home)
th_offset = np.array([0.0, pi/2, 0.0, -pi/2, 0.0, 0.0], dtype=float)

# ----------------------- FK & Jacobian -----------------------
def fk_ur5(q):
    """Forward kinematics from joint angles q (rad) to world T0E."""
    q = np.asarray(q, float)
    T = np.eye(4)
    a, A, d, thf = DH_Base[0]
    T = T @ T_of(a, A, d, thf)
    for i in range(6):
        a, A, d, thf = DH[i]
        T = T @ T_of(a, A, d, thf + q[i] + th_offset[i])
    a, A, d, thf = DH_EE[0]
    T = T @ T_of(a, A, d, thf)
    return T

def numeric_jacobian(q, h=1e-6):
    """6x6 numerical Jacobian via forward difference."""
    q = np.asarray(q, float)
    T0 = fk_ur5(q); p0 = T0[:3,3]; R0 = T0[:3,:3]
    J = np.zeros((6,6), float)
    for j in range(6):
        dq = np.zeros(6); dq[j] = h
        Tj = fk_ur5(q + dq); pj = Tj[:3,3]; Rj = Tj[:3,:3]
        J[:3, j] = (pj - p0)/h
        dR = R0.T @ Rj
        J[3:, j] = rot_log(dR)/h
    return J

def ik_dls(q0, T_target, max_iters=200, tol_pos=2e-4, tol_ori=2e-3, lam=1e-2):
    """Damped least-squares IK on [pos_err; ori_err] (ori via log(Rd R^T))."""
    q = np.array(q0, float)
    for _ in range(max_iters):
        T = fk_ur5(q)
        ep = T_target[:3,3] - T[:3,3]
        eo = rot_log(T_target[:3,:3] @ T[:3,:3].T)
        if np.linalg.norm(ep) < tol_pos and np.linalg.norm(eo) < tol_ori:
            return q, True
        J = numeric_jacobian(q)
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (lam**2)*np.eye(6), np.hstack([ep, eo]))
        # limit step for stability
        n = np.linalg.norm(dq)
        if n > 0.2:
            dq = dq * (0.2 / n)
        q += dq
    return q, False

# ----------------------- CoppeliaSim glue -----------------------
def sysCall_init():
    global sim, jh, end_dummy, target, summary
    sim = require('sim')
    # handles
    jh = [
        sim.getObject('/UR5/joint'),
        sim.getObject('/UR5/joint/link/joint'),
        sim.getObject('/UR5/joint/link/joint/link/joint'),
        sim.getObject('/UR5/joint/link/joint/link/joint/link/joint'),
        sim.getObject('/UR5/joint/link/joint/link/joint/link/joint/link/joint'),
        sim.getObject('/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint'),
    ]
    end_dummy = sim.getObject('/UR5/EndPoint')

    # create/find 50mm cuboid
    target = None
    try:
        target = sim.getObject('/IKTargetCuboid')
    except:
        pass
    if target is None or target == -1:
        size = [0.05, 0.05, 0.05]
        target = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size, 0)
        sim.setObjectAlias(target, 'IKTargetCuboid', 0)
        sim.setObjectPosition(target, -1, [-0.60, 0.25, 0.90])
        sim.setObjectOrientation(target, -1, [20*pi/180, -50*pi/180, -120*pi/180])

    summary = []
    summary.append('==================== IK SUMMARY ====================')

def read_q():
    return np.array([sim.getJointPosition(h) for h in jh], float)

def set_q(q):
    for i,h in enumerate(jh):
        sim.setJointPosition(h, float(q[i]))

def get_target_T():
    pos = sim.getObjectPosition(target, -1)
    eul = sim.getObjectOrientation(target, -1)  # Euler XYZ (rad)
    return pose_to_T(pos, eul)

def get_scene_T_end():
    pos = sim.getObjectPosition(end_dummy, -1)
    eul = sim.getObjectOrientation(end_dummy, -1)
    return pose_to_T(pos, eul)

def sysCall_thread():
    try:
        # Single-shot IK to the current cuboid pose
        Td = get_target_T()
        q0 = read_q()
        q_sol, ok = ik_dls(q0, Td, max_iters=200, tol_pos=2e-4, tol_ori=2e-3, lam=1e-2)
        set_q(q_sol)
        # allow one step for scene to update
        sim.switchThread()
        # collect final values
        T_fk = fk_ur5(q_sol)
        T_sc = get_scene_T_end()
        p_fk, e_fk = T_to_pose(T_fk)
        p_sc, e_sc = T_to_pose(T_sc)
        summary.append(f"Converged: {ok}")
        summary.append(f"q_sol (deg) = {np.round(np.degrees(q_sol),3).tolist()}")
        summary.append(f"FK position = {np.round(p_fk,6).tolist()}")
        summary.append(f"Scene position = {np.round(p_sc,6).tolist()}")
        summary.append(f"FK eulXYZ° = {np.round(np.degrees(e_fk),3).tolist()}")
        summary.append(f"Scene eulXYZ° = {np.round(np.degrees(e_sc),3).tolist()}")
    except Exception as ex:
        summary.append(f"[ERROR] {type(ex).__name__}: {ex}")

def sysCall_cleanup():
    # print once at the end
    print('\n'.join(summary))
    print('[done]')
