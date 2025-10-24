# UR5_IK_Target_Cuboid_numeric_MIN.py
# Minimal numeric IK controller (no simIK). English-only comments.

import math, numpy as np
pi, d2r, r2d = math.pi, math.pi/180.0, 180.0/math.pi

# ---- Tunables ----
HSTEP = 1e-4         # finite-diff step (rad)
LAMBDA = 1e-3        # damping
GAIN = 0.6           # step scale
DQ_CLAMP = 0.2       # per-iter joint clamp (rad)
POS_TOL = 1e-4       # m
ANG_TOL_DEG = 0.5    # deg
AUTO_SAMPLES = [
    ([0.35, 0.10, 0.45], [0.0,  90.0,  0.0]),
    ([0.45, 0.20, 0.30], [90.0,  0.0,  0.0]),
    ([0.55, 0.00, 0.40], [ 0.0,  90.0, 90.0]),
]

# ---- Small helpers (WORLD pose only) ----
def w_set(sim, h, p, e): sim.setObjectPosition(h, -1, p); sim.setObjectOrientation(h, -1, e)
def w_get_p(sim, h):     return np.array(sim.getObjectPosition(h, -1), float)
def w_get_e(sim, h):     return np.array(sim.getObjectOrientation(h, -1), float)
def w_get_R(sim, h):
    M = sim.getObjectMatrix(h, -1); return np.array(M, float).reshape(3,4)[:, :3]

def eulXYZ_to_R(ex,ey,ez):
    cx,sx,cy,sy,cz,sz = math.cos(ex),math.sin(ex),math.cos(ey),math.sin(ey),math.cos(ez),math.sin(ez)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz@Ry@Rx

def rotvec_from_R(R):
    tr = np.trace(R); c = max(-1.0, min(1.0, (tr-1)*0.5)); th = math.acos(c)
    if th < 1e-12: return np.zeros(3)
    v = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])/(2*math.sin(th))
    return v*th

def pose_err(sim, tip, p_goal, e_goal):
    p = w_get_p(sim, tip); R = w_get_R(sim, tip); Rg = eulXYZ_to_R(*e_goal)
    dp = p_goal - p; drot = rotvec_from_R(R.T@Rg); return np.hstack([dp, drot])

def J_numeric(sim, joints, tip):
    q0 = [sim.getJointPosition(j) for j in joints]
    p0 = w_get_p(sim, tip); R0 = w_get_R(sim, tip)
    J = np.zeros((6, len(joints)))
    for i,jh in enumerate(joints):
        sim.setJointPosition(jh, q0[i] + HSTEP)
        p1 = w_get_p(sim, tip); R1 = w_get_R(sim, tip)
        sim.setJointPosition(jh, q0[i])
        J[:3,i] = (p1 - p0)/HSTEP
        J[3:,i] = rotvec_from_R(R0.T@R1)/HSTEP
    return J

# ---- CoppeliaSim hooks ----
def sysCall_init():
    global sim, h_base, h_tip, JNTS, h_cube, h_tgt
    sim = require("sim")

    h_base = sim.getObject("/UR5")
    h_tip  = sim.getObject("/UR5/EndPoint")
    JNTS = [
        sim.getObject("/UR5/joint"),
        sim.getObject("/UR5/joint/link/joint"),
        sim.getObject("/UR5/joint/link/joint/link/joint"),
        sim.getObject("/UR5/joint/link/joint/link/joint/link/joint"),
        sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint"),
        sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint"),
    ]

    # Create 50mm cuboid (primitive ? pure fallback)
    s = 0.05
    try:    h_cube = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [s,s,s], 0)
    except: 
        try: h_cube = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [s,s,s], 0, None)
        except: h_cube = sim.createPureShape(0, 0, [s,s,s], 0.1, None)

    # Place cuboid above plane; create child target (identity local)
    w_set(sim, h_cube, [0.45,0.00,0.30], [0,0,0])
    h_tgt = sim.createDummy(0.02)
    sim.setObjectParent(h_tgt, h_cube, True)
    sim.setObjectPosition(h_tgt, h_cube, [0,0,0])
    sim.setObjectOrientation(h_tgt, h_cube, [0,0,0])

    print("[IK Controller] Minimal numeric IK ready.")

def _solve_once(tag, max_iters=140):
    p_goal = w_get_p(sim, h_tgt); e_goal = w_get_e(sim, h_tgt)
    for k in range(max_iters):
        err = pose_err(sim, h_tip, p_goal, e_goal)
        if np.linalg.norm(err[:3]) < POS_TOL and np.linalg.norm(err[3:])*r2d < ANG_TOL_DEG: break
        J = J_numeric(sim, JNTS, h_tip); JT = J.T
        dq = JT @ np.linalg.solve(J@JT + (LAMBDA**2)*np.eye(6), err)
        dq = np.clip(GAIN*dq, -DQ_CLAMP, DQ_CLAMP)
        for i,jh in enumerate(JNTS):
            sim.setJointPosition(jh, sim.getJointPosition(jh) + float(dq[i]))
        sim.wait(0.005)

    q = [sim.getJointPosition(j) for j in JNTS]
    p = w_get_p(sim, h_tip); e = w_get_e(sim, h_tip)
    print(f"\n=== {tag} === iters={k+1}")
    print("  TCP pos (m): [{:+.5f}, {:+.5f}, {:+.5f}]".format(*p))
    print("  TCP eul (deg): [{:+.2f}, {:+.2f}, {:+.2f}]".format(*(e*r2d)))
    print("  Joints (deg): [{}]".format(", ".join(f"{x*r2d:+.2f}" for x in q)))

def sysCall_thread():
    # Auto: MOVE THE CUBOID (target follows as identity local)
    for i,(p,e_deg) in enumerate(AUTO_SAMPLES):
        w_set(sim, h_cube, p, [x*d2r for x in e_deg])
        _solve_once(f"Auto #{i+1}")

    print("\n[IK Controller] Manual: move the cuboid; solver will track 3 times.")
    for k in range(1, 4):
        sim.wait(0.75)
        _solve_once(f"Manual #{k}")

def sysCall_cleanup():
    print("[IK Controller] Cleanup.")
