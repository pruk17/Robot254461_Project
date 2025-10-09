#F:\StudyStuff_CMU\scripts>py -3.12 ur5_external_fk_jacobian.py
"""
UR5 Forward Kinematics & Jacobian (CoppeliaSim ZeroMQ Remote API)
-----------------------------------------------------------------
Changes in this version:
1) Print labels in full words (no abbreviations). Example:
   - "position_predicted" instead of "position_predicted"
   - "position_simulated" instead of "position_simulated"
   - "euler_XYZ_degrees_predicted" instead of "euler_XYZ_degrees_predicted"
   - "euler_XYZ_degrees_simulated" instead of "euler_XYZ_degrees_simulated"
   - "linear_velocity_x" / "angular_velocity_x" (Jacobian rows)
2) Add a Denavit–Hartenberg (DH) table printer that lists rows in the order:
   alpha, a, d, theta   (theta uses the mapping S*q + OFFS for each joint).
3) Add clear English comments/docstrings to explain every step.
"""
# ur5_external_fk_jacobian.py
# External Python via ZeroMQ Remote API:
# - ใช้ DH table "ชุดล่าสุด" ตามที่ผู้ใช้ยืนยัน
# - คาลิเบรต T6->tip ด้วย "มุมที่ใช้งานจริง" (หลัง mapping) เพื่อลด error เชิงเฟรม
# - คำนวณ FK (pos + Euler X-Y-Z) และเทียบกับค่าจากซีน
# - ดึง Jacobian แบบ robust ด้วย simIK.addElementFromScene + computeGroupJacobian

#F:\StudyStuff_CMU\scripts>py -3.12 ur5_external_fk_jacobian.py
import math, time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


# (a_i, alpha_i, d_i) [m, rad]
# DH = [


def print_dh_table_alpha_a_d_theta():
    """Print the current DH parameters in the order: alpha, a, d, theta.
    - alpha [rad], a [m], d [m], theta expression uses the mapping S*q + OFFS.
    This helps verify that your DH and angle mapping match the scene.
    """
    print("Denavit–Hartenberg table (alpha, a, d, theta):")
    for i,(a, alpha, d) in enumerate(DH):
        s = S[i] if i < len(S) else 1
        offs = OFFS[i] if i < len(OFFS) else 0.0
        theta_expr = f"{s}*q{i+1} + {offs}"
        print(f"  joint {i+1}: alpha={alpha:.9f} rad, a={a:.6f} m, d={d:.6f} m, theta={theta_expr}")
    print("-"*60)


#     ( 0.0,           0.0,        0.0892 ),      # i=1
#     ( -0.425,        +math.pi/2, 0.0    ),      # i=2
#     ( -0.3922,       0.0,        0.0    ),      # i=3
#     ( 0.0,           0.0,        0.0    ),      # i=4
#     ( 0.0,           -math.pi/2, -0.03972),     # i=5
#     ( 0.0,           +math.pi/2, 0.0    )       # i=6
# ]

DH = [
    ( 0.0,           0.0,        0.0     ),     #i=1
    ( 0.0,           +math.pi/2, 0.0     ),     # i=2 
    ( 0.4251,        0.0,        0.0     ),     # i=3
    ( 0.39215,       0.0,        0.11    ),     # i=4    
    ( 0.0,           -math.pi/2, 0.09475),     # i=5
    ( 0.0,           +math.pi/2, 0.0    )       # i=6
]

# ========= Connect =========
client = RemoteAPIClient('localhost', 23000)
sim   = client.require('sim')
simIK = client.require('simIK')

client.setStepping(True)
if sim.getSimulationState() == sim.simulation_stopped:
    sim.startSimulation()
    # รอให้ซิมเริ่มจริง ๆ
    for _ in range(60):
        if sim.getSimulationState() != sim.simulation_stopped:
            break
        time.sleep(0.05)

# ========= Utilities =========
def I4():
    """Return 4x4 identity matrix."""
    return [[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]]

def mmul(A,B):
    """Matrix multiplication for 4x4 transforms."""
    return [[sum(A[i][k]*B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

def invRB(T):
    # rigid transform inverse
    R = [row[:3] for row in T[:3]]
    Rt = [[R[j][i] for j in range(3)] for i in range(3)]
    t  = [T[0][3], T[1][3], T[2][3]]
    mt = [-sum(Rt[i][k]*t[k] for k in range(3)) for i in range(3)]
    return [[Rt[0][0],Rt[0][1],Rt[0][2],mt[0]],
            [Rt[1][0],Rt[1][1],Rt[1][2],mt[1]],
            [Rt[2][0],Rt[2][1],Rt[2][2],mt[2]],
            [0,0,0,1]]

def T_DH(a, alpha, d, th):
    """Classic DH homogeneous transform from (a, alpha, d, theta)."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(th),    math.sin(th)
    return [
        [ ct,     -st*ca,   st*sa,  a*ct ],
        [ st,      ct*ca,  -ct*sa,  a*st ],
        [  0,         sa,      ca,     d ],
        [  0,          0,       0,     1 ]
    ]

def mat12_to_4x4(M):
    # CoppeliaSim poseToMatrix (12) -> 4x4
    return [[M[0], M[3], M[6],  M[9]],
            [M[1], M[4], M[7],  M[10]],
            [M[2], M[5], M[8],  M[11]],
            [0,0,0,1]]

def eulerXYZ_from_T(T):
    # intrinsic X-Y-Z
    r11,r12,r13 = T[0][0],T[0][1],T[0][2]
    r21,r22,r23 = T[1][0],T[1][1],T[1][2]
    r31,r32,r33 = T[2][0],T[2][1],T[2][2]
    # clamp เพื่อกัน numeric drift
    s = max(-1.0, min(1.0, -r31))
    by = math.asin(s)           # Y
    cx = math.atan2(r32, r33)   # X
    cz = math.atan2(r21, r11)   # Z
    return (cx,by,cz)

def rad2deg(t):
    return tuple(round(v*180.0/math.pi, 3) for v in t)

def fk_DH(q):
    """Compute UR5 forward kinematics using DH and joint angles q[0..5]."""
    T = I4()
    for i,(a,al,d) in enumerate(DH):
        T = mmul(T, T_DH(a, al, d, q[i]))
    return T

# ========= Auto-detect joints & tip & base =========
def find_endpoint_dummy():
    try:
        return sim.getObject('/UR5/EndPoint')
    except:
        pass
    for h in sim.getObjectsInTree(sim.handle_scene, sim.object_dummy_type, 0):
        name = sim.getObjectAlias(h, 1)
        if name and 'endpoint' in name.lower():
            return h
    raise RuntimeError("ไม่พบ EndPoint dummy ในซีน")

def joints_from_tip(tip):
    joints = []
    node = tip
    seen = set()
    while True:
        node = sim.getObjectParent(node)
        if node == -1:
            break
        if sim.getObjectType(node) == sim.object_joint_type and sim.getJointType(node) == sim.joint_revolute_subtype:
            if node not in seen:
                joints.insert(0, node); seen.add(node)
    return joints

def find_base_link(first_joint):
    # เดินขึ้นจนเจอ object ที่ไม่ใช่ joint (shape/dummy/model)
    b = sim.getObjectParent(first_joint)
    while b != -1 and sim.getObjectType(b) == sim.object_joint_type:
        b = sim.getObjectParent(b)
    return b if b != -1 else sim.handle_world

tip    = find_endpoint_dummy()
joints = joints_from_tip(tip)
base   = find_base_link(joints[0])

print("[info] joints:", [sim.getObjectAlias(j,1) for j in joints])
print("[info] tip:", sim.getObjectAlias(tip,1))
print("[info] base link:", sim.getObjectAlias(base,1) if base!=-1 else "<world>")

# ========= Mapping มุม (ค่าพื้นฐาน: ไม่ใช้ OFFS/S) =========
#  - หากอยากกลับทิศ joint1 ให้ S[0] = -1 ได้ (เช่นบางซีน)
S     = [ 1, 1, 1, 1, 1, 1 ]
OFFS  = [ 0.0 ]*6

def q_map_from_raw(q_raw):
    """Map simulator joint angles to the DH convention via S*q + OFFS."""
    # map = S*q + OFFS
    return [ S[i]*q_raw[i] + OFFS[i] for i in range(6) ]

# ========= Calibrate T6 -> tip (ใช้ "มุมที่ map แล้ว") =========
def calibrate_T6_to_tip_mapped(q_raw):
    qm = q_map_from_raw(q_raw)
    T06 = fk_DH(qm)
    T0_tip = mat12_to_4x4(sim.poseToMatrix(sim.getObjectPose(tip, sim.handle_world)))
    return mmul(invRB(T06), T0_tip)

q0 = [sim.getJointPosition(j) for j in joints]
T6_to_tip = calibrate_T6_to_tip_mapped(q0)
print("[info] calibrated T6->tip (with mapped angles)")

# ========= IK group (robust Jacobian) =========
ikEnv  = simIK.createEnvironment()
ikGrp  = simIK.createGroup(ikEnv)
target = sim.createDummy(0.01, None)
sim.setObjectPose(target, sim.handle_world, sim.getObjectPose(tip, sim.handle_world))

# สร้าง element อัตโนมัติจากซีน
simIK.addElementFromScene(ikEnv, ikGrp, base, tip, target, simIK.constraint_pose)

# ========= Helpers (แสดงผลครั้งเดียวตาม q กำหนด) =========
def print_fk_and_jacobian_once(q_raw):
    # (1) FK จาก DH
    qm  = q_map_from_raw(q_raw)
    T06 = fk_DH(qm)
    T0_tip_pred = mmul(T06, T6_to_tip)
    pos_fk  = (round(T0_tip_pred[0][3],4), round(T0_tip_pred[1][3],4), round(T0_tip_pred[2][3],4))
    eul_fk  = rad2deg(eulerXYZ_from_T(T0_tip_pred))

    # (2) ค่าจากซีน
    T0_tip = mat12_to_4x4(sim.poseToMatrix(sim.getObjectPose(tip, sim.handle_world)))
    position_simulated = (round(T0_tip[0][3],4), round(T0_tip[1][3],4), round(T0_tip[2][3],4))
    eul_sim = rad2deg(eulerXYZ_from_T(T0_tip))

    print(f"[Forward Kinematics] position_predicted {pos_fk} | position_simulated {position_simulated}")
    print(f"[Forward Kinematics] euler_XYZ_degrees_predicted {eul_fk} | euler_XYZ_degrees_simulated {eul_sim}")

    # (3) Jacobian (6 x dof)
    Jlist, err = simIK.computeGroupJacobian(ikEnv, ikGrp)
    dof = len(joints)
    J = [Jlist[i*dof:(i+1)*dof] for i in range(6)]
    print(f"[Jacobian] linear_velocity_x:{[round(v,4) for v in J[0]]}")
    print(f"[Jacobian] linear_velocity_y:{[round(v,4) for v in J[1]]}")
    print(f"[Jacobian] linear_velocity_z:{[round(v,4) for v in J[2]]}")
    print(f"[Jacobian] angular_velocity_x:{[round(v,4) for v in J[3]]}")
    print(f"[Jacobian] angular_velocity_y:{[round(v,4) for v in J[4]]}")
    print(f"[Jacobian] angular_velocity_z:{[round(v,4) for v in J[5]]}")

# ========= เดินซิมและแสดงผลหลายสเต็ป (ตัวอย่าง) =========
for step in range(7):
    client.step()
    q = [sim.getJointPosition(j) for j in joints]
    print("-"*60 if step>0 else "")
    print_fk_and_jacobian_once(q)

print("[done]")

# ========= (ฟังก์ชันสำหรับใช้ในวิดีโอ/ทดลองด้วยมุมกำหนดเอง) =========
def fk_and_jacobian_for_degrees(deg_list):
    """
    ใส่มุมองศา [θ1..θ6] เพื่อคำนวณ FK + Jacobian 1 ครั้ง
    ใช้ตอนอัดวิดีโอ: ใส่มุมเองแล้วเรียกฟังก์ชันนี้
    """
    rad = [math.radians(d) for d in deg_list]
    print_fk_and_jacobian_once(rad)
