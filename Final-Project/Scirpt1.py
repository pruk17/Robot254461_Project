#luaExec wrapper='pythonWrapper'
# IRB4600 pick ? lift ? place using ONLY this Python file (legacy pythonWrapper).
# Uses Lua snippets via sim.runScriptString() to access simIK and move joints by HANDLE (no relative paths).

import math

def deg2rad(d): return d * math.pi / 180.0

def sysCall_init():
    # Acquire API via legacy wrapper
    global sim
    sim = require('sim')

    # Scene paths (match your hierarchy)
    global PATH_ROBOT, PATH_TIP, PATH_TARGET, PATH_GRIPPER
    PATH_ROBOT   = '/IRB4600'
    PATH_TIP     = '/IRB4600/IkTip'
    PATH_TARGET  = '/IRB4600/IkTarget'
    PATH_GRIPPER = '/IRB4600/connection/PGripRightAngle'

    # Resolve key handles
    global H
    H = {}
    H['robot']  = sim.getObject(PATH_ROBOT)
    H['tip']    = sim.getObject(PATH_TIP)
    H['target'] = sim.getObject(PATH_TARGET)

    # First 6 joints under robot (revolute/prismatic)
    allj = sim.getObjectsInTree(H['robot'], sim.object_joint_type, 1)
    joints = []
    for j in allj:
        jt = sim.getJointType(j)
        if jt == sim.joint_revolute_subtype or jt == sim.joint_prismatic_subtype:
            joints.append(j)
        if len(joints) == 6:
            break
    if len(joints) < 6:
        raise RuntimeError('Could not find six joints under /IRB4600')
    H['j0'], H['j1'], H['j2'], H['j3'], H['j4'], H['j5'] = joints[:6]
    H['joints'] = joints[:6]

    # Gripper joints (single or dual)
    try:
        pg = sim.getObject(PATH_GRIPPER)
        gjs = sim.getObjectsInTree(pg, sim.object_joint_type, 1)
    except:
        gjs = []
    if len(gjs) >= 2:
        H['gripL'], H['gripR'] = gjs[0], gjs[1]
        H['grip_is_single'] = False
    elif len(gjs) == 1:
        H['grip'] = gjs[0]
        H['grip_is_single'] = True
    else:
        H['grip'] = None
        H['grip_is_single'] = True

    # Bootstrap IK environment on Lua side once
    _lua_bootstrap_ik_env()

def _lua_bootstrap_ik_env():
    # Create persistent simIK env/group in Lua
    lua = r'''
        sim=require'sim'
        simIK=require'simIK'
        local base, tip, tgt = ...
        if not _pyIK then _pyIK={} end
        if not _pyIK.env then
            _pyIK.env=simIK.createEnvironment()
            _pyIK.grp=simIK.createGroup(_pyIK.env)
            _pyIK.base=base
            _pyIK.tip=tip
            _pyIK.tgt=tgt
            simIK.addElementFromScene(_pyIK.env,_pyIK.grp,_pyIK.base,_pyIK.tip,_pyIK.tgt,simIK.constraint_pose)
            simIK.setGroupCalculation(_pyIK.env,_pyIK.grp,simIK.method_pseudo_inverse,0.01,50)
        end
    '''
    sim.runScriptString(lua, H['robot'], H['tip'], H['target'])

def _lua_ik_set_pose_world(pos, eul_deg):
    # Apply one IK step at world pose (meters, degrees)
    lua = r'''
        sim=require'sim'
        simIK=require'simIK'
        local px,py,pz, ex,ey,ez = ...
        if not _pyIK or not _pyIK.env then return end
        local tgt=_pyIK.tgt
        sim.setObjectParent(tgt, -1, true)
        sim.setObjectPosition(tgt, {px,py,pz}, -1)
        sim.setObjectOrientation(tgt, {math.rad(ex),math.rad(ey),math.rad(ez)}, -1)
        simIK.handleGroup(_pyIK.env,_pyIK.grp,{syncWorlds=true})
        sim.setObjectParent(tgt, _pyIK.tip, true)
    '''
    sim.runScriptString(lua,
                        float(pos[0]), float(pos[1]), float(pos[2]),
                        float(eul_deg[0]), float(eul_deg[1]), float(eul_deg[2]))

def _lua_moveJ_to(q_rad, duration):
    # Joint-space interpolation using joint HANDLES (no relative paths)
    lua = r'''
        sim=require'sim'
        local j0,j1,j2,j3,j4,j5, q1,q2,q3,q4,q5,q6, dur = ...
        local js = {j0,j1,j2,j3,j4,j5}
        local dt = math.max(sim.getSimulationTimeStep(),0.01)
        local steps = math.max(math.floor(dur/dt),40)
        local q0 = {}
        for i=1,6 do q0[i]=sim.getJointPosition(js[i]) end
        for k=1,steps do
            local s=k/steps
            for i=1,6 do
                local tgt = ({q1,q2,q3,q4,q5,q6})[i]
                local q = q0[i] + s*(tgt - q0[i])
                sim.setJointTargetPosition(js[i], q)
            end
            sim.switchThread()
        end
    '''
    sim.runScriptString(lua,
                        H['j0'], H['j1'], H['j2'], H['j3'], H['j4'], H['j5'],
                        float(q_rad[0]), float(q_rad[1]), float(q_rad[2]),
                        float(q_rad[3]), float(q_rad[4]), float(q_rad[5]),
                        float(duration))

def _moveL_linear_world(p0, e0deg, p1, e1deg, duration):
    # Linear world path; IK step each frame on Lua side
    steps = max(int(duration / max(sim.getSimulationTimeStep(), 0.01)), 40)
    for k in range(1, steps+1):
        s = k/steps
        p = [ p0[i] + s*(p1[i]-p0[i]) for i in range(3) ]
        e = [ e0deg[i] + s*(e1deg[i]-e0deg[i]) for i in range(3) ]
        _lua_ik_set_pose_world(p, e)
        sim.switchThread()

def _get_joint_positions():
    # Read 6 joint positions in radians
    return [ sim.getJointPosition(j) for j in H['joints'] ]

def _set_gripper_open():
    # Open gripper (tune values to your model)
    if 'gripL' in H and 'gripR' in H:
        sim.setJointTargetPosition(H['gripL'], -0.02)
        sim.setJointTargetPosition(H['gripR'],  0.02)
    elif H.get('grip') is not None:
        sim.setJointTargetPosition(H['grip'], 0.00)

def _set_gripper_close():
    # Close gripper (tune values to your model)
    if 'gripL' in H and 'gripR' in H:
        sim.setJointTargetPosition(H['gripL'], 0.00)
        sim.setJointTargetPosition(H['gripR'], 0.00)
    elif H.get('grip') is not None:
        sim.setJointTargetPosition(H['grip'], 0.04)

def sysCall_thread():
    # Define poses (meters, degrees)
    start_pos = [0.000, 0.000, 0.0975]
    start_eul = [0.0, 0.0, 90.0]

    pick_pos  = [2.000, 0.000, 1.600]
    pick_eul  = [0.0, 0.0, 90.0]

    lift_pos  = [2.000, 0.000, 2.600]   # +1.0 m
    lift_eul  = [0.0, 0.0, 90.0]

    place_pos = [0.000, -1.600, 1.200]
    place_eul = [0.0, 0.0, 90.0]

    # Move to start pose (IK once)
    _lua_ik_set_pose_world(start_pos, start_eul)
    _set_gripper_open()
    sim.wait(0.3)

    # Small joint-space settle (current joints ? same joints)
    q_now = _get_joint_positions()
    _lua_moveJ_to(q_now, 0.5)

    # To pick
    _moveL_linear_world(start_pos, start_eul, pick_pos, pick_eul, 1.2)
    _set_gripper_close()
    sim.wait(0.2)

    # Lift +1.0 m
    _moveL_linear_world(pick_pos, pick_eul, lift_pos, lift_eul, 1.0)

    # To place
    _moveL_linear_world(lift_pos, lift_eul, place_pos, place_eul, 1.4)
    _set_gripper_open()
    sim.wait(0.2)

    # Back to start
    _moveL_linear_world(place_pos, place_eul, start_pos, start_eul, 1.2)

def sysCall_cleanup():
    # No explicit cleanup required
    pass
