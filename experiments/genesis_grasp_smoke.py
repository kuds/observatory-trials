"""Genesis grasp-stability smoke test.

Franka Panda pinch-grasps a 4 cm cube and lifts it, sweeping only the gripper
command. Used to characterize how sensitive grasp stability is to grip force /
closure -- see docs/simulator-evaluation-genesis.md section 4.1.

Usage:
    python genesis_grasp_smoke.py force -0.5     # newtons, force-controlled fingers
    python genesis_grasp_smoke.py pos    0.012   # metres, position-controlled fingers

Headless without a GPU also needs `apt-get install libosmesa6` and
PYOPENGL_PLATFORM=osmesa, because Genesis 1.3 builds the rasterizer
unconditionally even when show_viewer=False and no cameras are added.
"""

import sys
import time

import numpy as np
import genesis as gs

MODE = sys.argv[1]         # "force" | "pos"
GRIP = float(sys.argv[2])  # newtons if force, metres if pos

gs.init(backend=gs.cpu, logging_level="warning")

scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1 / 100, substeps=4), show_viewer=False)
scene.add_entity(gs.morphs.Plane())
cube = scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)))
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

# Fingertip force readout, to confirm the tactile sensor API wires up end to end.
contact_force = scene.add_sensor(
    gs.sensors.ContactForce(
        entity_idx=franka.idx,
        link_idx_local=franka.get_link("left_finger").idx_local,
    )
)

scene.build()

arm, fingers, all_dofs = np.arange(7), np.array([7, 8]), np.arange(9)
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]), all_dofs)
franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]), all_dofs)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    all_dofs,
)

ee = franka.get_link("hand")
DOWN = np.array([0, 1, 0, 0])


def ik(pos):
    return np.asarray(franka.inverse_kinematics(link=ee, pos=np.array(pos), quat=DOWN).cpu())[:7]


def glide(q_from, q_to, n, grip_pos=None, grip_force=None):
    """Ramp the arm target over n steps -- a step change makes the high-gain
    controller slam the gripper through the cube before it can close."""
    for i in range(n):
        franka.control_dofs_position(q_from + (q_to - q_from) * ((i + 1) / n), arm)
        if grip_force is None:
            franka.control_dofs_position(np.array([grip_pos, grip_pos]), fingers)
        else:
            franka.control_dofs_force(np.array([grip_force, grip_force]), fingers)
        scene.step()


q_home = np.asarray(franka.get_qpos().cpu())[:7]
q_pre = ik([0.65, 0.0, 0.25])
q_grasp = ik([0.65, 0.0, 0.135])
q_lift = ik([0.65, 0.0, 0.35])

squeeze = {"grip_force": GRIP} if MODE == "force" else {"grip_pos": GRIP}

glide(q_home, q_pre, 250, grip_pos=0.04)      # rise to pre-grasp, gripper open
glide(q_pre, q_grasp, 250, grip_pos=0.04)     # descend around the cube
glide(q_grasp, q_grasp, 150, **squeeze)       # squeeze

z_after_squeeze = cube.get_pos()[2].item()
force = np.asarray(contact_force.read()).ravel()
penetration = np.asarray(cube.get_contacts()["penetration"]).ravel()

glide(q_grasp, q_lift, 400, **squeeze)        # lift
glide(q_lift, q_lift, 150, **squeeze)         # settle

pos = np.asarray(cube.get_pos().cpu())

print(f"mode / command        : {MODE} {GRIP}")
print(f"cube z after squeeze  : {z_after_squeeze:.4f} m")
print(f"fingertip force (N)   : {np.round(force, 2).tolist()}")
if penetration.size:
    print(f"contacts on cube      : {penetration.size}, max penetration {penetration.max() * 1000:.4f} mm")
else:
    print("contacts on cube      : 0")
print(f"cube pos after lift   : {np.round(pos, 4).tolist()}")
print(f"lateral drift         : {np.linalg.norm(pos[:2] - [0.65, 0.0]):.4f} m")
print(f"GRASP HELD            : {bool(pos[2] > 0.20)}")

t0 = time.perf_counter()
for _ in range(300):
    scene.step()
print(f"CPU throughput        : {300 / (time.perf_counter() - t0):.1f} steps/s (1 env, dt=1/100, 4 substeps)")
