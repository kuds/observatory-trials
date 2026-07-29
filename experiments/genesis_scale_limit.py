"""Does Genesis contact resolution survive watch-part scale?

Abstract two-jaw gripper squeezes a cube of side S and lifts it. Sweeps S from
macro (40 mm) down to watch-component scale, reporting penetration as a FRACTION
OF PART SIZE -- the number that matters for micro-assembly, where an absolute
penetration that is negligible on a 40 mm cube is the entire feature size on a
0.1 mm pivot.

Usage: python scale_test.py <side_mm> [precision 32|64]
"""
import sys
import os
import tempfile

import numpy as np
import genesis as gs

SIDE = float(sys.argv[1]) / 1000.0
PREC = sys.argv[2] if len(sys.argv) > 2 else "32"
STEEL = 7800.0

gs.init(backend=gs.cpu, precision=PREC, logging_level="warning")

jaw_x, jaw_t, jaw_z = SIDE * 0.6, SIDE * 0.15, SIDE * 0.35
gap = SIDE * 0.8                       # open half-gap
carriage_z = SIDE * 0.55               # jaws straddle mid-part, clear of the ground

car_mass = STEEL * (jaw_x * jaw_t * jaw_z * 8)
car_in = car_mass * jaw_x ** 2 / 6.0
part_mass = STEEL * SIDE ** 3

xml = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "micro_gripper.xml")).read()
for k, v in [("JAWX", jaw_x), ("JAWT", jaw_t), ("JAWZ", jaw_z), ("JAWOFF", gap),
             ("CARMASS", car_mass), ("CARIN", car_in)]:
    xml = xml.replace(k, repr(v))
f = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
f.write(xml)
f.close()

scene = gs.Scene(sim_options=gs.options.SimOptions(dt=1 / 2000, substeps=4), show_viewer=False)
scene.add_entity(gs.morphs.Plane())
part = scene.add_entity(gs.morphs.Box(size=(SIDE, SIDE, SIDE), pos=(0.0, 0.0, SIDE / 2)))
grip = scene.add_entity(gs.morphs.MJCF(file=f.name, pos=(0.0, 0.0, carriage_z)))
scene.build()

dofs = np.arange(3)
grip.set_dofs_kp(np.full(3, 2e4 * (car_mass + part_mass)), dofs)
grip.set_dofs_kv(np.full(3, 5e2 * (car_mass + part_mass)), dofs)

close_to = -(gap - SIDE / 2 + SIDE * 0.03)   # both jaws close on negative q
pen_frac_max = 0.0


def run(lift, close, n):
    global pen_frac_max
    for _ in range(n):
        grip.control_dofs_position(np.array([lift, close, close]), dofs)
        scene.step()
        p = np.asarray(part.get_contacts()["penetration"]).ravel()
        if p.size:
            pen_frac_max = max(pen_frac_max, p.max() / SIDE)


run(0.0, 0.0, 400)          # settle with jaws open
run(0.0, close_to, 1200)    # close and squeeze -- lift held at zero throughout

n_contacts = np.asarray(part.get_contacts()["penetration"]).ravel().size
q_close = np.asarray(grip.get_dofs_position().cpu())[1]

run(SIDE * 8, close_to, 2500)   # lift by 8 part-heights
run(SIDE * 8, close_to, 800)    # settle

pos = np.asarray(part.get_pos().cpu())
lifted = (pos[2] - SIDE / 2) / SIDE
held = bool(lifted > 4.0)

print(f"side={SIDE*1000:>8.3f} mm  fp{PREC}  part_mass={part_mass:.3e} kg  "
      f"contacts={n_contacts:>2d}  jaw_q={q_close:+.6f}/{close_to:+.6f}  "
      f"max_pen={pen_frac_max*100:>7.3f}% of part  "
      f"lifted={lifted:>6.2f} heights  HELD={held}")
os.unlink(f.name)
