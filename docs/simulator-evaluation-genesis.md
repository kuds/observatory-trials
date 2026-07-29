# Genesis as the simulator for fine / nimble grasping

**Status:** evaluation, not a decision. July 2026.
**Scope:** whether the Genesis simulation environment is a good option for the
fine-grained, nimble grasping tier of this project.

---

## 0. A note on the plan

There is no plan in this repository. `main` contains a single commit whose only
content is a one-line `README.md` (`# observatory-trials`). There are no issues,
no pull requests, no design docs, and no other branches carrying content. I also
searched the connected Drive for a simulator/platform plan and found nothing
matching.

So this document does **not** review a proposed set of simulation options — it
can't. Instead it evaluates Genesis on its merits for fine grasping, against the
alternatives a project like this would realistically be choosing among, and
states its assumptions explicitly:

- **A1** — there is a "bulk" tier (locomotion / pick-and-place / vision policies
  trained at scale) and a "fine" tier (dexterous, contact-rich, tolerance-sensitive
  grasping). This document is about the fine tier only.
- **A2** — sim-to-real transfer to physical hardware is a goal, not just
  in-sim benchmarking.
- **A3** — GPU training capacity is available; the numbers below were measured
  CPU-only and are not throughput guidance.

If the actual plan says something different, the recommendation in §6 may change.
Point me at it and I'll redo this against the real document.

---

## 1. Verdict

**Yes — Genesis is a good option for the fine grasping tier, but as a specialist
second track, not as the program's backbone.**

The reason to pick Genesis is *not* speed. In 2026 that argument is dead (§4.6).
The reason is that Genesis is the only engine that puts **batched tactile
sensing, intersection-free IPC contact, and rigid–deformable coupling in one
scene with one state**. If the fine-grasping work is genuinely about tactile
feedback, soft fingertips, or deformable objects, that combination does not exist
elsewhere and is worth real integration cost.

The reason to keep it off the critical path is maturity: the specific contact
features that make it attractive for grasping are **weeks old**, and its grasp
behavior is measurably sensitive to contact parameters (§4.1).

---

## 2. What Genesis actually is, as of July 2026

| | |
|---|---|
| Package | `genesis-world`, v1.3.0 (installed and tested below) |
| Repo | `Genesis-Embodied-AI/genesis-world` |
| License | Apache 2.0 |
| Backing | started as an academic project (Dec 2024); development now officially supported by Genesis AI, the commercial entity behind the GENE-26.5 manipulation model |
| Solvers | Rigid, FEM, MPM, Particle (PBD/SPH), `uipc` (IPC), SAP, plus an explicit coupler — all sharing one scene and one state |
| Backends | CUDA, AMD ROCm, Apple Metal, Vulkan, x86, ARM64 via the Quadrants compiler |

Note the naming collision: "Genesis" is both the open-source engine and the
company (Genesis AI). Vendor performance claims about the platform are not the
same thing as properties of the Apache-2.0 engine you would `pip install`.

---

## 3. Why it's a genuinely strong fit for fine grasping

### 3.1 Tactile sensing is first-class, in-engine, and batched — the real differentiator

This is the single strongest argument. Verified directly on the installed
package — `gs.sensors` exposes, among others:

`ContactForce`, `Contact`, `ContactProbe`, `ContactDepthProbe`,
`ElastomerTaxel`, `KinematicTaxel`, `ProximityTaxel`, `SurfaceDistanceProbe`,
`TemperatureGrid`, `JointTorque`, `IMU`, `Raycaster`, `Lidar`, `DepthCamera`

— composed with `ViscoelasticHysteresisOptionsMixin` and
`SpatialCrosstalkOptionsMixin`, i.e. the sensors model hysteresis and
cross-taxel bleed rather than returning idealized contact values.

The supporting result is *Tactile Genesis: Exploring Tactile Sensors at Scale for
Learning Dexterous Tasks* (arXiv 2606.22332), which simulates seven tactile
modalities — binary contact, contact depth, per-taxel force/torque, elastomer
marker displacement, geometry-aware proximity, contact audio, temperature — across
**20,000+ parallel environments**, trains dexterous policies on an XHand, and
**transfers them to the real XHand1**. Its findings are directly actionable for a
grasping program:

- whole-hand sensor coverage substantially outperforms fingertip-only placement;
- **per-taxel force/torque is consistently the most useful modality**;
- resolution matters little — ~200 taxels across the hand suffices;
- proprioception alone is insufficient on every task tested.

Neither MuJoCo nor Isaac Lab offers this in-engine at this fidelity. In those
stacks, taxel simulation is something you build and validate yourself.

### 3.2 Intersection-free contact (IPC) with articulation coupled in

Genesis World 1.0 added an **External Articulation Constraint** built on
`libuipc` that embeds joint-space dynamics directly into IPC's optimization, so
**joint-space forces and contact forces resolve simultaneously** rather than being
staggered across separate solvers. A "barrier-free elastodynamics" formulation
replaces IPC's logarithmic barrier with an augmented Lagrangian, reported at up to
**103× faster than traditional IPC** in contact-heavy scenes while preserving
intersection-free guarantees.

For fine grasping this matters more than it sounds: intersection-free means thin
objects and tight tolerances don't tunnel or interpenetrate, which is exactly the
failure mode that makes precision grasping unreliable in penalty-based solvers.

### 3.3 Rigid–deformable coupling in the same scene

Soft fingertips, elastomer pads, cloth, cables, food. If any of these are in
scope, having FEM/MPM/PBD coupled to the rigid solver in one state is a large
practical saving over bolting a second simulator onto MuJoCo or Isaac.

### 3.4 The recent solver work is precisely the grasping-relevant work

The last three releases read like a fine-grasping changelog:

- **v1.3.0** — torsional and rolling friction added to the rigid solver
  (critical for in-hand rotation, pivoting, and rolling contacts); differentiable
  rigid-body simulation promoted beyond experimental.
- **v1.2.3** — elliptic friction cone with a high-impedance option to accurately
  model **static friction**; constraint-solver convergence improved under fp32.
- **v1.2.2** — realistic tactile sensor suite for RL; robust non-convex collision
  detection fixing **spurious deep contacts and thin-shell tunneling**.

This is a strong positive signal about direction — and simultaneously the
maturity risk in §4.2, because it means these behaviors are new.

### 3.5 Differentiability

Differentiable rigid-body simulation is no longer experimental as of v1.3.0.
Useful for grasp-pose optimization and, more valuably, for **system
identification** — fitting friction and compliance parameters against real
hardware logs rather than hand-tuning them.

### 3.6 Apache 2.0

Permissive, commercially usable, no negotiation needed.

---

## 4. Where it will hurt

### 4.1 Measured: grasp stability is sharply sensitive to grip force

I installed `genesis-world==1.3.0` (CPU, Python 3.11, torch 2.13) and ran a
Franka Panda pinch-grasp-and-lift on a 4 cm cube, sweeping only the gripper
command. Script: [`experiments/genesis_grasp_smoke.py`](../experiments/genesis_grasp_smoke.py).

| Gripper command | Contacts | Max penetration | Result |
|---|---|---|---|
| force, −0.5 N | 9 | 0.051 mm | **held**, lifted to z = 0.229 m |
| force, −5.0 N | 6 | 0.139 mm | **ejected** — cube shot sideways |
| position, 0.018 m | 12 | 0.035 mm | **ejected** |
| position, 0.012 m | 12 | 0.074 mm | **held**, lifted to z = 0.230 m |

Two readings, and both matter:

**Good:** contact resolution is genuinely fine. Maximum penetration stayed
between 0.03 and 0.14 mm on a 40 mm object — sub-0.4% of object scale. The engine
resolves contact geometry well, and a correct grasp is rock-solid (lateral drift
under 1 cm through a 23 cm lift).

**Concerning:** a 10× increase in grip force flips a stable grasp into ejection.
On real hardware, a Panda squeezing a 4 cm cube at 5 N holds it comfortably. This
is squeeze-out instability, and there is a specific likely culprit — on load,
Genesis warns:

```
(MJCF) Approximating tendon by joint actuator for `finger_joint1`
(MJCF) Approximating tendon by joint actuator for `finger_joint2`
```

The Panda's fingers are mechanically coupled by a tendon; Genesis approximates
this as two independent joint actuators. Decoupled fingers let asymmetric contact
build a net lateral impulse that pops the object out. **For a program whose entire
premise is fine grasping, characterizing this sensitivity is the first thing to
do — not an afterthought.** It is very possibly tunable (friction, solver
impedance, `condim`); it is not safe to assume so.

### 4.2 Version churn on exactly the features you'd depend on

v1.0.0 shipped late May 2026; v1.3.0 in July 2026 — four minor releases in about
two months, and the friction and contact semantics changed in three of them.
Contact behavior *will* move under you. Pin an exact version, and treat a Genesis
upgrade as an event that requires re-validating grasp policies, not a routine bump.

### 4.3 Known open contact bugs

- Changing the friction vector reportedly does not affect contact behavior
  (`Genesis` issue #1139) — near-zero friction and `condim` adjustments failed to
  produce expected rotational slippage. If true and unresolved, that directly
  undercuts friction-based grasp studies.
- Gripper cannot maintain grasp on cloth; slipping (`genesis-world` issue #199).

Both should be re-checked against 1.3.0 before committing.

### 4.4 Headless operation needs work

`scene.build()` **unconditionally constructs the rasterizer** — "Rasterizer is
always needed for depth and segmentation mask rendering" — even with
`show_viewer=False` and no cameras added. On a GPU-less container this crashes in
EGL init. Workaround: install `libosmesa6` and set `PYOPENGL_PLATFORM=osmesa`.
Budget for this in CI and on headless cluster nodes.

Also, scene build is slow: 10–90 s for a Franka + cube, dominated by convex
decomposition of the robot meshes. Cache decompositions.

### 4.5 Sim-to-real evidence is thinner and largely first-party

Genesis AI reports 89% correlation between simulation and real-world rollouts and
a 45% smaller reality-gap FID than alternatives. These are vendor-reported, on
their own evaluation harness, for their platform. The strongest *independent-style*
evidence for grasping specifically is the Tactile Genesis XHand transfer (§3.1) —
real, but a narrow base compared to the years of published MuJoCo and Isaac
manipulation transfer results.

### 4.6 Speed is no longer a differentiator — and the old claims were overstated

Genesis's headline "10–80× faster than Isaac Gym / MuJoCo" was criticized because
the comparison was largely against **single-threaded MuJoCo and Isaac Gym, with
MJX absent from the benchmark entirely** (see the DeepMind MuJoCo discussion
#2303). Treat the historical throughput marketing as unreliable.

More importantly, the landscape moved. **MuJoCo Warp** (MuJoCo 3.5) reports
**252× faster locomotion and 475× faster manipulation than MJX** on an RTX PRO
6000 Blackwell (152×/313× on a 4090), and **NVIDIA Newton 1.0 went GA at GTC
2026** as a production-ready foundation for dexterous manipulation. Whatever
throughput edge Genesis had in 2025 is gone.

### 4.7 The tension nobody states out loud

**Genesis's speed configuration and its fine-grasping-fidelity configuration are
not the same configuration.** The eye-catching FPS numbers come from the rigid
solver on simple scenes. The stack you actually want for nimble grasping — IPC
contact, FEM deformables, per-taxel tactile with hysteresis — is a different and
far more expensive path. Do not build a capacity plan that assumes you get both
at once. Measure the fidelity configuration you'll actually train in.

### 4.8 Ecosystem depth

MuJoCo Playground, Isaac Lab, robosuite, and ManiSkill have far more ready-made
manipulation environments, baselines, published hyperparameters, and sim-to-real
recipes. Choosing Genesis means writing more of that yourself.

---

## 5. Head-to-head for the fine-grasping tier

| Criterion | Genesis 1.3 | MuJoCo (Warp / 3.5) | Isaac Lab + Newton 1.0 |
|---|---|---|---|
| Contact accuracy, rigid | Good; sub-0.1 mm penetration measured; friction model recently overhauled | Gold standard, years of validation | Strong; Newton GA targets contact-rich |
| Intersection-free / IPC | **Yes (uipc + articulation constraint)** | No | No |
| Tactile / taxel sensing | **Best in class, in-engine, batched, with hysteresis + crosstalk** | Build it yourself | Build it yourself; strong vision instead |
| Deformables coupled to rigid | **Yes, one scene, one state** | Limited | Improving (cloth via Newton) |
| Differentiable | Yes (rigid, no longer experimental) | MJX via JAX | Limited |
| GPU throughput | Good, claims unreliable | **Excellent (MJWarp)** | **Excellent** |
| Photorealistic vision | Good | Batch renderer, improving | **Best in class (RTX, tiled)** |
| Maturity / API stability | **Weakest** — 4 minor releases in 2 months | **Strongest** | Strong, GA |
| Ecosystem / baselines | Thin | Deep (Playground) | Deep (Isaac Lab) |
| Sim-to-real track record for grasping | Narrow, promising | Extensive | Extensive |
| License | Apache 2.0 | Apache 2.0 | Mixed / NVIDIA-tied |

---

## 6. Recommendation

**Tier the simulators; don't pick one.**

1. **Backbone (bulk RL, vision, locomotion, coarse pick-and-place):** MuJoCo Warp
   or Isaac Lab + Newton. Mature, fast, deep ecosystem, well-understood transfer.

2. **Fine / nimble grasping tier:** Genesis — *conditionally*, and specifically
   when the research question involves tactile feedback, soft contact, or
   deformables. If the fine-grasping work is really just tighter tolerances on
   rigid objects, MuJoCo is the lower-risk answer and Genesis adds little.

3. **Do not** make Genesis the single environment the whole program depends on
   at this maturity level.

The decision hinges on one question I can't answer from an empty repo: **is
tactile sensing actually in scope?** If yes, Genesis is close to compelling and
the integration cost is justified. If no, the case weakens sharply.

---

## 7. Proposed bake-off before committing

Two weeks, one engineer, GPU access. Genesis vs MuJoCo Warp on identical tasks.

**Tasks** (increasing contact difficulty)
1. Pinch-grasp and lift a rigid cube — baseline, must be boring.
2. Grasp a thin object (card, washer, 2 mm plate) — tests tunneling and
   intersection-free contact; expected Genesis win.
3. In-hand pivot / reorientation — tests torsional and rolling friction (new in
   v1.3.0).
4. Grasp a deformable (sponge, cable) — Genesis-only capability in practice.
5. Tactile-conditioned slip detection and regrasp — the actual differentiator.

**Metrics**
- Grasp success rate across object poses and masses.
- **Sensitivity: success rate as a function of grip force and friction
  coefficient.** Given §4.1, this is the headline metric, not success rate at one
  tuned setpoint. A simulator whose success depends on a narrow force band is a
  simulator that will not transfer.
- Max penetration and contact-force realism vs. hardware measurement.
- Wall-clock throughput **in the fidelity configuration you'd actually train in**.
- Engineer-days to first working environment.

**Gates — Genesis proceeds to the fine tier only if:**
- G1: grasp success ≥ 90% across a ≥ 5× grip-force range on task 1. *(Currently
  failing: 0.5 N holds, 5 N ejects.)*
- G2: issue #1139 is resolved or a working friction-control path is demonstrated.
- G3: tasks 2 and 5 are demonstrably easier in Genesis than in MuJoCo — i.e. the
  differentiator is real, not theoretical.
- G4: a pinned version runs headless in CI without per-node GL hacks.

If G1 fails and cannot be tuned away, Genesis is not ready for *fine* grasping
regardless of how good the tactile story is.

---

## 8. Reproducing the measurements here

```bash
python -m venv venv && ./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install genesis-world==1.3.0
sudo apt-get install -y libosmesa6          # headless only; see §4.4
PYOPENGL_PLATFORM=osmesa ./venv/bin/python experiments/genesis_grasp_smoke.py force -0.5
PYOPENGL_PLATFORM=osmesa ./venv/bin/python experiments/genesis_grasp_smoke.py force -5.0
```

Measured on CPU only (no GPU in the evaluation container): 183–691 steps/s for a
single Franka + cube environment depending on timestep and substep count. **These
are not throughput guidance** — they exist only to confirm the engine runs and to
support the grasp-stability finding.

---

## 9. Open questions

1. Where is the actual plan, and what simulation options were already proposed?
2. Is tactile sensing in scope? This is the deciding question (§6).
3. What is the target hardware — parallel-jaw gripper, or a multi-finger hand
   like the XHand/Allegro? The Tactile Genesis evidence is hand-specific.
4. Are deformable objects in scope?
5. Is there physical hardware to validate against, or is this sim-only for now?

---

## Sources

- [Genesis-Embodied-AI/genesis-world (GitHub)](https://github.com/Genesis-Embodied-AI/genesis-world)
- [genesis-world releases](https://github.com/Genesis-Embodied-AI/genesis-world/releases)
- [Tactile Genesis: Exploring Tactile Sensors at Scale for Learning Dexterous Tasks (arXiv 2606.22332)](https://arxiv.org/abs/2606.22332)
- [The Role of Simulation in Scalable Robotics, Genesis World 1.0, and the Path Forward (Genesis AI)](https://www.genesis.ai/blog/the-role-of-simulation-in-scalable-robotics-genesis-world-10-and-the-path-forward)
- [Genesis AI Releases Nyx, Quadrants, and Genesis World 1.0 (MarkTechPost)](https://www.marktechpost.com/2026/05/30/genesis-ai-releases-nyx-quadrants-and-genesis-world-1-0-physics-platform-for-scalable-robotics-foundation-model-evaluation/)
- [Genesis speed claims discussion — google-deepmind/mujoco #2303](https://github.com/google-deepmind/mujoco/discussions/2303)
- [Bug: changing friction vector doesn't affect contact behavior — Genesis #1139](https://github.com/Genesis-Embodied-AI/Genesis/issues/1139)
- [Robot gripper cannot grasp cloth (slipping) — genesis-world #199](https://github.com/Genesis-Embodied-AI/genesis-world/issues/199)
- [MuJoCo Warp (MJWarp) documentation](https://mujoco.readthedocs.io/en/latest/mjwarp/)
- [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp)
- [Newton Adds Contact-Rich Manipulation and Locomotion Capabilities (NVIDIA)](https://developer.nvidia.com/blog/newton-adds-contact-rich-manipulation-and-locomotion-capabilities-for-industrial-robotics/)
- [Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning (arXiv 2511.04831)](https://arxiv.org/pdf/2511.04831)
- [A Survey of Robotic Navigation and Manipulation with Physics Simulators in the Era of Embodied AI (arXiv 2505.01458)](https://arxiv.org/html/2505.01458v1)
