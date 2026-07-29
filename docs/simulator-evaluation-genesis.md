# Simulator selection for micro-manipulation (watch assembly)

**Status:** evaluation, not a decision. July 2026.
**Scope:** choosing a simulation environment for fine-grained *micro*-manipulation
— watch-building-class assembly of sub-millimetre to few-millimetre components.
Genesis is the primary candidate under review; MuJoCo, Isaac Lab + Newton, and
Drake are the comparators.

> **Revision note.** An earlier draft of this document evaluated Genesis for
> "fine grasping" generically. The scope was then clarified to *micro*-manipulation
> at watch-component scale. That change is material — it weakens the strongest
> original argument for Genesis (batched tactile arrays) and promotes a candidate
> the first draft under-weighted (Drake). §3 and §7 are the revised conclusions.

---

## 0. On the plan

There is no plan in this repository. `main` is a single commit containing a
one-line `README.md`. No issues, no PRs, no design docs, no other content-carrying
branches; nothing matching in the connected Drive either. So this document does
not review a proposed shortlist — it builds one.

Working assumptions, to be corrected if wrong:

- **A1** — target components are watch-scale: hairspring wire ~0.03 mm, pivots
  ~0.07–0.12 mm, screws ~0.3–1.4 mm, jewels ~1–1.5 mm, wheels and balance
  ~5–10 mm. So the workspace spans **roughly two and a half orders of magnitude**,
  straddling the boundary where surface forces overtake gravity.
- **A2** — end effector is tweezer- or micro-gripper-like under magnification,
  not a multi-finger anthropomorphic hand.
- **A3** — the goal is eventually sim-to-real, not sim-only benchmarking.
- **A4** — motion is largely quasi-static and precision-dominated rather than
  dynamic and throughput-dominated.

**A2 and A4 are the assumptions most likely to change the answer.** If you
actually intend multi-finger dexterous hands, or need massive-scale RL, §7 shifts
back toward Genesis.

---

## 1. The finding that dominates everything else

**At watch-component scale, the physics that decides success is not the physics
these simulators model.**

Below roughly 100 µm, van der Waals, capillary, and electrostatic forces dominate
part interactions; for a 1 µm particle, adhesion exceeds gravity by a factor of
>10⁶. The practical consequence is the **release problem**, a long-standing
open challenge in micromanipulation: picking a micro-part up is easy, *putting it
down where you want it* is not, because it clings to the tool. Any watchmaker
recognizes this — the part vanishes onto the tweezers, or pings off the bench.

**None of Genesis, MuJoCo, Isaac Lab, or Drake model adhesion, capillary bridging,
or electrostatics.** All four model gravity, contact, and friction. Out of the
box, **every candidate simulates the wrong dominant physics** for the smaller end
of your part range.

This reframes the selection criteria. The top question is no longer "which engine
has the best contact solver?" It is:

> **Which engine will let me inject a custom adhesion/capillary force model and
> calibrate it against real hardware?**

Contact fidelity still matters — it decides whether the 1–10 mm components
(wheels, jewels, screws, plates) behave — but for anything below ~0.1 mm, a
stock simulator will be confidently wrong regardless of which one you choose.

---

## 2. Measured: where Genesis actually breaks at small scale

I installed `genesis-world==1.3.0` (CPU, Python 3.11, torch 2.13) and ran two
experiments. Scripts are in [`experiments/`](../experiments/).

### 2.1 Contact accuracy degrades as parts shrink

An abstract two-jaw gripper closes on a cube and lifts it. Geometry is *identical
in relative terms* at every scale — only absolute size changes. Penetration is
reported as a **fraction of part size**, which is the number that matters: 0.05 mm
of interpenetration is 0.1% of a 40 mm cube and irrelevant, but is the entire
diameter of a watch pivot.

| Part side | Max penetration (% of part) | Contacts |
|---|---|---|
| 40 mm | 5.1 % | 14 |
| 10 mm | 8.1 % | 14 |
| 4 mm | 16.6 % | 14 |
| 1 mm | 22.7 % | 4 |
| 0.3 mm | **model rejected — see 2.2** | — |

Relative interpenetration grows monotonically as parts shrink, at fixed timestep.
fp32 vs fp64 made no meaningful difference from 40 mm down to 4 mm.

**It is recoverable, at a price.** Re-running the 1 mm case with a finer timestep:

| Timestep | Max penetration (% of part) |
|---|---|
| 1/2000 s | 22.7 % |
| 1/20 000 s | 2.7 % |
| 1/100 000 s | 2.6 % |

So accuracy is buyable with timestep — roughly **50× the compute** to bring a
1 mm part back to the relative fidelity a 40 mm part gets for free. Any
throughput projection for this project must be made at the timestep micro-parts
actually require, not at default settings.

### 2.2 A hard floor at ~0.3 mm: `mjMINVAL`

At 0.3 mm the model does not build at all:

```
ValueError: Error: mass and inertia of moving bodies must be larger than mjMINVAL
Element name 'carriage', id 1, line 5
```

Genesis parses MJCF through **MuJoCo's model compiler**, so it inherits MuJoCo's
`mjMINVAL` floor (1e-15) on body mass and rotational inertia. A 0.3 mm steel
component has inertia ≈ 2.9e-16 kg·m² — **below the floor**. In SI metres, watch
parts are not representable.

This is the single most actionable finding in this document, and it is **not
Genesis-specific** — MuJoCo hits it identically, for the same reason.

**Consequence: unit rescaling is mandatory, not optional.** You must simulate in
rescaled units (e.g. millimetres-as-metres, ×1000) and rescale gravity and time
consistently to keep the dynamics right. This is a foundational decision that
touches every asset, controller gain, and force reading in the project, so make
it deliberately on day one rather than discovering it later.

### 2.3 Honest limits of these measurements

- The grasp **success/failure** column from my harness at 1 mm and at 1000 mm is
  **confounded by my own controller-gain scaling** (gains scale with part mass and
  become too weak at the small end). Do not read "failed to lift at 1 mm" as a
  physics result. The penetration-versus-scale trend and the `mjMINVAL` floor are
  the solid findings; the lift outcomes at the extremes are not.
- CPU-only container, no GPU. No throughput guidance here.
- An abstract jaw gripper is not a tweezer, and a cube is not a jewel.

### 2.4 Carried over from the general-purpose evaluation

Still relevant, measured on a Franka + 40 mm cube:

- **Grasp stability is sharply grip-force sensitive** — 0.5 N holds, 5 N ejects
  the cube, where real hardware would hold comfortably. Genesis warns
  `Approximating tendon by joint actuator for finger_joint1`: it models the
  Panda's mechanically coupled fingers as independent actuators, so asymmetric
  contact builds a lateral impulse that pops the object out. Whether this
  generalizes to a custom micro-gripper model is untested, but it is a caution
  about contact-parameter sensitivity in exactly the regime you care about.
- **Contact resolution itself is good** at macro scale — sub-0.1 mm penetration,
  under 1 cm drift through a 23 cm lift.
- **Headless needs work** — `scene.build()` constructs the rasterizer
  unconditionally even with `show_viewer=False` and no cameras; needs
  `libosmesa6` + `PYOPENGL_PLATFORM=osmesa` on a GPU-less box.
- **Version churn** — v1.0.0 (late May 2026) → v1.3.0 (July 2026), with friction
  and contact semantics changing in three of four releases. Pin the version.
- **Open contact bugs** — friction vector changes reportedly not affecting contact
  behavior (Genesis #1139); gripper cannot hold cloth (genesis-world #199).

---

## 3. How the micro scope changes the case for Genesis

### Weakened

**The tactile-array argument, which was the strongest pro-Genesis point in the
general case, largely does not apply here.** Genesis's standout capability is
batched per-taxel tactile sensing (`ElastomerTaxel`, `KinematicTaxel`,
`ProximityTaxel`, with viscoelastic hysteresis and spatial crosstalk), validated
by *Tactile Genesis* on a 20 000-environment XHand study with real-hardware
transfer. That is a **multi-finger-hand** capability. Watch assembly under a
microscope uses tweezers and micro-grippers; there is no taxel array to simulate,
and the Tactile Genesis findings (whole-hand coverage beats fingertip-only, ~200
taxels suffice) do not transfer to this setting.

Genesis's throughput story also matters less under **A4** — precision-dominated,
quasi-static work does not need 20 000 parallel environments, and §2.1 shows the
timestep you need would erode the advantage anyway.

### Strengthened

**Intersection-free IPC contact matters *more* here, not less.** §2.1 shows
relative interpenetration is the characteristic micro-scale failure. Genesis's
`uipc` integration with the External Articulation Constraint — which resolves
joint-space and contact forces simultaneously and maintains intersection-free
guarantees, with a barrier-free formulation reported up to 103× faster than
classical IPC — attacks precisely that failure mode. This is now the **primary**
technical argument for Genesis on this project.

**FEM deformables become essential rather than nice-to-have.** A hairspring is a
spiral of ~0.03 mm wire; a mainspring is a coiled elastic strip. These are not
rigid bodies in any useful approximation. Genesis couples FEM/MPM/PBD to the rigid
solver in one scene with one state. In MuJoCo or Isaac this is painful or absent.
**If hairsprings and mainsprings are in scope, this may be decisive on its own.**

**Differentiability** (rigid sim, non-experimental as of v1.3.0) is more valuable
here than in the general case — the calibration problem in §1 is exactly a
parameter-fitting problem: given real pick-and-release trials, fit the adhesion
and friction parameters. Gradients help.

**Torsional and rolling friction** (new in v1.3.0) matter for the rolling and
pivoting contacts of small cylindrical parts.

---

## 4. The candidate Drake, which the first draft under-weighted

For **precision-first, throughput-second** contact work, Drake is a serious
contender and arguably a better fit than any of the RL-oriented engines:

- **Double precision throughout.** Not a flag — the design point. Given §2, this
  is not a minor consideration.
- **SAP (Semi-Analytic Primal) contact solver** with theoretically guaranteed
  global convergence, stable at larger timesteps. (Notably, Genesis lists SAP
  among its own solvers — the idea's provenance is Drake.)
- **Hydroelastic contact** — a continuous pressure field over a finite contact
  *patch* rather than a set of point contacts. For small parts with tight
  tolerances this is far better conditioned than point contact, and it is the
  model NVIDIA cites as inspiration for Newton's contact-rich work.
- **Demonstrated on peg-in-hole insertion**, which is structurally the same
  problem as setting a pivot into a jewel.

Drake's cost: no massive parallel RL story, thinner learning ecosystem, steeper
C++/Python systems framework, and no first-class deformables to speak of. If you
need hairsprings, Drake alone won't do it.

---

## 5. Comparison for *this* task

| Criterion (micro-assembly weighting) | Genesis 1.3 | Drake | MuJoCo (Warp) | Isaac Lab + Newton |
|---|---|---|---|---|
| Models the dominant micro physics (adhesion) | **No** | **No** | **No** | **No** |
| Extensibility to inject custom force models | Good (Python, differentiable) | Good (force elements) | Good (callbacks/plugins) | Moderate |
| Numerical conditioning at sub-mm | fp64 available; `mjMINVAL` floor via MJCF | **Double precision by design** | Same `mjMINVAL` floor | Untested here |
| Intersection-free contact | **Yes (uipc + articulation)** | No (but hydroelastic patches) | No | No |
| Contact model quality for precision fits | Good | **Hydroelastic — best fit** | Very good | Good |
| Deformables (hairspring, mainspring) | **Yes, coupled, one scene** | Weak | Limited | Improving |
| Differentiable | Yes | Partial (AutoDiff; not with SAP contact) | MJX | Limited |
| Parallel RL throughput | Good | Weak | **Excellent** | **Excellent** |
| Photoreal vision | Good | Weak | Improving | **Best** — but less relevant under a microscope |
| Maturity / API stability | **Weakest** | **Strongest** | Strong | Strong |
| License | Apache 2.0 | BSD-3 | Apache 2.0 | NVIDIA-tied |

---

## 6. What none of this solves

Worth stating plainly before any recommendation: **you should not expect
sim-to-real transfer for sub-0.1 mm manipulation from any of these tools without
building custom physics.** The literature treats micro-scale capture and release
as requiring explicit models of van der Waals, capillary, electrostatic, and
pull-off forces. That work is yours to do regardless of engine.

A realistic framing is a **two-regime project**:

- **≥ ~1 mm** (plates, bridges, wheels, jewels, screws) — a stock engine with
  rescaled units and a tightened timestep can be genuinely useful today.
- **< ~0.1 mm** (pivots, hairsprings) — treat as a physics-modelling research
  problem with a simulator as substrate, not as a simulation task.

If the project's value is concentrated in the second regime, simulator choice is
much less important than the adhesion-modelling work, and the honest
recommendation is to prototype that model first, in whatever is quickest.

---

## 7. Recommendation

**Split the decision by regime, and start with a rescaling spike, not an engine
commitment.**

1. **Before choosing anything — do the unit-rescaling spike (1–2 days).** Establish
   the unit convention (mm-as-m, with gravity and time rescaled consistently) and
   verify it clears `mjMINVAL` and holds contact accuracy at 0.1 mm. This work is
   engine-independent and is a prerequisite for every option. Doing it first will
   teach you more about the real constraints than any further comparison.

2. **Primary recommendation: Genesis — but only if deformables are in scope.**
   If hairsprings, mainsprings, or compliant micro-grippers are part of the target,
   Genesis is the only candidate that does rigid + FEM + intersection-free IPC in
   one scene, and that combination is the right shape for this problem. Accept the
   maturity risk and pin the version.

3. **If deformables are *not* in scope: prefer Drake.** For rigid micro-assembly
   dominated by precision — pivot-into-jewel, screw placement, wheel seating —
   Drake's double precision, SAP convergence guarantees, and hydroelastic contact
   are a better match than Genesis's newer, faster-moving stack, and the RL
   throughput you'd give up is not something A4 needs.

4. **Do not choose Isaac Lab for this.** Its principal strengths — photorealistic
   vision at scale, massive parallel RL — are the two things this task needs least.

5. **MuJoCo remains the safe fallback** if ecosystem maturity outweighs everything,
   but note it shares the `mjMINVAL` floor and lacks both IPC and deformables.

**The question that decides between (2) and (3): are hairsprings and mainsprings
in scope, or is this rigid-part assembly only?** I can't answer that from here,
and it flips the recommendation.

---

## 8. Revised bake-off

Three weeks, one engineer. Genesis vs Drake, on rescaled units from the start.

**Step 0 — rescaling spike** (see §7.1). Gate: a 0.1 mm part builds and holds
contact with <2% relative penetration.

**Tasks, in increasing order of what they'd actually prove**
1. Seat a 1 mm jewel into a bore — baseline precision fit.
2. Insert a 0.1 mm pivot into a jewel hole — the real target; tests whether
   relative-penetration control survives at the small end.
3. Place and drive a 0.5 mm screw — threaded contact, torsional friction.
4. Deform a hairspring — Genesis-only in practice; skip if out of scope.
5. **Pick *and release* a 0.3 mm part with an injected adhesion model** — the
   task that actually reflects §1.

**Metrics**
- Relative penetration as a fraction of feature size (not absolute).
- Placement accuracy vs. tolerance budget for the real assembly.
- **Sensitivity of success to contact and friction parameters** — given §2.4, a
  narrow band of working parameters is a red flag for transfer.
- Wall-clock at the timestep micro-parts actually need (§2.1), not at defaults.
- Engineer-days to inject a custom adhesion force.

**Gates**
- G1: unit convention clears `mjMINVAL` for the smallest target part.
- G2: <2% relative penetration at 0.1 mm at a tractable timestep.
- G3: a custom adhesion/pull-off force can be injected and calibrated — if this
  is hard in an engine, that engine is disqualified for the sub-0.1 mm regime
  regardless of its other merits.
- G4: success rate stable across a ≥5× range of contact-stiffness and friction
  parameters.
- G5 (Genesis only): pinned version runs headless in CI without per-node GL hacks.

---

## 9. Reproducing

```bash
python -m venv venv && ./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install genesis-world==1.3.0
sudo apt-get install -y libosmesa6            # headless only

# scale sweep -- part side in mm, precision 32|64
PYOPENGL_PLATFORM=osmesa ./venv/bin/python experiments/genesis_scale_limit.py 40 32
PYOPENGL_PLATFORM=osmesa ./venv/bin/python experiments/genesis_scale_limit.py 1 32
PYOPENGL_PLATFORM=osmesa ./venv/bin/python experiments/genesis_scale_limit.py 0.3 32   # mjMINVAL

# macro grasp-force sensitivity
PYOPENGL_PLATFORM=osmesa ./venv/bin/python experiments/genesis_grasp_smoke.py force -0.5
PYOPENGL_PLATFORM=osmesa ./venv/bin/python experiments/genesis_grasp_smoke.py force -5.0
```

---

## 10. Open questions

1. **Are hairsprings/mainsprings in scope?** Decides Genesis vs Drake (§7).
2. **What is the smallest part you need to manipulate?** If ≥1 mm, this is a
   tractable simulation project. If 0.1 mm, it is a physics-modelling project.
3. **Is the end effector a tweezer/micro-gripper, or a multi-finger hand?** A
   hand would revive the tactile argument and swing this back to Genesis.
4. **Is there real hardware to calibrate an adhesion model against?** Without it,
   the sub-0.1 mm regime cannot be validated.
5. **Is this RL-driven, or planning/control-driven?** RL would re-weight
   throughput, which currently barely features in the recommendation.
6. Where is the actual plan, and what options were already proposed?

---

## Sources

- [Genesis-Embodied-AI/genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world) · [releases](https://github.com/Genesis-Embodied-AI/genesis-world/releases)
- [Tactile Genesis: Exploring Tactile Sensors at Scale for Learning Dexterous Tasks (arXiv 2606.22332)](https://arxiv.org/abs/2606.22332)
- [The Role of Simulation in Scalable Robotics, Genesis World 1.0 (Genesis AI)](https://www.genesis.ai/blog/the-role-of-simulation-in-scalable-robotics-genesis-world-10-and-the-path-forward)
- [Genesis speed-claim discussion — google-deepmind/mujoco #2303](https://github.com/google-deepmind/mujoco/discussions/2303)
- [Friction vector doesn't affect contact behavior — Genesis #1139](https://github.com/Genesis-Embodied-AI/Genesis/issues/1139)
- [Gripper cannot grasp cloth — genesis-world #199](https://github.com/Genesis-Embodied-AI/genesis-world/issues/199)
- [Drake: Modeling Compliant Contact](https://drake.mit.edu/doxygen_cxx/group__compliant__contact.html) · [hydroelastic contact tutorial](https://github.com/RobotLocomotion/drake/blob/master/tutorials/hydroelastic_contact_basics.ipynb)
- [Irrotational Contact Fields (arXiv 2312.03908)](https://arxiv.org/html/2312.03908v3)
- [Micro-Manipulation and Adhesion Forces (Springer)](https://link.springer.com/chapter/10.1007/978-3-7091-2498-7_28)
- [Adhesion force modeling and measurement for micromanipulation (SPIE)](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/3519/1/Adhesion-force-modeling-and-measurement-for-micromanipulation/10.1117/12.325737.short)
- [Manipulation of Microobjects Based on Dynamic Adhesion Control](https://journals.sagepub.com/doi/10.5772/51507)
- [Microassembly: A Review on Fundamentals, Applications and Recent Developments](https://www.engineering.org.cn/engi/EN/1159991032756626078)
- [Newton Adds Contact-Rich Manipulation and Locomotion Capabilities (NVIDIA)](https://developer.nvidia.com/blog/newton-adds-contact-rich-manipulation-and-locomotion-capabilities-for-industrial-robotics/)
- [MuJoCo Warp (MJWarp)](https://mujoco.readthedocs.io/en/latest/mjwarp/)
