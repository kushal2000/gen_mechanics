# Bimanual manipulation with sim-to-real transfer: literature report (2019 to Oct 2026)

Written 15 Oct 2026 for the twist-apart pen project. I checked every paper listed here against its arXiv page, project page or proceedings entry. Numbers are quoted only where I read them in the paper. Anything I could not check is in the "Unverified" list at the end.

Task kinds used throughout:
(a) one hand stabilises, the other operates; (b) opposing force/torque on one object (twist, pull apart, tear); (c) both hands control one large object (lift, carry); (d) handover or regrasp; (e) deformable objects; (f) independent or parallel jobs.

---

## 1. Summary

- **Sim-to-real RL with two multi-finger hands is new and small.** The first clear example is Lin et al., *Twisting Lids Off with Two Hands* (CoRL 2024). Since then about ten papers have done it, mostly in 2025–26. Most use 2×Allegro or 2×Shadow on fixed arms, Isaac Gym/Lab or MuJoCo, PPO with an asymmetric critic, and domain randomisation, often with teacher–student distillation.
- **Solved, with moderate reliability:** lifting a box with two hands (77–80% real success: Lin 2025, SymDex), handover (52.5% on a humanoid; throw-and-catch 33–60%), and table-top pick, place and insert (75–80%).
- **Twisting is the best-studied opposing-force task, and it is close to working.** Lid twisting: Lin 2024 gets 33.75% full lid removal over 8 household objects. AsymDex gets 18/20 on one jar. Grasp2Twist (Sep 2026 preprint) gets 53/60 (88%) over 6 jars with finger gaiting. In all three, one hand holds the container and the other turns the lid, so the tasks are (a) and (b) together.
- **Not solved or not attempted:** pulling apart, tearing, and twist-apart of two free parts held only in the air (no table, no fixture). I found no sim-to-real paper on pulling apart or tearing with dexterous hands. Friction in the joint is modelled ad hoc: a prismatic "brake link" pressing on the lid (Lin 2024), or joint damping (Grasp2Twist). No paper sweeps the joint's friction torque as a controlled variable.
- **Hand roles are asymmetric.** AsymDex reaches 16–18/20 on real tasks with the holding hand's fingers frozen; only its wrist pose is learned. This suggests the holding role needs little finger dexterity. That fits the idea of a palm-side finger group that only grips while a second group does the work.
- **No paper compares one non-anthropomorphic hand against two hands on the same task.** The nearest work is in three groups:
  - single-hand two-part assembly with RL and sim-to-real: *Assembling Two Parts in One Hand* (CoRL 2026), which used a Sharpa hand and ranked Sharpa and Wuji best in simulation;
  - non-anthropomorphic hands with two grip groups and relative motion between them, controlled by scripts and not compared with two hands: the Cartesian Hand (2026), the EPFL detachable hand, Yale Model Q, and a screwdriving gripper;
  - hand co-design for single-hand tasks: House of Dextra (ICLR 2026), and ISyHand, whose articulated palm beat a fixed palm in simulation.
- **The open space for the thesis:** a controlled comparison under one RL recipe of a human-like hand, a fixed-palm hand, a ring hand, and later two hands, swept over joint friction torque on an in-air twist-apart task. Nobody has published this, and the field has no metric for it (for example, the maximum friction torque at which success stays above X%).

---

## 2. Table: sim-to-real bimanual papers

Dexterous multi-finger papers come first. Two-arm parallel-jaw or tactile-pad papers follow, in less detail. "Real result" is copied from the paper.

| Paper | Year / venue | Hardware (arms + hands) | Task | Kind | Sim / RL method | Key sim-to-real ingredients | Real-world result |
|---|---|---|---|---|---|---|---|
| [Twisting Lids Off with Two Hands](https://arxiv.org/abs/2403.02338) (Lin, Yin, Qi, Abbeel, Malik) | 2024, CoRL | 2× UR5e (fixed) + 2× Allegro | Twist and remove bottle and jar lids, object held in air between hands | (a)+(b) | Isaac Gym, PPO, asymmetric critic | "Brake link" friction model for the lid; 2-point object representation (SAM + XMem + depth); keypoint contact reward; DR on mass, friction 0.5–1.5, PD gains, forces, noise, lag; action EMA | Lid removal over 8 household objects: 10–60% each, 33.75% overall. Best bottle: about 946° in 30 s (about 4 turns on average for the best policy) |
| [Grasp2Twist](https://arxiv.org/abs/2609.32064) (Xu, Deng, Wang, Hanna, Mutlu) | Sep 2026, arXiv (under review) | 2× Franka Panda + 2× Tesollo DG-3F-B (3-finger); Azure Kinect | Jar opening with finger gaiting, jar held in hand | (a)+(b) | mjlab/MuJoCo, PPO (RSL-RL), asymmetric critic, 3-stage curriculum | Geometric "enclosure" reward and switch; lid resistance as joint damping 0.10–0.50; DR on mass, friction 0.2–2.5, effort, PD, delay | 53/60 (88%) over 6 jars, against 18/60 for replay |
| [AsymDex](https://arxiv.org/abs/2411.13020) (Yang, Han, Hu, Ravichandar) | 2024–25, arXiv (CoRL 2024 workshop version) | 2× Kinova Gen3 + Allegro + Psyonic Ability | Block in cup, pour, twist lid | (a), (b) for twist | Isaac Gym (2× Shadow in sim), PPO | Holding hand controls wrist pose only; observations and actions relative to the held object; observation noise; friction 0.5–1.5 for twist | Block in cup 16/20, pour 17/20, twist lid 18/20 |
| [Dynamic Handover: Throw and Catch with Bimanual Hands](https://arxiv.org/abs/2309.05655) (Huang, Chen, Wang, Qin, Yang, Atanasov, Wang) | 2023, CoRL | 2× xArm-6 + 2× Allegro, facing each other; RealSense D435 | Throw from one hand, catch with the other | (d) | Isaac Gym, MAPPO (two agents), learned goal estimator, joint fine-tuning | DR (mass, friction, joint limits, stiffness, damping, scale, noise, gravity); PD system ID; trajectory-prediction model | Success: ball 0.60, cylinder 0.53, triangle 0.33 (hit rate 0.80–0.93) |
| [Sim-to-Real RL for Vision-Based Dexterous Manipulation on Humanoids](https://arxiv.org/abs/2502.20396) (Lin, Sachdev, Fan, Malik, Zhu) | 2025, CoRL | Fourier GR1 (two 7-DoF arms) + Fourier hands (Inspire hands for cross-embodiment); head and third-person D435 | Grasp-and-reach, box lift, handover | (c), (d) | Isaac Gym, PPO, asymmetric critic; per-group specialists distilled to a diffusion policy | Real-to-sim autotune of physics (<4 min); contact "stickers" reward; task-aware init from teleop; segmented depth + 3D object position; broad DR | Best policy per task: grasp-and-reach 62.3%, box lift 80%, handover 52.5% |
| [Object-Centric Dexterous Manipulation from Human Motion Data](https://arxiv.org/abs/2411.04005) (Chen, Wang, Yang, Liu) | 2024, CoRL | 2× UR10e + 2× Shadow; 4 Femto Bolt cameras | Articulated objects from ARCTIC (open laptop, notebook, mixer, microwave; lift box, coffee maker) | (a)/(b)/(c), depends on object | Generative wrist planner from mocap + RL finger controller | Teacher–student (DAgger) to drop velocity inputs; FoundationPose per part at 15 Hz; EMA on actions | Completion rate over 20 trials: box 69.8, microwave 100, laptop 76.7, coffee maker 74.8, mixer 82.8, notebook 64.3 |
| [SymDex: Morphologically Symmetric RL for Ambidextrous Bimanual Manipulation](https://arxiv.org/abs/2505.05287) (Li, Jin, Ordoñez-Apraez, Semini, Liu, Chalvatzaki) | 2025, CoRL | 2× xArm UF850 + 2× Allegro V4; ZED2i | Box lift, table clean (sim also: drawer-insert, threading, bowl-stir, handover) | (c), (f) | Equivariant per-hand subtask policies distilled to one ambidextrous policy | Curriculum adds symmetry, pose and physics randomisation in stages; FoundationPose + SAM2; EMA | Box lift 0.77, table clean 0.63 (30 trials). Without curriculum: 0.13 and 0.07 |
| [Scalable Multi-Task Data Generation via RL for Language-Conditioned Bimanual Dexterous Manipulation](https://arxiv.org/abs/2606.22471) (Li, Jin, Liu, Peters, Chalvatzaki) | Jun 2026, arXiv | Same rig as SymDex | Box lift, two-object pick-and-place, drawer insert, multi-task | (c), (f) | Isaac Lab, IPPO teachers, data distilled to a language-conditioned diffusion policy on point clouds | System ID; DR over about 100 objects, colours, camera pose and intrinsics, lighting; auxiliary object-state loss | Out of 20: multi-object 15, insert 16, lift 14, multi-task 8 |
| [DexGarmentLab](https://arxiv.org/abs/2505.11032) (Wang, Wu, et al.) | 2025, NeurIPS | Sim-to-real rig: 2× UR10e + 2× Shadow; Azure Kinect | Hang trousers, put on hat | (e) | Sim data from one demo, imitation (HALO); not RL | Structural correspondence for data generation; affordance points | Sim data only: 8/15 and 9/15. With 15 real demos added: 13/15 each |
| [DexMimicGen](https://arxiv.org/abs/2410.24185) (Jiang, Xie, Lin, Xu, Wan, Mandlekar, Fan, Zhu) | 2025, ICRA | Fourier GR1 + 2× Inspire | Can sorting (the paper's task includes a handover phase) | (d) | Sim demo generation from 4 teleop demos, then a diffusion policy; not RL | Digital twin; each generated demo checked in sim before running on the real robot | 90% with 40 generated demos, 0% with the 4 source demos |
| [Bi-Touch](https://arxiv.org/abs/2307.06423) (Lin, Church, Yang, Li, Lloyd, Zhang, Lepora) | 2023, RA-L | 2× Dobot MG400 (4-DoF) + TacTip tactile end-effectors, no fingers | Bi-push, bi-reorient, bi-gather | (c) | Tactile Gym 2.0, PPO | Real-to-sim tactile image translation (GAN); tuned sim sensor stiffness and damping, which the authors call the main fix | Push error 14–17 mm; reorient error 7.5–13.4°; gather 10/10 without perturbation |
| [Bi-Manual Manipulation and Attachment via Sim-to-Real RL](https://arxiv.org/abs/2203.08277) (Kataoka, Ghasemipour, Freeman, Mordatch) | 2022, arXiv | 2× xArm6, parallel grippers | Pick up two magnetic blocks and join them | (b)/(c) | Sim RL, joint-space control at 4 Hz | Raw observations; simulator modified for training | Pick up 100%, connect 65% |
| [Bi-Manual Block Assembly via Sim-to-Real RL](https://arxiv.org/abs/2303.14870) (Kataoka et al.) | 2023, arXiv | 2× xArm6, parallel grippers | Assemble a U-shape from 3 magnetic blocks | (b)/(c) | Sim RL | Behaviour constraints, perception and actuation noise, delays, interpolation | 50% real (>90% in sim) |
| [Learning to Centralize Dual-Arm Assembly](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.830007/full) (Alles, Aljalbout) | 2022, Frontiers Robotics & AI | 2× Franka Panda | Dual-arm peg-in-hole (2 mm clearance) | (a)/(b) | PyBullet, SAC + HER, sparse reward | Action space choice: Cartesian impedance transferred best; no DR | Zero-shot. The text gives no numbers (only a figure); impedance control was "by far" the best |
| [FlingBot](https://arxiv.org/abs/2105.03655) (Ha, Song) | 2021, CoRL | 2× UR5, parallel grippers | Fling to unfold cloth | (e) | PyFleX/SoftGym, self-supervised | Blender rendering with colour jitter; fine-tuning on 257 real steps | Coverage 81.9–89.2% (pick-and-place baseline 38–43%) |
| [Cloth Funnels](https://arxiv.org/abs/2210.09347) (Canberk, Chi, Ha, Burchfiel, Cousineau, Feng, Song) | 2023, ICRA | Dual UR5 | Canonicalise garments before folding | (e) | Sim-trained multi-primitive policy | Not checked in detail | Long-sleeve shirt: IoU 0.648, coverage 0.806 |
| [Sim-to-Real Learning for Humanoid Box Loco-Manipulation](https://arxiv.org/abs/2310.03191) (Dao, Duan, Fern) | 2023, arXiv | Agility Digit (two arms, no hands) | Pick up and carry boxes | (c) | MuJoCo RL | Dynamics DR, noisy state, box randomisation, action warm-up | Qualitative: boxes of 1–8 kg moved on hardware |

---

## 3. Notes on the important papers

The "single hand?" line is my own judgement, not the authors'.

**Twisting Lids Off with Two Hands (Lin et al., CoRL 2024).** This is the closest prior work to the pen task. Two Allegro hands on fixed UR5e arms hold a bottle in the air. One hand puts fingers around the lid and twists; the other stabilises the body. The authors modelled lid friction with a "brake link": a prismatic joint that presses on the lid. They say it was the only approach that worked in their simulator. The policy sees joint positions plus two 3D points, the centres of the body and the lid from SAM + XMem masks and depth. A keypoint contact reward was needed; without it, policies did poorly. Noise parameters caused the most variance between policies. In the real world all objects were grasped stably. 3 of 5 bottles turned at a useful speed. Full removal averaged 33.75% over 8 household objects, and failures were mostly on lids needing 5 turns.
*Relevance:* this is the direct baseline for the two-hands condition. The brake-link and damping models show how ad hoc friction modelling still is.
*Single hand?* Likely, for a hand with a holding group and a rotating group. The lid-hand motion is continuous rotation about one axis. A ring joint gives that without regrasping.

**Grasp2Twist (Xu et al., arXiv Sep 2026).** Two Franka arms with 3-finger Tesollo DG-3F-B hands open jars held in the air. The policy is PPO in mjlab/MuJoCo. A geometric "enclosure" measure, built from the convex hull of palm and fingertips, guides the grasp and signals the switch to twisting. A 3-stage curriculum widens domain randomisation late in training. The policy learns finger gaiting: it re-grips the lid to keep turning. Result: 53/60 (88%) over 6 jars, against 18/60 for open-loop replay. Lid resistance is joint damping, 0.10–0.50; no torque values are reported.
*Relevance:* this is the most recent and strongest twist result. It uses the MuJoCo/mjlab stack, not Isaac. The finger gaiting is exactly the cost that a continuous-rotation joint would remove.
*Single hand?* Likely, by the same argument as Lin 2024.

**AsymDex (Yang et al., 2024–25).** This paper splits the hands into a "facilitating" hand, whose wrist pose alone is learned, and a "dominant" hand with full finger control. Observations and actions are expressed in the frame of the held object. The real system has one Allegro and one Ability Hand on Kinova arms. Results: 18/20 on twist-lid, 16/20 on block-in-cup, 17/20 on pour.
*Relevance:* this is the clearest evidence that the stabilising role needs little dexterity. It supports building one hand with a passive or low-DoF holding group and a dexterous working group, and it gives a cheap two-hands baseline design. The relative-coordinate trick maps directly to a ring hand: express actions in the cap or barrel frame.
*Single hand?* Yes for twist-lid and block-in-cup, which have short relative motion. Pour needs reach.

**Dynamic Handover (Huang et al., CoRL 2023).** Two Allegro hands on facing xArms throw and catch objects. Thrower and catcher are separate agents trained with MAPPO in Isaac Gym. A learned goal estimator predicts where the object will land. Real success was 0.33–0.60, against hit rates of 0.80–0.93; the gap was failure to close the grasp.
*Relevance:* low for the pen task. It is a reference for multi-agent PPO on two hands.
*Single hand?* No. Transport over distance is the point of the task.

**Sim-to-Real RL for Vision-Based Dexterous Manipulation on Humanoids (Lin et al., CoRL 2025).** This is a recipe paper on a Fourier GR1. It has four parts:
- automatic real-to-sim tuning of physics and URDF constants;
- contact-sticker rewards;
- task-aware initialisation from short teleop sessions;
- divide-and-conquer distillation of specialists into a depth-based diffusion policy.

Results: box lift 80%, handover 52.5%, grasp-and-reach 62.3%. Ablations: depth alone failed (0–2/10); depth plus a 3D position worked (5–10/10).
*Relevance:* the autotune and distillation ideas carry over to any hand. Box lift and handover are not the thesis tasks.
*Single hand?* No for box lift (too large for one hand by design). Handover is moot.

**Object-Centric Dexterous Manipulation from Human Motion (Chen et al., CoRL 2024).** A generative model trained on human mocap plans wrist motion. An RL controller in simulation tracks a goal trajectory for an articulated object with two Shadow hands on UR10e arms. A DAgger student removes velocity inputs, and four cameras with FoundationPose track each part. Real completion rates were 64–100% across 6 objects, against mostly under 20% for retargeting baselines.
*Relevance:* tasks like opening a laptop or a mixer move two parts of one object relative to each other. That is the twist-apart structure on a revolute joint, but the motion follows human data, not task reward.
*Single hand?* Partly. For small articulated objects (the mixer) the answer is likely yes. Laptops and boxes are too large.

**SymDex (Li et al., CoRL 2025) and its follow-up RLDC (Li et al., arXiv Jun 2026).** Both use one rig: 2× xArm UF850 with 2× Allegro V4. SymDex uses the left–right mirror symmetry of the robot. Each hand gets its own subtask policy with equivariant networks, and these are distilled into one ambidextrous policy. Real results: box lift 0.77, table clean 0.63. Without the curriculum: 0.13 and 0.07. RLDC uses RL teachers to generate data for a language-conditioned point-cloud diffusion policy and reports 14–16/20 on single tasks.
*Relevance:* the curriculum effect (0.77 against 0.13) is a warning that staged randomisation matters for transfer. The symmetry trick does not apply to an asymmetric ring hand.
*Single hand?* No for box lift. Table clean is two parallel jobs, kind (f), so it is not a single-hand question.

**Bi-Touch (Lin et al., RA-L 2023).** Two 4-DoF desktop arms with TacTip pads push, reorient and gather objects. PPO runs on simulated tactile images, and a GAN maps real tactile images to simulated ones. The authors say tuning the simulated sensor's stiffness and damping was the main transfer fix.
*Relevance:* low. It shows that tactile sim-to-real works for two-arm tasks.

**DexGarmentLab (NeurIPS 2025) and DexMimicGen (ICRA 2025).** Both train policies on simulated data by imitation, not RL, and deploy them on two-hand robots. DexGarmentLab: hanging trousers and putting on a hat reached 8–9/15 from sim data alone and 13/15 with 15 real demos added. DexMimicGen: humanoid can sorting reached 90%.
*Relevance:* low. They are listed for completeness.

**Two-arm parallel-jaw papers (brief).** Kataoka et al. (2022, 2023) join magnetic blocks with two xArms, 65% and 50% real. Alles & Aljalbout (2022) found that the action space (Cartesian impedance) mattered more than domain randomisation for dual-arm peg insertion. FlingBot (2021) and Cloth Funnels (2023) use two arms for cloth, which is the main two-arm sim-to-real success outside rigid objects. Dao et al. (2023) carry boxes with a Digit humanoid's two arms.
*Single hand?* Mostly no for cloth and boxes. Possibly yes for the assembly tasks.

---

## 4. Context

### 4a. Simulation-only bimanual benchmarks and methods (no real-robot results found)

- **Bi-DexHands** — Chen et al., *Towards Human-Level Bimanual Dexterous Manipulation with RL*, NeurIPS 2022 Datasets & Benchmarks. Two Shadow hands in Isaac Gym with tens of tasks, including bottle opening, pot lifting, catch and handover. PPO handles the simple tasks; multi-task and few-shot settings fail. [link](https://proceedings.neurips.cc/paper_files/paper/2022/hash/217a2a387f52c30755c37b0a73430291-Abstract.html)
- **DexPBT** — Petrenko et al., RSS 2023. Population-based training for one- and two-arm Allegro/Kuka setups, including two-arm regrasping and reorientation. Simulation only. [arXiv:2305.12127](https://arxiv.org/abs/2305.12127)
- **ArtiGrasp** — Zhang, Christen, et al., 3DV 2024. RL synthesis of two-hand grasping and articulation (RaiSim), with a curriculum from one hand to two. [arXiv:2309.03891](https://arxiv.org/abs/2309.03891)
- **BiDexHD** — Zhou et al., arXiv 2024. Builds tasks from the TACO two-hand dataset, with teacher–student learning. 74.59% task fulfilment on trained tasks and 51.07% on unseen tasks (simulation). [arXiv:2410.02477](https://arxiv.org/abs/2410.02477)
- **ManipTrans / DexManipNet** — Li et al., CVPR 2025. Residual transfer of human two-hand motion to robot hands. The dataset includes pen capping and bottle unscrewing in simulation. The real demo is trajectory replay on 2× Realman + Inspire (toothpaste cap) with no success rates, so it is not a closed-loop sim-to-real result. [arXiv:2503.21860](https://arxiv.org/abs/2503.21860)
- **DexMachina** — Mandi et al., ICML 2026 (per the NVIDIA listing). Functional retargeting with decaying virtual object controllers, for long-horizon two-hand tasks with articulated objects. [arXiv:2505.24853](https://arxiv.org/abs/2505.24853)
- **Dexterous bimanual catch** — Kim, Yoon, Kim, arXiv 2025. Adversarial–cooperative heterogeneous-agent RL; 2× UR3 + Allegro in Isaac Gym. The authors state that all experiments are in simulation. [arXiv:2502.11437](https://arxiv.org/abs/2502.11437)
- **DyDexHandover** — Zhou, You, Wang, arXiv 2025. RGB-only multi-agent RL throw-and-catch, evaluated in Isaac Sim only. [arXiv:2509.17350](https://arxiv.org/abs/2509.17350)
- **RoboTwin** — Mu et al., CVPR 2025. Dual-arm benchmark built with generative digital twins. Pre-training on its synthetic data, then fine-tuning on real data, helped real success on COBOT Magic. [CVPR page](https://openaccess.thecvf.com/content/CVPR2025/html/Mu_RoboTwin_Dual-Arm_Robot_Benchmark_with_Generative_Digital_Twins_CVPR_2025_paper.html)
- **BiGym** — Chernyadev et al., CoRL 2024. 40 mobile two-arm tasks with human demonstrations. [arXiv:2407.07788](https://arxiv.org/abs/2407.07788)
- **PerAct2 / RLBench2** — Grotz, Shridhar, Asfour, Fox, arXiv 2024. 13 two-arm tasks added to RLBench. [arXiv:2407.00278](https://arxiv.org/abs/2407.00278)

### 4b. Real-only bimanual systems (teleoperation or imitation, no sim training)

- **ALOHA / ACT** — Zhao, Kumar, Levine, Finn, RSS 2023. Low-cost two-arm teleoperation and action chunking; 80–90% on fine tasks such as opening a condiment cup. [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)
- **Mobile ALOHA** — Fu, Zhao, Finn. [arXiv:2401.02117](https://arxiv.org/abs/2401.02117). **ALOHA Unleashed** — Zhao et al. [arXiv:2410.13126](https://arxiv.org/abs/2410.13126). Both are parallel-jaw systems that scale up teleop data.
- **HATO** — Lin, Zhang, Li, Qi, Yi, Levine, Malik, ICRA 2025. Two UR5 arms with tactile Psyonic Ability hands, VR teleop and diffusion policies. Tasks include slippery handover and wine pouring. [arXiv:2404.16823](https://arxiv.org/abs/2404.16823)
- **BiDex** — Shaw et al., CoRL 2024. Low-cost glove plus GELLO-style teleop for two LEAP-class hands on xArms. [arXiv:2411.13677](https://arxiv.org/abs/2411.13677)
- **DexCap** — Wang, Shi, Wang, Zhang, Fei-Fei, Liu, arXiv 2024. Mocap gloves feeding 2× Franka + LEAP. Two-hand tasks include tea preparation, where one hand holds the bottle and the other uncaps it, and scissor cutting, where one hand holds the paper. [arXiv:2403.07788](https://arxiv.org/abs/2403.07788)
- **ScrewMimic** — Bahety, Mandikal, Abbatematteo, Martín-Martín, RSS 2024. Models the two hands as a screw joint and learns from one human video plus self-improvement on a Tiago++. Tasks include opening a bottle (16/25 episodes with the full method). The screw-axis action space is a useful way to frame the twist-apart task. [arXiv:2405.03666](https://arxiv.org/abs/2405.03666)
- **Bunny-VisionPro** — Ding et al., arXiv 2024 (two-hand dexterous teleop) [arXiv:2407.03162](https://arxiv.org/abs/2407.03162). **Open-TeleVision** — Cheng et al., CoRL 2024 [arXiv:2407.01512](https://arxiv.org/abs/2407.01512).

---

## 5. Gaps and opportunities for the thesis

**1. Nobody has compared one non-anthropomorphic hand with two hands on the same task.** I searched for single hand against two hands, unimanual against bimanual, and hand design for bimanual tasks. I found no study that runs the same task with one designed hand and with two hands under one control or learning method. The nearest work is below.

- *Assembling Two Parts in One Hand* (Pei, Wu, Zhang, Luo, Song; CoRL 2026, [arXiv:2609.10137](https://arxiv.org/abs/2609.10137)).
  - What it does: mates two rigid parts inside one hand with sim-to-real RL. The parts are a bottle and lid, a syringe, and a marker and cap. Hardware is a 22-DoF Sharpa Wave hand on a Franka with one camera.
  - Results: 15/20, 17/20 and 16/20 assemblies.
  - Hand comparison: in simulation the authors compared hands. Sharpa and **Wuji** did best; Allegro could not insert deeply; XHand often failed.
  - Limits: the assembly is push-in, with no twisting (threading is listed as future work), and there is no two-hand baseline.
  - Relevance: this is the closest existing paper to the proposal. It argues that in-hand assembly can benchmark hand morphology. The pen work extends it in three ways: a twist joint, non-anthropomorphic hands, and a two-hand baseline.
- *The Cartesian Hand* (Xia et al., arXiv Sep 2026, [arXiv:2609.25696](https://arxiv.org/abs/2609.25696)).
  - What it does: a 7-DoF end-effector with two independent parallel grippers holding different parts of an object, plus four translating fingertips that create relative motion between them. This is close to the "two finger groups with a joint between" idea.
  - Results: cap opening and closing, screwdriving and pipetting on 35 objects, 350/350 trials.
  - Limits: control is scripted primitives with no vision and no learning, and there is no comparison with two hands. The authors argue that "one arm does not need to serve primarily as a fixture for the other."
- *A detachable crawling robotic hand* (Gao, Yao, Junge, Hughes, Billard; Nature Communications 2026, [doi:10.1038/s41467-025-67675-8](https://www.nature.com/articles/s41467-025-67675-8)). Up to six identical, reversible fingers. EPFL's press release says it can unscrew a large bottle cap with one hand, a task that normally needs two. I read the press coverage, not the paper, so details are unverified.
- *Yale OpenHand Model Q* ([project page](https://www.eng.yale.edu/grablab/openhand/model_q.html); used in [arXiv:2201.07928](https://arxiv.org/abs/2201.07928)). Two precision fingers plus a pair of power-grasp fingers on a palm rotary joint (about 110°). It is an early hardware example of independent finger groups with a rotation between them. It was used for finger gaiting, not bimanual-style tasks.
- *Screwdriving Gripper That Mimics Human Two-Handed Assembly Tasks* (Han et al., Robotics (MDPI) 2022, [doi:10.3390/robotics11010018](https://www.mdpi.com/2218-6581/11/1/18)). One gripper with two grasping fingers and one screwdriving finger, built to replace a two-handed assembly step. Industrial, no learning.

**2. Hand design and co-design work exists, but only for single-hand tasks.**

- *House of Dextra: Cross-embodied Co-design for Dexterous Hands* (Fay et al., ICLR 2026, [arXiv:2512.03743](https://arxiv.org/abs/2512.03743)). Morphology-conditioned policy plus graph search over hand designs, with fabrication and real deployment in under 24 hours. Tasks include in-hand rotation. No bimanual-style tasks.
- *ISyHand* (Richardson et al., Humanoids 2025, [arXiv:2509.26236](https://arxiv.org/abs/2509.26236)). An articulated palm clearly outperformed a fixed-palm version of the same hand on simulated cube reorientation, and the policy was deployed on the real hand. This is the closest published evidence that a joint inside the palm helps RL. It is directly comparable to the fixed-palm against ring-hand contrast.
- *MORPH* (He & Ciocarlie, ICRA 2024, [arXiv:2309.17227](https://arxiv.org/abs/2309.17227)): co-optimisation through a differentiable hardware proxy. *Function-based Parametric Co-Design of Dexterous Hands* (Mirzaee, Gupta, Yuan, arXiv 2026, [arXiv:2604.27557](https://arxiv.org/abs/2604.27557)): grasp stability. *Generating Robot Hands from Human Demonstrations* (Yi et al., arXiv 2026, [arXiv:2606.20549](https://arxiv.org/abs/2606.20549)): fingertip tracking. None of these targets opposing-motion tasks.

**3. The task space has a hole.**

- Sim-to-real dexterous work covers twisting where one hand holds a container (Lin 2024, AsymDex, Grasp2Twist), lifting, and handover.
- I found no sim-to-real dexterous paper on pulling apart, tearing, or twisting apart two small free parts in the air. The pen task fills this hole.
- Simulated pen capping exists only as human-motion transfer (ManipTrans), not as task-reward RL with transfer.

**4. Joint friction is not treated as an experimental variable.**

- Lin 2024 uses a brake link. Grasp2Twist uses damping of 0.10–0.50. Both randomise friction coefficients for robustness.
- No paper sweeps the torque needed to break the joint, or reports success as a function of it.
- A friction-torque sweep would give a physical axis on which hand designs separate. The expectation is that a human-like hand fails at lower torque, because its "holding" fingers and "twisting" fingers share one palm and one wrist.
- The metric is new; something like "maximum sustained relative torque at success at or above X%".

**5. Two-hands baseline: design choices to borrow.**

- AsymDex: holding hand with wrist-only control, and relative coordinates.
- Lin 2024: two-point object representation, keypoint contact reward, action EMA.
- Lin 2025: real-to-sim autotune.
- Grasp2Twist: enclosure reward and staged domain randomisation.

Reusing these makes the two-hands condition a fair, strong baseline rather than a straw man. Note the hardware: all the strong two-hand twist results use fixed arms and grasp in the air. Matching that (two wrists on fixed mounts in simulation) keeps the comparison clean.

**6. Measurement.** Published two-hand twisting results report angle turned, time to failure, or success per object over 10–20 trials. Most report a single training run. Given the large seed-to-seed variance in this kind of RL, reporting several seeds per hand would already be above the field's norm.

---

## 6. Unverified or excluded

- **VTAO-BiManip** (Sun et al., arXiv 2025, [arXiv:2501.03606](https://arxiv.org/abs/2501.03606)): RL for two-hand bottle-cap twisting in Isaac Gym (Shadow). A real platform with LEAP hands and a Unitree Z1 is shown, but I found no real success numbers or transfer method. Not in the table.
- **Contact-Anchored Retargeting and Residual Policy Learning** (Yang et al., arXiv Sep 2026, [arXiv:2609.24093](https://arxiv.org/abs/2609.24093)): the abstract claims four contact-rich two-hand tasks on hardware with no real training data. Hardware and numbers not checked.
- **DemoBot** (ByteDance Seed, arXiv 2025, [arXiv:2601.01651](https://arxiv.org/abs/2601.01651)): two-hand tasks in simulation, but real deployment was single-arm (UR3e + XHand). Excluded.
- **HDMI** and **OmniRetarget** (humanoid whole-body loco-manipulation on Unitree G1, 2025): I could not confirm a two-arm object task from the abstracts. Not included.
- **Curriculum Design and Sim2Real Transfer for RL in Robotic Dual-Arm Assembly** (Machines (MDPI) 2024): the page returned 403, so authors and numbers are unchecked.
- **Do Robots Really Need Anthropomorphic Hands? A Comparison of Human and Robotic Hands** (J. Intelligent & Robotic Systems, 2026): seen in search only; content not read.
- Venues not confirmed: Mobile ALOHA, ALOHA Unleashed, DexCap and Bunny-VisionPro are cited as arXiv only. AsymDex has no main-track venue that I could find.

---

## 7. References

1. Lin, Yin, Qi, Abbeel, Malik. Twisting Lids Off with Two Hands. CoRL 2024. https://arxiv.org/abs/2403.02338 · project: https://toruowo.github.io/bimanual-twist/
2. Xu, Deng, Wang, Hanna, Mutlu. Grasp2Twist: Learning Bimanual Dexterous Jar Opening by Reinforcement Learning. arXiv 2026. https://arxiv.org/abs/2609.32064
3. Yang, Han, Hu, Ravichandar. AsymDex: Asymmetry and Relative Coordinates for RL-based Bimanual Dexterity. arXiv 2024/25. https://arxiv.org/abs/2411.13020
4. Huang, Chen, Wang, Qin, Yang, Atanasov, Wang. Dynamic Handover: Throw and Catch with Bimanual Hands. CoRL 2023. https://arxiv.org/abs/2309.05655
5. Lin, Sachdev, Fan, Malik, Zhu. Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids. CoRL 2025. https://arxiv.org/abs/2502.20396
6. Chen, Wang, Yang, Liu. Object-Centric Dexterous Manipulation from Human Motion Data. CoRL 2024. https://arxiv.org/abs/2411.04005
7. Li, Jin, Ordoñez-Apraez, Semini, Liu, Chalvatzaki. Morphologically Symmetric Reinforcement Learning for Ambidextrous Bimanual Manipulation. CoRL 2025. https://arxiv.org/abs/2505.05287
8. Li, Jin, Liu, Peters, Chalvatzaki. Scalable Multi-Task Data Generation via Reinforcement Learning for Language-Conditioned Bimanual Dexterous Manipulation. arXiv 2026. https://arxiv.org/abs/2606.22471
9. Wang, Wu, et al. DexGarmentLab: Dexterous Garment Manipulation Environment with Generalizable Policy. NeurIPS 2025. https://arxiv.org/abs/2505.11032
10. Jiang, Xie, Lin, Xu, Wan, Mandlekar, Fan, Zhu. DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning. ICRA 2025. https://arxiv.org/abs/2410.24185
11. Lin, Church, Yang, Li, Lloyd, Zhang, Lepora. Bi-Touch: Bimanual Tactile Manipulation with Sim-to-Real Deep Reinforcement Learning. RA-L 8(9), 2023. https://arxiv.org/abs/2307.06423
12. Kataoka, Ghasemipour, Freeman, Mordatch. Bi-Manual Manipulation and Attachment via Sim-to-Real Reinforcement Learning. arXiv 2022. https://arxiv.org/abs/2203.08277
13. Kataoka, Chung, Ghasemipour, Sanketi, Gu, Mordatch. Bi-Manual Block Assembly via Sim-to-Real Reinforcement Learning. arXiv 2023. https://arxiv.org/abs/2303.14870
14. Alles, Aljalbout. Learning to Centralize Dual-Arm Assembly. Frontiers in Robotics and AI 9, 2022. https://doi.org/10.3389/frobt.2022.830007
15. Ha, Song. FlingBot: The Unreasonable Effectiveness of Dynamic Manipulation for Cloth Unfolding. CoRL 2021. https://arxiv.org/abs/2105.03655
16. Canberk, Chi, Ha, Burchfiel, Cousineau, Feng, Song. Cloth Funnels: Canonicalized-Alignment for Multi-Purpose Garment Manipulation. ICRA 2023. https://arxiv.org/abs/2210.09347
17. Dao, Duan, Fern. Sim-to-Real Learning for Humanoid Box Loco-Manipulation. arXiv 2023. https://arxiv.org/abs/2310.03191
18. Pei, Wu, Zhang, Luo, Song. Assembling Two Parts in One Hand. CoRL 2026 (to appear). https://arxiv.org/abs/2609.10137
19. Xia, Li, Shin, Yang, Liu, Chen. The Cartesian Hand: In-Hand Manipulation with All-Linear Fingers. arXiv 2026. https://arxiv.org/abs/2609.25696
20. Gao, Yao, Junge, Hughes, Billard. A detachable crawling robotic hand. Nature Communications 17, 2026. https://www.nature.com/articles/s41467-025-67675-8
21. Yale GRAB Lab. OpenHand Model Q. https://www.eng.yale.edu/grablab/openhand/model_q.html ; Complex In-Hand Manipulation via Compliance-Enabled Finger Gaiting and Multi-Modal Planning. https://arxiv.org/abs/2201.07928
22. Han et al. Screwdriving Gripper That Mimics Human Two-Handed Assembly Tasks. Robotics 11(1):18, 2022. https://doi.org/10.3390/robotics11010018
23. Fay et al. House of Dextra: Cross-embodied Co-design for Dexterous Hands. ICLR 2026. https://arxiv.org/abs/2512.03743
24. Richardson, Grüninger, Mack, Stueckler, Kuchenbecker. ISyHand: A Dexterous Multi-finger Robot Hand with an Articulated Palm. Humanoids 2025. https://arxiv.org/abs/2509.26236
25. He, Ciocarlie. MORPH: Design Co-optimization with Reinforcement Learning via a Differentiable Hardware Model Proxy. ICRA 2024. https://arxiv.org/abs/2309.17227
26. Mirzaee, Gupta, Yuan. Function-based Parametric Co-Design Optimization of Dexterous Hands. arXiv 2026. https://arxiv.org/abs/2604.27557
27. Yi, Hansen, Bai, Sferrazza, Tolley, Wang. Generating Robot Hands from Human Demonstrations. arXiv 2026. https://arxiv.org/abs/2606.20549
28. Chen et al. Towards Human-Level Bimanual Dexterous Manipulation with Reinforcement Learning (Bi-DexHands). NeurIPS 2022 D&B. https://proceedings.neurips.cc/paper_files/paper/2022/hash/217a2a387f52c30755c37b0a73430291-Abstract.html
29. Petrenko, Allshire, State, Handa, Makoviychuk. DexPBT. RSS 2023. https://arxiv.org/abs/2305.12127
30. Zhang, Christen, et al. ArtiGrasp: Physically Plausible Synthesis of Bi-Manual Dexterous Grasping and Articulation. 3DV 2024. https://arxiv.org/abs/2309.03891
31. Zhou, Yuan, Fu, Lu. Learning Diverse Bimanual Dexterous Manipulation Skills from Human Demonstrations (BiDexHD). arXiv 2024. https://arxiv.org/abs/2410.02477
32. Li, Li, Liu, Li, Huang. ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning. CVPR 2025. https://arxiv.org/abs/2503.21860
33. Mandi, Hou, Fox, Narang, Mandlekar, Song. DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation. ICML 2026. https://arxiv.org/abs/2505.24853
34. Kim, Yoon, Kim. Learning Dexterous Bimanual Catch Skills through Adversarial-Cooperative Heterogeneous-Agent Reinforcement Learning. arXiv 2025. https://arxiv.org/abs/2502.11437
35. Zhou, You, Wang. DyDexHandover: Human-like Bimanual Dynamic Dexterous Handover using RGB-only Perception. arXiv 2025. https://arxiv.org/abs/2509.17350
36. Mu et al. RoboTwin: Dual-Arm Robot Benchmark with Generative Digital Twins. CVPR 2025.
37. Chernyadev et al. BiGym: A Demo-Driven Mobile Bi-Manual Manipulation Benchmark. CoRL 2024. https://arxiv.org/abs/2407.07788
38. Grotz, Shridhar, Asfour, Fox. PerAct2: Benchmarking and Learning for Robotic Bimanual Manipulation Tasks. arXiv 2024. https://arxiv.org/abs/2407.00278
39. Zhao, Kumar, Levine, Finn. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ALOHA). RSS 2023. https://arxiv.org/abs/2304.13705
40. Fu, Zhao, Finn. Mobile ALOHA. arXiv 2024. https://arxiv.org/abs/2401.02117
41. Zhao et al. ALOHA Unleashed: A Simple Recipe for Robot Dexterity. arXiv 2024. https://arxiv.org/abs/2410.13126
42. Lin, Zhang, Li, Qi, Yi, Levine, Malik. Learning Visuotactile Skills with Two Multifingered Hands (HATO). ICRA 2025. https://arxiv.org/abs/2404.16823
43. Shaw et al. Bimanual Dexterity for Complex Tasks (BiDex). CoRL 2024. https://arxiv.org/abs/2411.13677
44. Wang, Shi, Wang, Zhang, Fei-Fei, Liu. DexCap. arXiv 2024. https://arxiv.org/abs/2403.07788
45. Bahety, Mandikal, Abbatematteo, Martín-Martín. ScrewMimic: Bimanual Imitation from Human Videos with Screw Space Projection. RSS 2024. https://arxiv.org/abs/2405.03666
46. Ding et al. Bunny-VisionPro: Real-Time Bimanual Dexterous Teleoperation for Imitation Learning. arXiv 2024. https://arxiv.org/abs/2407.03162
47. Cheng et al. Open-TeleVision: Teleoperation with Immersive Active Visual Feedback. CoRL 2024. https://arxiv.org/abs/2407.01512
48. Sun et al. VTAO-BiManip. arXiv 2025. https://arxiv.org/abs/2501.03606 (see Unverified)
