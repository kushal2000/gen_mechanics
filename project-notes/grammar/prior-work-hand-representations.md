# Prior work on universal robot-hand representations (survey, 2026-09-25)

Produced by a Sonnet 5 investigator agent in a 40-minute web survey, relayed unedited below the summary. Confidence labels are the agent's own: "[verified: fetched]" means it opened the primary page; "[search snippet]" means it relied on a search-engine summary and the claim is a lead, not a fact. Nothing here has been re-verified by the coordinator yet.

## Coordinator summary

1. Nobody has built a representation that covers existing hardware hands AND serves as a generative design space AND feeds a controller. The description collections (dex-urdf, mujoco_menagerie, DexGraspNet) and the generative co-design papers are disjoint artifacts.
2. Palms in the closest generative works are single rigid extruded outlines (parametric polygon in arXiv 2604.27557; convex hull of grip points in arXiv 2512.03743, "House of Dextra"), with fingers placed on the outline; neither models staggered mount heights or an articulated arch. Biomechanical models (Cobos 2008, 25-DoF models) put the palmar arch into an extra CMC joint DoF; the MANO-to-URDF paper (arXiv 2512.07359) keeps one rigid palm and encodes the transverse arch through per-finger MCP axis choices.
3. A palm that is itself a small branching sub-tree of bodies (staggered mounts, actuatable arch), plus couplings and closed chains as first-class constructs, was not found anywhere. Those are the gaps our representation already targets.
4. Controller side: per-joint tokens with origin, axis, limits and link geometry, connected by the real graph topology, are the standard interface (GET-Zero, UniMorphGrasp, GraspGraphNet); GET-Zero's auxiliary FK self-modeling loss gave about 20% zero-shot gain. CrossDex-style work suggests the policy can consume far less (fingertip frames plus a palm frame) than the design side must store.
5. Geometry: where solid palms occur, prior work uses meshes with simulator-side convex decomposition (CoACD), not hand-authored capsule hulls. Capsules per finger link are standard. The palm remains the open choice.

## Agent report (verbatim)

### 1. Universal / cross-embodiment hand representations for learning

| Work | Year | Representation | Coverage | Geometry | Code | Notes/limits |
|---|---|---|---|---|---|---|
| UniDexGrasp / UniDexGrasp++ | 2023 | Maps grasps from diverse hands into a unified human-like canonical hand-pose space; grasp generation conditioned on graph-encoded hand kinematics [search snippet] | Multiple hands via canonicalization; palm/coupling detail unclear | Not established | — | Could not confirm palm/coupling handling |
| DexGraspNet (pku-epic.github.io/DexGraspNet) | 2022/23 | Per-hand URDF + FK to compute hand mesh [search snippet] | URDF joint tree; no unifying schema beyond "give it a URDF" | Hand mesh from URDF meshes | pku-epic/DexGraspNet | Not a new unifying format |
| D(R,O) Grasp (arXiv 2410.01702) | 2024/25 | Interaction-based: point-to-point distance matrix between robot and object [search snippet] | 3 hands | Point-based, no explicit palm model | — | Sidesteps an explicit palm/finger-placement schema |
| T(R,O) Grasp (arXiv 2510.12724) | 2025 | Graph diffusion extension of D(R,O) [search snippet] | — | — | — | Not opened |
| CEDex (arXiv 2509.24661) | 2025 | Cross-embodiment grasp generation from human-like contact representations [search snippet] | — | — | — | Not opened |
| MachaGrasp (arXiv 2510.06068) | 2025 | Morphology embedding + hand-specific eigengrasp set from a morphology description; kinematic-aware articulation loss [verified: fetched] | 3 hands in sim, few-shot to 1 unseen | Object point cloud + wrist pose | connor-zh.github.io/MachaGrasp | Palm articulation not confirmed |
| UniMorphGrasp (arXiv 2602.00915) | 2026 | Graphormer morphology encoder: joint geometry, limits, origin, axis as tokens [search snippet] | 94% in-domain, 91.3% zero-shot novel morphologies [search snippet] | — | — | Direct analogue of kinematic-graph tokens |
| GraspGraphNet (arXiv 2607.11031) | 2026 | URDF-derived kinematic graph (links = nodes, joints = edges) + differentiable FK + message passing to object surface [verified: fetched, partial] | Barrett, Allegro, Shadow; 72.7% on finger-removal variants without retraining | Hand geometry not itemized | lysees.github.io/graspgraphnet-page | Closest to our graph-of-links-and-joints, for grasp generation |
| "CrossDex" (arXiv 2410.02479; name unconfirmed) | 2024 | Eigengrasp universal action space (PCA over human poses); observation = fingertip + palm positions only [search snippet] | 4 hands, 80%; zero-shot 35–39% on 2 unseen | Keypoint-based | — | Fingertip+palm observations sufficed |
| UniGrasp (arXiv 1910.10900) | 2019/20 | Gripper as point cloud, PointNet embedding, point-set contact selection [search snippet] | N-fingered grippers | Point cloud only | stanford-iprl-lab/UniGrasp | No skeletal semantics |
| GraspXL | 2024 | Shared training recipe, separate model per gripper [search snippet] | — | — | — | Not cross-embodiment transfer |
| NeuralGrasps (arXiv 2207.02959) | 2022 | Implicit multi-hand grasp representation [search snippet] | — | — | — | Not opened |
| Learning Cross-hand Policies (arXiv 2404.09150) | 2024 | Gripper-agnostic policy on predefined keypoints + per-gripper adapter [verified: abstract] | Several grippers | Finger-level keypoints | — | "Agnostic policy + per-hand adapter" pattern |
| AnyDexGrasp | ~2024/25 | Contact maps, eigengrasps [search snippet] | — | — | — | Not opened |
| DexRep | ~2023 | Hand-object interaction descriptor, not a morphology representation | — | — | — | Different concern |
| GET-Zero (get-zero-paper.github.io) | 2024 | Embodiment graph: joints = nodes; per-joint tokens with hardware properties; graph connectivity as attention bias; auxiliary per-joint FK self-modeling loss [verified: fetched] | 4-finger hand variants (joints removed, links extended); ~20% gain on unseen graphs | Link length scalar; no palm geometry | real-stanford/get_zero | Best evidence that graph + FK self-modeling helps transfer |
| DexFormer (arXiv 2602.08278) | 2026 | History-conditioned transformer, morphology inferred implicitly [search snippet] | Zero-shot to unseen canonical hands | — | — | Counterpoint: explicit descriptor not strictly required |
| UniDexTok (arXiv 2606.10683) | 2026 | Shared 22-DoF semantic state interface, tokenizer [search snippet] | Sub-mm MPJPE | State only | — | Retargeting-style, not structural |
| RoboToken | ~2025/26 | Tokenizes 11 robots incl. one hand [search snippet] | Broad | — | — | Citation not found |
| MetaMorph (arXiv 2203.11931), AnyMorph (arXiv 2206.12279), NerveNet (2018), 3D-SGRL (2023) | 2018–2023 | Token-sequence or GNN morphology-agnostic control for modular robots | Not hands | — | — | Precedent architectures |
| Isaac Lab / Bi-DexHands / DexterousHands, DexVerse (arXiv 2607.08751), DexPBT, Visual Dexterity, RoboHive | 2019–2026 | Multi-hand environments/benchmarks; per-hand obs/action spaces | Not a shared representation | Per-hand URDF/MJCF | various | DexVerse: best policies ~34% success across 6 hands |

Not found: "GeoDex"; a paper titled "CrossDex"; GLSO applied to hands.

### 2. Hand description collections and formats

| Collection | Format | Palm/fingertip/coupling encoding | Notes |
|---|---|---|---|
| dex-urdf (github.com/dexsuite/dex-urdf) | URDF | Vendor URDF only | Curated for SAPIEN/IsaacGym/PyBullet/yourdfpy |
| dex-retargeting (dexsuite) | uses dex-urdf | Fingertip frames as retargeting targets | — |
| mujoco_menagerie | MJCF | No hand-specific schema | Simulation-tuned |
| DexGraspNet hand set | URDF | Per-hand URDF + FK | Includes MANO as a hand |
| GraspIt! eigengrasp files | GraspIt config + eigengrasp text (DIMENSIONS/ORIGIN/NORM) [verified: fetched] | Predefined eigengrasps for Robonaut, DLR, Barrett | Oldest standardized low-dim action-space format found |
| OpenRAVE / OpenGRASP | OpenRAVE XML | Generic robot XML | No hand ontology |

Not found: any schema beyond URDF/MJCF/SDF/USD specifically for hands.

### 3. Design taxonomies and parametric kinematic models

| Work | Year | What it is | Palm handling |
|---|---|---|---|
| Piazza, Grioli, Catalano, Bicchi, "A Century of Robotic Hands" (Annu. Rev. Control Robot. Auton. Syst.) | 2019 | Review of actuation/soft-structure design trends | Not palm-specific |
| Feix et al. GRASP taxonomy | 2016 | Grasp types, not hand designs | N/A |
| Cobos et al. (oa.upm.es/4040) | 2008 | 24-DoF DH human-hand model; CMC joint simulates the palm arc | Arch = extra CMC DoF, not palm geometry |
| DLR functional-anatomy hand model (elib.dlr.de/76346) | 2012 | Kinematic model with size adaptation | — |
| 25-DoF hand model | ~2015 | 4 extra DoF at CMC/wrist for ring and small fingers to arch the palm | Arch as explicit DoF for ulnar fingers only |
| MANO (Romero et al.) | 2017 | 16-joint tree, shape+pose blendshapes; palm is a deformable skinned region | Stagger emerges from the mesh, not from kinematic parameters |
| Multi-Rigid-Body Approximation of Human Hands (arXiv 2512.07359) [verified: fetched] | 2025/26 | MANO to URDF: one rigid palm + 15 phalanges; CMC/MCP 2-DoF, PIP/DIP 1-DoF | Per-finger MCP axis chosen from different geometric references (middle uses ring-to-index MCP vector, "capturing the transverse arch"); stagger baked into fixed joint origins; geometry = MANO mesh per segment |

### 4. Procedural generation / evolution / co-design

| Work | Year | Representation | Palm / finger placement | Geometry | Loops/couplings |
|---|---|---|---|---|---|
| RoboGrammar (Zhao et al., SIGGRAPH Asia 2020) | 2020 | Graph grammar over robot assemblies, MCTS+GNN search | Not hands | — | — |
| Text2Robot (arXiv 2406.19963), RoboMorph (arXiv 2407.08626) | 2024 | LLM/evolutionary quadruped and modular-robot design | Not hands | — | — |
| Lang2Morph (arXiv 2509.18937) [verified: abstract] | 2025 | Task text to semantic tags to structural grammar to OPH (Open Parametric Hand) parameters | Unclear from abstract | 3D-printable | Unclear |
| Open Parametric Hand (Science Robotics ads6437) | 2024/25 | Single-piece 3D-printed parametric anatomical hand | Page 403; not verified | — | — |
| Function-based Parametric Co-Design of Dexterous Hands (arXiv 2604.27557) [verified: fetched] | 2026 | Unified 28-parameter design space | Palm = extruded parametric 2D polygon; each finger base has normal offset, side offset, angle, orientation; LEAP-inspired thumb base joint | Mesh links, convex-decomposed pads, STL output | Open serial chains only |
| Cross-Embodied Co-Design, "House of Dextra" (arXiv 2512.03743) [verified: fetched] | 2025/26 | Joint, finger, palm generation + morphology-conditioned cross-embodied RL (GNN embedding of G=(V,E,Xv,Xe)) | Palm = 2D convex hull of grip points around finger bases, extruded; fingers on a circle with min-separation rejection sampling; f in {3,4,5}, joints per finger in {0,2,3} | Modular 3D-printed parts, CoACD decomposition | None |
| Evolving Robotic Hand Morphology Through Grasping and Learning (IEEE 10631681) | 2024 | Reconfigurable prototype, 37 hardware parameters incl. finger location and palm curvature [search snippet] | Palm curvature as an evolvable scalar | — | — |
| Co-optimization of design and skill (PubMed 35944514); Co-designing hardware and control (Sci. Robot. abg2133) | 2021/22 | Underactuated hand co-optimization | Not detailed | — | — |
| Sims 1994 | 1994 | Directed-graph genetic language | Not hands | — | Recursive graphs |

Not found: any graph grammar or L-system applied to hands; "Neural Fingers"; "computational design of robotic hands (Disney/ETH/MIT)" under those names.

### 5. Palm and finger placement specifically

| Hand/work | Palm mechanism | Confidence |
|---|---|---|
| Schunk SVH | Palm joint bending inward with the thumb to form an arch | search snippet |
| DLR/HIT Hand II | Independent palm, 5 identical modular fingers | search snippet |
| RBO Hand 3 (arXiv 2201.10883) | Bellows actuator inside the palm for thumb/little-finger opposition; actuated spread | search snippet |
| Shadow Hand LF5 | Not confirmed this session | not found |
| Cobos / 25-DoF models | Arch as added CMC DoF | see §3 |
| MANO-to-URDF | Single rigid palm; stagger via per-finger MCP axis reference | verified |
| Function-based Parametric Co-Design | Extruded polygon palm; per-finger base offsets (closest to explicit stagger) | verified |
| House of Dextra | Convex hull of grip points, extruded; circular finger placement | verified |

No source combined "palm as a small branching sub-tree of bodies" with primitive geometry.

### 6. What controllers need

| Evidence | Implication | Confidence |
|---|---|---|
| CrossDex-style RL: fingertip + palm positions only, 80% across 4 hands, 35–39% zero-shot | Fingertip/contact-frame abstraction may suffice on the observation side; morphology detail still needed for retargeting | search snippet |
| GET-Zero: per-joint tokens + graph attention bias + FK self-modeling loss, ~20% zero-shot gain | Graph structure and per-joint geometry help; FK as auxiliary task is a verified technique | verified |
| UniMorphGrasp: tokens of joint geometry, limits, origin, axis | URDF-level joint features are a sufficient token featurization | search snippet |
| GraspGraphNet: URDF graph + differentiable FK; generalizes to finger removal | Kinematic-graph representation supports structural generalization in grasp generation | verified (partial) |
| House of Dextra: PPO conditioned on GNN morphology embedding | Graph embeddings are the standard design-to-controller interface in the closest co-design system | verified |
| Claim that padded joint supersets underperform graph representations | Unverified search synthesis; needs a primary ablation | not verified |
| DexFormer: implicit morphology from history | Explicit descriptor not strictly required near the training distribution | search snippet |

### Agent's synthesis (abridged)

(a) No truly universal robot-hand representation exists. Closest for the controller interface: GET-Zero, UniMorphGrasp, GraspGraphNet. Closest for generative palm+finger design: arXiv 2604.27557 and 2512.03743 (single-lab hand families, open chains, no couplings). Closest biomechanical templates: Cobos 2008 and arXiv 2512.07359.
(b) Palm strategies found: flat/curved extruded outline with fingers on its boundary; or single rigid palm with the arch folded into joint-axis choice or an extra CMC DoF. Neither gives the palm a branching sub-structure at different heights.
(c) No surveyed work uses a convex hull of capsules for the palm; solid palms use meshes with simulator-side convex decomposition. Capsules per finger link are precedented. A palm made of several bodies (one per mount, low-DoF joints for arch curvature) would be a departure from everything found.
(d) Per-joint tokens (origin, axis, type, limits, link geometry) over the real graph topology, plus an FK self-modeling loss, is the converging controller-side interface; the policy may consume a distilled subset (fingertip frames, contact points).
(e) Gaps: branching palm sub-structure; one representation covering real hardware hands and a generative design space; couplings and closed chains as first-class constructs; a primary ablation of padded supersets vs. graph representations for hands.

Unverified or out of time: Shadow LF5 palm joint; iLimb; OPH paper (403); GLSO; GeoDex; standalone "CrossDex"; "hand_comparison_tool".
