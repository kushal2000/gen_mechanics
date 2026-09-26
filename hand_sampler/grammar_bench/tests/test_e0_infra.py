"""E0 (experiment infrastructure) acceptance tests: proxy metrics, canonical
form / phenotype hashing, phenotype distance, small-step ``vary`` operators,
named ``Distribution`` variants, and the experiment runner.

The replay-identity constants below (``REPLAY_HASHES_SEED_0_99``) were
computed from ``HEAD`` (commit 84332cc, before any E0 edits to derive.py /
distributions.py) with::

    for seed in range(100):
        _, model = generate(seed)
        h = hashlib.sha256(to_json(model).encode("utf-8")).hexdigest()

so this test proves ``sample_derivation``/``derive`` under the (unchanged)
default distribution replay byte-identically after the E0 changes.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hand_sampler.grammar.adapters.json_io import to_json
from hand_sampler.grammar.canonical import canonical_form, phenotype_hash
from hand_sampler.grammar.derive import (
    OPERATORS,
    SMALL_STEP_OPERATORS,
    Derivation,
    VariationImpossible,
    derive,
    generate,
    vary,
)
from hand_sampler.grammar.distributions import DEFAULT_DISTRIBUTION
from hand_sampler.grammar.experiments.runner import registered_experiments, run_experiment
from hand_sampler.grammar.kinematics import AffineCoupling, Body, Frame, Joint, KinematicModel, validate
from hand_sampler.grammar.phenodist import phenotype_distance
from hand_sampler.grammar.proxy import all_proxies, tip_frames
from hand_sampler.grammar import variants


# ---------------------------------------------------------------------------
# Replay-identity constants (computed pre-edit; see module docstring).
# ---------------------------------------------------------------------------

REPLAY_HASHES_SEED_0_99: tuple = (
    'c3a46641b3829a1e8f72780fe83fb8d1a0ad77f6ea4b691eb62b2cdc50070193', '990ca2de10601f17ce1800ddc7743f33440ff717cfd0459cd28180388b420f30', 'ebe84c1c3c9265a8e48e5431a54208f409cb53488a6b8596e39ee24c8762946a', '04293d273f14631bd7f6a65f6a6e5737fa5fab12d02c5558a6bee85697dc5e08',
    '03125f93abc0f0b53a0e902b7462b6368a13273471d81786984eb7259e29cac2', 'a1f4e8e7fc60e98c029bf34dfc8371dd4b9baac93a403caaa0cbb5ec3c811299', 'e0603c55223549e587cc7f408feba4be1dc2683d469bddba0d635dd7338c66b0', '83f210627687c917f3b837a15b823b196e5749e9584a0c0440d49529d083a0bc',
    '5b85d26121ebe0e7f940eef04e58ca4e615ba640736c4960a6b41736eb1874de', '9a9748a5f7bf6d50c4fccd1ad3fa48fe5bb84ccb9c37bc4ab347578a75b78494', '9e6d445839b9fe67bd9e13c7066511b8ae08c5418e0721e8b7a934901504c199', '15f881d6bf212ba62a345f537db542915188b63ccb7852a1bb2a7f390a7b5966',
    '6dc51a89c5ee8b6a42eee7af780d0e80439eb0cce398208bf57239e3afa3d734', 'df4ebd19208f973dda9f47b62346f5a57026971da3d8f9dec6c0e59b98368ca2', 'f892c9f5dc6e54637277223b563055fed1ee6c4d73f2372dff8233d8aeb0ff2e', '1b0ca459c139430be21dfdd4dbfbc6f70ae1138cf1fae42ffdd9dc9ad3fc69c3',
    '06a86d6258051aa2b0e23b40a96bf759b868f0ba8db0d81323b582a4b4bc5686', '65d58a5765140d911f76e7e43bfc0a5128bacf73352804b1c84514f06796c91b', 'eb1eb9a7b67f2f86ffb49a2ac15413f2d8e3fdf5a414d75c9d806d5cd625b1aa', '28772014f399407735e03f564883c2a5136cb206fe5a0c709861962fcdb3884a',
    '9974d40294ee21be1118cfb57d265b94fc609fa0f4e644f17265ce52bbf6b109', '8b5983a42ac90707a73c26ce5f83a65b38c8589343ea874d37259060ac9bea8c', 'c3be00c5a2ab2ab30d9bd78f642540d9426fa931493d74b8fff918cc3b1b7149', '9bffe91c9709a01e57bb903fa964906e8dc3ecc9f63ab7a618be130bd5b7a70e',
    'c9901f5b0ca4fd545d5748d3e485fbcde659d3a3f41b67065ebf97bfee1b274d', '3d5f8f13b7f679942655f03097ba9f47f5a65f2d9fcf77beeef856cc9e7835e8', '8048158ae6a9dafe5d94b3e4de3e52f78cc630fcb08c2de97551640c398ab04d', 'badca0c7e8eb293d97ad2c95f7589e8e15f2945d957c0eed751c7259f2b8d6a9',
    'e2b211de5efbbd31d18d3891c57cdfb4d1d5df00bf56aba783ead9dce40449a0', '227cb9e0406303f09edc3a156082201e6a2511df18ec39517d6257ecbb754d03', 'e60c6b178fa2df06e52955b26a5cb1925ae054ede92c2b9bd871fd882f893139', 'c53bbf603a72b25c32240111fa2da56cf391ffe21c9d07828ce75acb75877761',
    '7d6fd759c3f2b959e66c1f28615787424102aa20da2ee1da427e6ffcf2092543', '05ee0f96dc5ae494285c7e221ab39f55d62aad47b8377676cf641b751cd6ca39', '54942380feb30ebbb4718e4f315b2926231fb20e745e5f44694a6458aa8eeb86', '317c97ed05b075a22596bbade729fc9766d461414fae1b623a78506102e0f339',
    'ceaf97ba88ab6c4d3bd7b2a0215c238482705b042d8c997d661ed659ec10e557', '4847ee68ea3c0c51fef6c6890ce726c70ce29080e4133aa5f7d154cdb50b5727', '032519b3b1c351273bc8851456c478ed0efc482ec5a2a475dc71cebe72866f2c', '5a0e60e7c0313fedd00cc085889eff5a710ee5b17749970f4b5ee6aadd2d6c66',
    '92b32ab0091a2a8d73af9dd3e5736bb0049a0906af273774141b9d0df9998502', 'cfeb0ca88aaf468b8501270a0914099fca5e0238ecdf8ea11549b98f169b4964', '5c2f9c38bd24775714c2a45a5689c97667d6712eadb31b4fa74cf1c62e53fad0', 'a34c144151aaa830309f7a0bd8239f2ffb82e46ce3e541c54001fa292a57c4d0',
    '13164b9de28b1633ff86d110241d9e60067e90455ae8dffba21eafbfb6b746b2', '1b2c362bc6ca579712d3a79090d05dc3a46e4f87064df1a45cb4fdc3dfb4bf88', '423b0676c6356d212ea5ffc96d92a368a3c9af45c7064ca12709e3827f10feb9', 'c74312e29b77737a87c3e937ff3c0b425d7cae596177cd4c8986974d890bf0e0',
    '9235733993b8962b9acdb8d86a05f4b657254adbc1feec76cdc9edf8d3854c3c', '8bd3e1117c82208b98dbe6d5a280255f7090b20c4e8b7cbb2c6ad2333863ed5a', 'ff708e6572c5fe1f31078dda9de14ab91e6b2d377bbc11ad029d9a5046783d98', '1cbee8851637f69b678ce188d3a8013afae84993c18861c7f960eb399c30b72b',
    '610d80374540c4495e2263dd082ab4d93c6d169fa9f345e5c2f6358418fc5857', 'b41120a1a0fa0619e465f2d6262d3e3baca0fb8d724ab179ad821fab55827fb9', 'db2727820cb17ece79f04a7891af0c7e473a661eacac8ab6f1c301ca532d7b37', '88d85ae48fb760ced5bca751b8067f7ab65d071c64a3389201d29b9ff31fd2de',
    '6140c72350b12d198ebb7db4a3fad9eb1eca3db0bd3615d0a24590b1f21c7378', '6d231c137a1e19460a25c6b9438a316aac617557951f7622788e000d788bc731', 'da642456e4ee9897667f843e8121f680026dfce7a820089c9cc94c3fea912e44', '89cc24ada3c22f571e7b3da288a55fa2a96432fb5527dd171642a189b2383b96',
    'e40cdd424194110510865ffe28c0b26686877723f880d0340ff0b9290bfb9664', '69913999f034bdec7cf4f0410a7822897e8d66d6596fc903f663d452804c3ae4', '7b1ad540e21aa5b1afb0e14758e0263f2ff12bb9e3e8183da3500c0106b7bce4', '40881ea9fac0acfcd5aab88c7542449abd81961f5b4d62c7fd4a4f904392cf7c',
    '7242990bdc5f4a12a7e8b1b946ae891db04990e2c162010a9dda3bf2c532ca72', 'a171dbd598ac1ead2fd43ac6d33122fd44cd75966c3c700f39c19d3d5b8405fe', 'ccd82dfba27d4f6701a41c3854f83d68c1625c34bb5615e7c7b5e953be31c493', 'eb4248ca86792ae3d8f956a322135014761987a2947473c6a908a09124e4ac2d',
    '8c8548245fe6c8e03f9cdd33c6a10cc28bfed86ae0122c5ea7f6f58baf08dfcd', 'c2d33f4b6f8a3432221f93dc59e8822a5371f4652b026e5668bcb85116e29008', 'ec24e67af20e4da6a65b2c78c1b5aaf28dbdaea99b9ae9965f604387c92e20dd', '686ed6e20f16f2df848c41f7f4d8c2c430b93afd638985f809ce2d926b28c666',
    '69b039f539aa19bb3b594a7a60b244a9f76997c6419ca7f472492ce203cd700c', '3388eb9a9eb5bbc97193125e24fccf6b295285bdcfa5ebdc77ea1c60acc0eb74', 'e959096bd89d3cbf2793a5edfa5295c9783b596f99ebb2955b25061b53b14e72', '2d83459fe094cdc88d26336dacc3091178602078c61cee9755f045b8b601065e',
    '69ca1ec0dd757b28723dfc5a592bbe1f0117eff30e1b577ebc098bddcceaf7d9', '436e952fcddc3c2082511ceb135bcc5a329ff2020351dcfc433b17b2e57b0a99', '4e9b657a22ec01c0e2e8ae7878ab9f2a60393184791485ca835f7af75fd47c4e', '3138514e8d7be0cc3addeec34701e9f0c07f4bd236a10fa89a80aa4947959825',
    '33c023078446880eab5a6f48ab20c138d7a3ddc596b28acb8be10c96ad0d7f5a', '189bb8df92c8a0111c0ad790cdeca816bd1c5dff78a0ecef11b3cb926e591c06', '6e936d4293f1f7987fc90f1a2e9c557a8b864a09b9e117710d1994c6e220032d', 'a198b3282cca3328f57d7c5ea480dd35eddb66140b89f415fe5e30b20eb0e95d',
    '6b8ea7d4c9ffadd278e1ab7b12a7e16cc8a84d4a9b6fd815f16608d57c242bea', 'b15188903f1b30b7d33b0f513732a4f442a1c8ce0ebc2ac1e6a0c6bd0bc54fec', '2032e742f0ce4206e9782062ec3c97a5d72b0d91c5ffa55398a9c30067c0dda1', '85da1dad8d9091b87801e4a3c747419321c9af1ee433415bb921934e6f5bbc95',
    'fcf1123f674a615fcdc7ef93c5355658045c9ac3e028e81e034a3ae21e3924f3', '1a353e08b533c6cf45964886cda753120c3059e907378ee1c8d25983015be5c8', '026812c26737d00ae874fb7a0595c2d5b4e39ce9af0685f8629aec5ec77abc00', '77458f1977ae5a238dcfd8716392737f84dd6479b096a5d1e6febccaa79177a3',
    '43ed34c55bf1d524851fd327e0acb732a496ac9e9656f9aba5f661653a386e8f', '1c5cc112035a7ec77f9e3e3ae23211a1037fd907ee1436c06af50238f74e6c58', '75ff86dcd44c62e366ada2dcbd6d0fea8e52b9dfbe7cb3fbe399122438ed555a', 'd48ea52eb41b4bf8a90ab943e466dcfaa511433a99a33ca35870191312b1948b',
    '49f21ce7a8bd5d5917cf804e5cdf0c95088686913fec2430214a5a63dc553337', 'dfc818eeadfa5e82782eaacff5d934798aa6718a66acd0d4dddb0f6e8eecadda', '554ff69d75e2aeb716acdf37bf31064a9ea773299ed3cf788a03a8999359d46d', 'a53cfb87e5b880bc75692b3cda6a179f045b84067564398cff212abcf7d2d327',
)

N_REPLAY_SEEDS = len(REPLAY_HASHES_SEED_0_99)


def test_replay_identity_seeds_0_99():
    assert N_REPLAY_SEEDS == 100
    for seed in range(N_REPLAY_SEEDS):
        _, model = generate(seed, DEFAULT_DISTRIBUTION)
        h = hashlib.sha256(to_json(model).encode("utf-8")).hexdigest()
        assert h == REPLAY_HASHES_SEED_0_99[seed], f"seed {seed} to_json hash changed"


# ---------------------------------------------------------------------------
# proxy.py
# ---------------------------------------------------------------------------

PROXY_SEEDS = range(20)


def test_proxies_deterministic_and_bounded():
    for seed in PROXY_SEEDS:
        _, model = generate(seed)
        r1 = all_proxies(model, seed, n_configs=16)
        r2 = all_proxies(model, seed, n_configs=16)
        assert r1 == r2, f"seed {seed}: all_proxies not deterministic"
        assert 0.0 <= r1["opposition"] <= 1.0
        assert 0.0 <= r1["antipodal_pinch"] <= 1.0
        assert r1["reach_coverage"] >= 0.0 and math.isfinite(r1["reach_coverage"])
        assert r1["structural_cost"] >= 0.0 and math.isfinite(r1["structural_cost"])
        assert r1["n_tips"] == len(tip_frames(model))


def test_proxy_opposition_and_pinch_zero_with_one_tip():
    # A hand with a single, single-phalanx digit has exactly one tip: every
    # pairwise/cross-digit proxy must be well-defined and zero.
    from hand_sampler.grammar.distributions import Distribution

    dist = Distribution(digit_count_range=(1, 1), phalanx_count_range=(1, 1),
                         palm_body_count_range=(0, 0), branch_probability=0.0)
    for seed in range(10):
        _, model = generate(seed, dist)
        r = all_proxies(model, seed, n_configs=8)
        assert r["opposition"] == 0.0
        assert r["antipodal_pinch"] == 0.0


# ---------------------------------------------------------------------------
# canonical.py
# ---------------------------------------------------------------------------


def test_canonical_hash_order_independent_but_geometry_dependent():
    saw_reordering_effect = False
    for seed in range(20):
        d0, m0 = generate(seed)
        h0 = phenotype_hash(m0)

        # Shuffle the derivation's own steps (order-of-application changes,
        # the derived geometry does not: derive() resolves by path/
        # readiness, not list position) -- the canonical hash must be
        # unaffected even when the raw to_json IS order-sensitive (checked
        # below, to prove the two are actually different properties, not
        # accidentally equal for every derivation shape).
        steps = list(d0.steps)
        random.Random(4242).shuffle(steps)
        d1 = Derivation(seed=d0.seed, grammar_version=d0.grammar_version, steps=tuple(steps))
        m1 = derive(d1)
        assert phenotype_hash(m1) == h0, f"seed {seed}: canonical hash changed under step reordering"
        if to_json(m0) != to_json(m1):
            saw_reordering_effect = True

    assert saw_reordering_effect, "no seed exercised to_json's order-sensitivity; test is not discriminating"

    # A genuinely different hand hashes differently.
    _, m_a = generate(0)
    _, m_b = generate(1)
    assert phenotype_hash(m_a) != phenotype_hash(m_b)


def test_canonical_hash_renaming_invariant():
    _, m0 = generate(3)
    h0 = phenotype_hash(m0)

    rename_b = {b.name: f"X_{b.name}" for b in m0.bodies}
    rename_j = {j.name: f"XJ_{j.name}" for j in m0.joints}
    m1 = KinematicModel(
        name="renamed", root=rename_b[m0.root],
        bodies=tuple(Body(name=rename_b[b.name], palm=b.palm, radius=b.radius) for b in m0.bodies),
        joints=tuple(Joint(name=rename_j[j.name], type=j.type, parent=rename_b[j.parent],
                            child=rename_b[j.child], origin=j.origin, axis=j.axis, limits=j.limits)
                     for j in m0.joints),
        frames=tuple(Frame(name=f"X_{f.name}", body=rename_b[f.body], pose=f.pose) for f in m0.frames),
        couplings=tuple(AffineCoupling(dependent=rename_j[c.dependent], source=rename_j[c.source],
                                        multiplier=c.multiplier, offset=c.offset) for c in m0.couplings),
    )
    assert phenotype_hash(m1) == h0


def test_canonical_form_is_valid_model():
    for seed in range(20):
        _, m = generate(seed)
        validate(canonical_form(m))


def test_canonical_hash_duplicate_rate_500_seeds():
    hashes = set()
    for seed in range(500):
        _, m = generate(seed)
        hashes.add(phenotype_hash(m))
    # 500 grammar draws over a huge parameter space essentially never
    # collide; assert no *unexpected* duplication (every seed distinct).
    assert len(hashes) == 500


# ---------------------------------------------------------------------------
# phenodist.py
# ---------------------------------------------------------------------------


def test_phenotype_distance_zero_for_identical_and_symmetric():
    # I14 fix: ``phenotype_distance`` now aligns configurations by joint
    # NAME, drawing only ``a``'s own u-configurations (``a`` = parent, ``b``
    # = child -- see phenodist.py's own docstring); an identical model
    # against itself therefore has EVERY key at 0 except ``n_shared_joints``
    # (every one of its own joints, by construction), for ANY seed.
    _, m0 = generate(5)
    _, m1 = generate(6)
    d_self = phenotype_distance(m0, m0, seed=1, n_configs=8)
    assert d_self["n_shared_joints"] == float(len(m0.joints))
    for k, v in d_self.items():
        if k == "n_shared_joints":
            continue
        assert v == 0.0, f"{k} nonzero for identical model: {v}"
    # A different seed changes nothing about self-distance (the noise floor
    # this fix removes): still exactly 0 for every structural/tip key.
    d_self_seed2 = phenotype_distance(m0, m0, seed=99, n_configs=8)
    for k, v in d_self_seed2.items():
        if k == "n_shared_joints":
            continue
        assert v == 0.0, f"{k} nonzero for identical model under a different seed: {v}"

    # The purely STRUCTURAL keys (independent of which configs were drawn)
    # remain exactly symmetric.
    d_ab = phenotype_distance(m0, m1, seed=1, n_configs=8)
    d_ba = phenotype_distance(m1, m0, seed=1, n_configs=8)
    structural_keys = (
        "joint_count_delta", "digit_count_delta", "palm_body_delta",
        "motor_delta", "total_length_delta_m",
    )
    for k in structural_keys:
        assert d_ab[k] == pytest.approx(d_ba[k], abs=1e-9), f"{k} not symmetric"
    # A genuinely different hand should show up as some nonzero delta.
    assert any(v > 0.0 for v in d_ab.values())


# ---------------------------------------------------------------------------
# derive.py small-step operators
# ---------------------------------------------------------------------------


def _single_field_diff(a: Derivation, b: Derivation):
    assert len(a.steps) == len(b.steps)
    diffs = [i for i, (sa, sb) in enumerate(zip(a.steps, b.steps)) if sa != sb]
    assert len(diffs) == 1, f"expected exactly one changed step, got {len(diffs)}"
    sa, sb = a.steps[diffs[0]], b.steps[diffs[0]]
    assert sa.path == sb.path and sa.production == sb.production
    field_diffs = [k for k in sa.params if sa.params[k] != sb.params[k]]
    assert len(field_diffs) == 1, f"expected exactly one changed field, got {field_diffs}"


def test_small_step_operators_change_exactly_one_field_and_replay():
    for opname in SMALL_STEP_OPERATORS:
        n_applied = 0
        for seed in range(60):
            d0, _ = generate(seed)
            rng = np.random.default_rng(1000 + seed)
            try:
                d1 = vary(d0, rng, operator=opname)
            except VariationImpossible:
                continue
            _single_field_diff(d0, d1)
            m1a = derive(d1)
            m1b = derive(d1)
            assert to_json(m1a) == to_json(m1b), f"{opname} seed {seed}: replay not exact"
            n_applied += 1
        assert n_applied >= 30, f"{opname} applied on too few seeds ({n_applied}/60)"


def test_default_vary_operator_pool_unchanged():
    assert set(SMALL_STEP_OPERATORS).isdisjoint(OPERATORS)
    n_ok = 0
    for seed in range(30):
        d0, _ = generate(seed)
        rng = np.random.default_rng(seed)
        try:
            d1 = vary(d0, rng)
        except VariationImpossible:
            continue
        op_used = d1.lineage[-1][0]
        assert op_used in OPERATORS
        assert op_used not in SMALL_STEP_OPERATORS
        n_ok += 1
    assert n_ok >= 20, f"default vary() succeeded on too few seeds ({n_ok}/30)"


def test_vary_operators_kwarg_reaches_small_step_pool():
    n_ok = 0
    for seed in range(20):
        d0, _ = generate(seed)
        rng = np.random.default_rng(seed)
        try:
            d1 = vary(d0, rng, operators=SMALL_STEP_OPERATORS)
        except VariationImpossible:
            continue
        op_used = d1.lineage[-1][0]
        assert op_used in SMALL_STEP_OPERATORS
        n_ok += 1
    assert n_ok >= 15, f"operators= kwarg succeeded on too few seeds ({n_ok}/20)"


# ---------------------------------------------------------------------------
# variants.py
# ---------------------------------------------------------------------------

NAMED_VARIANT_ITEMS = tuple(variants.NAMED_DISTRIBUTIONS.items())


def test_variants_sample_valid_models():
    for name, dist in NAMED_VARIANT_ITEMS:
        for seed in range(50):
            _, model = generate(seed, dist)
            validate(model)  # raises ModelError on failure


def test_g_full_is_default_distribution():
    assert variants.G_FULL is DEFAULT_DISTRIBUTION


def test_g_serial_has_one_palm_body_and_no_branches():
    for seed in range(50):
        d, m = generate(seed, variants.G_SERIAL)
        n_palm = sum(1 for b in m.bodies if b.palm)
        assert n_palm == 1, f"seed {seed}: expected 1 palm body, got {n_palm}"
        for s in d.steps:
            if s.production == "Phalanx":
                assert s.params["branch_digit_count"] == 0


def test_g_nocouple_never_samples_coupled_module():
    for seed in range(50):
        d, _ = generate(seed, variants.G_NOCOUPLE)
        for s in d.steps:
            if s.production == "Phalanx":
                assert s.params["module"]["kind"] != "Coupled"


def test_g_full_small_pairs_g_full_with_small_step_operators():
    dist, operators = variants.G_FULL_SMALL
    assert dist is variants.G_FULL
    assert tuple(operators) == SMALL_STEP_OPERATORS


# ---------------------------------------------------------------------------
# experiments/runner.py
# ---------------------------------------------------------------------------


def test_runner_writes_result_json_and_summary_md():
    fns = registered_experiments()
    assert "smoke" in fns
    smoke = fns["smoke"]
    with tempfile.TemporaryDirectory() as td:
        out_dir = str(Path(td) / "smoke_out")
        result = run_experiment(
            "smoke", smoke, params={"n_configs": 8}, seeds=list(range(6)),
            out_dir=out_dir, processes=3,
        )
        result_path = Path(out_dir) / "result.json"
        summary_path = Path(out_dir) / "summary.md"
        assert result_path.exists()
        assert summary_path.exists()
        with open(result_path) as f:
            loaded = json.load(f)

        for key in (
            "name", "params", "seeds", "per_seed", "aggregate", "git_sha", "git_dirty",
            "python_version", "numpy_version", "grammar_version", "wall_time_s",
        ):
            assert key in loaded, f"missing required result.json field {key!r}"

        assert loaded["name"] == "smoke"
        assert loaded["seeds"] == list(range(6))
        assert len(loaded["per_seed"]) == 6
        assert loaded["aggregate"]["n_seeds"] == 6
        assert loaded["aggregate"]["n_ok"] == 6
        assert loaded["wall_time_s"] >= 0.0
        # Returned dict is the same content that was written to disk.
        assert result["name"] == loaded["name"]
        assert result["aggregate"]["n_ok"] == loaded["aggregate"]["n_ok"]

        summary_text = summary_path.read_text()
        assert "smoke" in summary_text
        assert "Aggregate" in summary_text


def test_runner_cli_list(capsys):
    from hand_sampler.grammar.experiments.runner import main

    rc = main(["--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "smoke" in out
