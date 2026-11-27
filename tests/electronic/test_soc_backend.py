"""The spin-orbit Hamiltonian returned by `PySCFBackend` (spec 2.2, 4 items 3-5): its layout, the
reference density of the SOMF operator, rotation and translation invariance, phase continuity
along a C=O stretch and on resume, and the unchanged paths without it.

System: formaldehyde (tests/data/formaldehyde.xyz) / 6-31G, SA-CASSCF(4,3) 2 singlets + 2
triplets and CIS (TDA-HF) 3 singlets + 3 triplets. The bounds are the spec's (section 4).
"""

from pathlib import Path

import numpy as np
import pytest

from gqteashmd.chem.xyz import parse_xyz
from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.units import ANGSTROM_TO_BOHR, HARTREE_IN_CM1
from gqteashmd.dynamics.spin import expanded_states
from gqteashmd.electronic import soc
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.pyscf_backend import PySCFBackend

pytestmark = pytest.mark.pyscf
DATA = Path(__file__).resolve().parents[1] / "data"
_GEOMETRY = parse_xyz((DATA / "formaldehyde.xyz").read_text())
SYMBOLS = _GEOMETRY.symbols
COORDS = np.array(_GEOMETRY.coords_angstrom) * ANGSTROM_TO_BOHR

HERMITIAN_BOUND = 1e-12  # Eh, spec 4 item 3
INVARIANCE_BOUND = 1e-7  # relative, spec 4 item 3
SIGN_THRESHOLD_CM1 = 1.0  # spec 4 item 4: no element above 1 cm-1 changes sign
STEP_FRACTION, STEP_FLOOR_CM1 = 0.10, 0.05  # spec 4 item 4: per-step change <= 10 % + 0.05 cm-1
RESUME_BOUND = 1e-10  # Eh, spec 4 item 4
STRETCH_BOHR = 0.005  # per step, the O atom along the C-O axis
N_POINTS = 5

SETTINGS = {
    "casscf": {
        "method": "casscf",
        "basis": "6-31g",
        "singlets": 2,
        "triplets": 2,
        "scf_conv_tol": 1e-12,
        "casscf": {"ncas": 3, "nelecas": 4, "conv_tol": 1e-12, "conv_tol_grad": 1e-7},
    },
    "tda": {
        "method": "tda",
        "basis": "6-31g",
        "functional": "hf",
        "singlets": 3,
        "triplets": 3,
        "scf_conv_tol": 1e-12,
        "td_conv_tol": 1e-10,
    },
}


def _settings(method: str, **update) -> ElectronicSection:
    return ElectronicSection.model_validate({**SETTINGS[method], **update})


def _backend(method: str, **update) -> PySCFBackend:
    return PySCFBackend(MoleculeSpec(SYMBOLS), _settings(method, **update))


def _counts(method: str) -> tuple[int, int]:
    return SETTINGS[method]["singlets"], SETTINGS[method]["triplets"]


def _pair_rss(h: np.ndarray, n_s: int, n_t: int) -> dict[str, float]:
    """Root sum of squares (Eh) per spin-free pair: S_I-T_J over its 3 elements, T_J-T_J' (J <=
    J') over its 9 (6 for J = J', the diagonal being the energies)."""

    def index(j: int, m: int) -> int:
        return n_s + m * n_t + j

    rss = {}
    for i in range(n_s):
        for j in range(n_t):
            rss[f"S{i}-T{j + 1}"] = float(
                np.sqrt(sum(abs(h[i, index(j, m)]) ** 2 for m in range(3)))
            )
    for j in range(n_t):
        for jb in range(j, n_t):
            total = sum(
                abs(h[index(j, m), index(jb, mb)]) ** 2
                for m in range(3)
                for mb in range(3)
                if (j, m) != (jb, mb)
            )
            rss[f"T{j + 1}-T{jb + 1}"] = float(np.sqrt(total))
    return rss


@pytest.fixture(scope="module", params=list(SETTINGS))
def computed(request):
    """(method, result with soc=True) at the formaldehyde geometry."""
    backend = _backend(request.param)
    return request.param, backend.compute(Request(COORDS, soc=True))


def test_the_hamiltonian_layout(computed):
    method, result = computed
    n_s, n_t = _counts(method)
    h = result.hamiltonian
    assert h is not None and h.shape == (n_s + 3 * n_t, n_s + 3 * n_t)
    assert np.max(np.abs(h - h.conj().T)) <= HERMITIAN_BOUND
    assert np.all(h[:n_s, :n_s] == np.diag(np.diag(h[:n_s, :n_s])))  # S-S exactly zero
    energies = [result.energies[s] for s in result.states]
    singlet_e = energies[:n_s]
    triplet_e = energies[n_s:]
    assert np.all(np.diag(h).real == np.array(singlet_e + triplet_e * 3))
    assert np.all(np.diag(h).imag == 0.0)
    assert result.hamiltonian_labels == tuple(s.label() for s in expanded_states((n_s, 0, n_t)))
    off = h - np.diag(np.diag(h))
    assert np.max(np.abs(off)) * HARTREE_IN_CM1 > 10.0  # non-vacuous: S0-T1 is about 65 cm-1


def test_the_json_carries_the_hamiltonian(computed):
    _, result = computed
    data = result.to_json()
    h = result.hamiltonian
    assert np.array(data["hamiltonian"]["real"]).tolist() == h.real.tolist()
    assert np.array(data["hamiltonian"]["imag"]).tolist() == h.imag.tolist()
    assert data["hamiltonian_labels"] == list(result.hamiltonian_labels)


@pytest.mark.parametrize("method", list(SETTINGS))
def test_the_somf_operator_uses_the_reference_density(method, monkeypatch):
    """PR4: the state-averaged density for SA-CASSCF, the SCF density for TDA."""
    seen = []
    original = soc.soc_operator

    def spy(mol, dm, kind):
        seen.append((np.array(dm), kind))
        return original(mol, dm, kind)

    monkeypatch.setattr(soc, "soc_operator", spy)
    backend = _backend(method)
    backend.compute(Request(COORDS, soc=True))
    live = backend._live
    expected = live.solver.make_rdm1() if method == "casscf" else live.mf.make_rdm1()
    assert len(seen) == 1 and seen[0][1] == "somf"
    assert np.max(np.abs(seen[0][0] - expected)) <= 1e-12
    if method == "casscf":  # the state average differs from the SCF density
        assert np.max(np.abs(seen[0][0] - live.mf.make_rdm1())) > 1e-3


def test_the_effective_charge_operator_is_used_when_chosen(monkeypatch):
    seen = []
    original = soc.soc_operator

    def spy(mol, dm, kind):
        seen.append(kind)
        return original(mol, dm, kind)

    monkeypatch.setattr(soc, "soc_operator", spy)
    somf = _backend("tda").compute(Request(COORDS, soc=True)).hamiltonian
    zeff = _backend("tda", soc_operator="effective_charge").compute(Request(COORDS, soc=True))
    assert seen == ["somf", "effective_charge"]
    n_s, n_t = _counts("tda")
    s0_t1_somf = _pair_rss(somf, n_s, n_t)["S0-T1"] * HARTREE_IN_CM1
    s0_t1_zeff = _pair_rss(zeff.hamiltonian, n_s, n_t)["S0-T1"] * HARTREE_IN_CM1
    assert abs(s0_t1_zeff - s0_t1_somf) > 1.0  # the two operators differ (Task 2: 70.8 vs 63.1)


def test_soc_without_triplets_is_refused():
    backend = _backend("tda", triplets=0)
    with pytest.raises(ValueError, match="triplet"):
        backend.compute(Request(COORDS, soc=True))


def _forbid_spin_orbit_integrals(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a spin-orbit integral was computed")

    for name in ("soc_operator", "somf_operator", "one_electron_operator"):
        monkeypatch.setattr(soc, name, refuse)
    monkeypatch.setattr(soc, "effective_charge_operator", refuse)


@pytest.mark.parametrize("triplets", [0, 3])
def test_no_spin_orbit_integral_without_the_request(triplets, monkeypatch):
    _forbid_spin_orbit_integrals(monkeypatch)
    backend = _backend("tda", triplets=triplets)
    result = backend.compute(Request(COORDS))
    assert result.hamiltonian is None and result.hamiltonian_labels is None
    second = backend.compute(Request(COORDS + 0.01, overlaps=True))  # a tracked geometry too
    assert second.hamiltonian is None
    data = second.to_json()
    assert "hamiltonian" not in data and "hamiltonian_labels" not in data


def _rotation(angle_deg: float, axis: np.ndarray) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    a = np.deg2rad(angle_deg)
    return np.eye(3) + np.sin(a) * kx + (1.0 - np.cos(a)) * kx @ kx


@pytest.mark.parametrize("method", list(SETTINGS))
def test_the_pair_magnitudes_are_invariant_under_rotation_and_translation(method):
    """Spec 4 item 3. A pair is compared relative to its own size, with a floor of 1e-3 of the
    largest pair, so that elements zero by symmetry (C2v to 4e-7 A) do not divide by zero."""
    n_s, n_t = _counts(method)
    rotation = _rotation(30.0, np.array([1.0, 2.0, 3.0]))  # no symmetry axis of the molecule
    moved = COORDS @ rotation.T + np.array([0.3, -0.2, 0.5])
    reference = _pair_rss(_backend(method).compute(Request(COORDS, soc=True)).hamiltonian, n_s, n_t)
    other = _pair_rss(_backend(method).compute(Request(moved, soc=True)).hamiltonian, n_s, n_t)
    floor = 1e-3 * max(reference.values())
    worst = max(abs(other[k] - reference[k]) / max(reference[k], floor) for k in reference)
    print(f"{method}: largest relative pair difference {worst:.3e}")
    print({k: round(v * HARTREE_IN_CM1, 4) for k, v in reference.items()})
    assert worst <= INVARIANCE_BOUND


def _stretched(k: int) -> np.ndarray:
    """Point k of the path: the O atom moved by k * STRETCH_BOHR along the C-O axis."""
    coords = COORDS.copy()
    axis = coords[1] - coords[0]
    coords[1] += k * STRETCH_BOHR * axis / np.linalg.norm(axis)
    return coords


@pytest.mark.parametrize("method", list(SETTINGS))
def test_the_phased_elements_are_continuous_along_a_stretch(method):
    backend = _backend(method)
    path = [
        backend.compute(Request(_stretched(k), soc=True, overlaps=True)).hamiltonian
        for k in range(N_POINTS)
    ]
    n = path[0].shape[0]
    off = ~np.eye(n, dtype=bool)
    worst_ratio, smallest_kept = 0.0, np.inf
    for k in range(N_POINTS - 1):
        a, b = path[k][off] * HARTREE_IN_CM1, path[k + 1][off] * HARTREE_IN_CM1
        big = (np.abs(a) > SIGN_THRESHOLD_CM1) & (np.abs(b) > SIGN_THRESHOLD_CM1)
        same_sign = np.real(a * np.conj(b)) > 0.0
        assert np.all(same_sign[big]), f"step {k}: an element above 1 cm-1 changed sign"
        if np.any(big):
            smallest_kept = min(smallest_kept, float(np.min(np.abs(a[big]))))
        allowance = STEP_FRACTION * np.abs(a) + STEP_FLOOR_CM1
        change = np.abs(b - a)
        worst_ratio = max(worst_ratio, float(np.max(change / allowance)))
        assert np.all(change <= allowance), f"step {k}: an element changed by more than allowed"
    print(
        f"{method}: largest per-step change / allowance {worst_ratio:.4f}; "
        f"smallest |element| above 1 cm-1 that kept its sign {smallest_kept:.4f} cm-1"
    )


@pytest.mark.parametrize("method", list(SETTINGS))
def test_a_resumed_chain_gives_the_same_phased_hamiltonian(method, tmp_path):
    """Backend A: points 0-4 with a save after point 2; backend C loads that file and computes
    points 3-4. One save/load chain, so both share A's sign frame.

    Owner decision 2026-11-27 (after the numbers were seen): the 1e-10 Eh bound applies to the
    off-diagonal spin-orbit elements only; the diagonal (spin-free energies) is printed, not
    asserted. Measured for TDA: in 9 of 14 runs a diagonal energy differed by about 1.3e-8 Eh,
    because the two SCF runs at point 3 stopped one cycle apart at the default gradient
    criterion (scf_conv_tol 1e-12, gradient 1e-6); the off-diagonal elements stayed within
    2.2e-11 Eh."""
    a = _backend(method)
    state = tmp_path / "state.npz"
    first = {}
    for k in range(N_POINTS):
        first[k] = a.compute(Request(_stretched(k), soc=True, overlaps=True)).hamiltonian
        if k == 2:
            a.save_state(state)
    c = _backend(method)
    c.load_state(state)
    worst_off, worst_diagonal = 0.0, 0.0
    for k in (3, 4):
        resumed = c.compute(Request(_stretched(k), soc=True, overlaps=True)).hamiltonian
        difference = np.abs(resumed - first[k])
        off = ~np.eye(difference.shape[0], dtype=bool)
        worst_off = max(worst_off, float(np.max(difference[off])))
        worst_diagonal = max(worst_diagonal, float(np.max(np.diag(difference))))
    print(
        f"{method}: largest resume difference off the diagonal {worst_off:.3e} Eh, "
        f"on the diagonal {worst_diagonal:.3e} Eh (not asserted)"
    )
    assert worst_off <= RESUME_BOUND
