"""Spin-orbit couplings against ORCA 6.1.1, run as a black box (spec 4 items 1-2).

Systems: formaldehyde and thioformaldehyde (tests/data/*.xyz, RHF/6-31G* geometries; the ORCA
inputs use the same coordinates) in 6-31G; SA-CASSCF CAS(4,3) 2 singlets + 2 triplets with equal
weights, and CIS (TDA-HF) 3 singlets (S0 included) + 3 triplets; the SOMF operator (ORCA `SOCType
3`, `SOCFlags 1,4,4,0`, no RI) and the effective-charge operator (ORCA `SOCType 1`).

The compared quantity is the root sum of squares per spin-free pair, which does not depend on the
triplet basis: ORCA's TDA prints the Cartesian components Z, X, Y of <T|H|S>, its CASSCF the Ms
components, and ours are Ms components. Each distinct matrix element is counted once (the
definition of the `single-point` table, `cli._print_soc_table`): S_I-T_J over its 3 elements,
T_J-T_J' (J < J') over its 9, T_J-T_J over the 3 with M < M' (zero for real states). Signs and
phases are not compared (the CI phases are arbitrary).

Rule (spec 4, fixed before any run): |ours - ORCA| <= max(0.05 cm-1, 0.2 % of ORCA) per pair.
Pairs that are zero by symmetry are compared under the same rule: the C2v symmetry of the
geometries is exact only to about 4e-7 Angstrom, so no exact zero is assumed on either side.

Before any coupling is compared, the states are mapped by spin-free excitation energy and the
energies are asserted to agree within 1e-6 Eh (so the comparison is between the same solutions).
ORCA's TDA is asked for 6 roots per multiplicity, twice the number needed: with 3 roots it skips
the true S3 of formaldehyde (Task 1 of the plan).

Not tested here: which density ORCA's CASSCF SOMF uses. Its effect (about 0.1 %, measured in
Task 1 by changing the state weights at fixed CI vectors) is below the 0.2 % rule, so this
comparison cannot tell a state-averaged density from another choice.
"""

import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gqteashmd.chem.xyz import parse_xyz
from gqteashmd.config.schema import ElectronicSection
from gqteashmd.core.states import StateLabel
from gqteashmd.core.units import ANGSTROM_TO_BOHR, HARTREE_IN_CM1
from gqteashmd.electronic.base import MoleculeSpec, Request
from gqteashmd.electronic.pyscf_backend import PySCFBackend

pytestmark = [
    pytest.mark.orca,
    pytest.mark.pyscf,
    pytest.mark.skipif(not os.environ.get("ORCA_EXE"), reason="needs ORCA_EXE"),
]
DATA = Path(__file__).resolve().parents[1] / "data"
MOLECULES = ("formaldehyde", "thioformaldehyde")

FLOOR_CM1, FRACTION = 0.05, 0.002  # spec 4: max(0.05 cm-1, 0.2 % of ORCA) per pair
ENERGY_TOLERANCE = 1e-6  # Eh, the spin-free excitation energies of the mapped states
TDA_ROOTS = 6  # ORCA roots per multiplicity: at least twice the 2 + 3 needed (controller ruling)

ORCA_OPERATOR = {
    "somf": "%rel SOCType 3 SOCFlags 1,4,4,0 end",
    "effective_charge": "%rel SOCType 1 end",
}
SETTINGS = {
    "casscf": {
        "method": "casscf",
        "basis": "6-31g",
        "singlets": 2,
        "triplets": 2,
        "scf_conv_tol": 1e-12,
        # as converged as ORCA's etol 1e-10 / gtol 1e-6 (Task 1: the solutions agree to 3e-8 Eh)
        "casscf": {"ncas": 3, "nelecas": 4, "conv_tol": 1e-11, "conv_tol_grad": 1e-6},
    },
    "tda": {
        "method": "tda",
        "basis": "6-31g",
        "functional": "hf",
        "singlets": 3,
        "triplets": 3,
        "scf_conv_tol": 1e-12,
        # In PySCF 2.14 td_conv_tol is the Davidson residual tolerance (tol_residual).
        # `_lr_eig.eigh` stops with "Linear dependency in trial subspace" once no new trial
        # vector has xt_norm**2 > td.lindep (default 1e-12). For thioformaldehyde that leaves
        # residuals of about 2.7e-8 on S2, T2 and T3 (measured), so 1e-8 and below are refused
        # as unconverged. The energy error is second order in the residual, the coupling error
        # first order (about 1e-6 relative).
        "td_conv_tol": 1e-7,
    },
}

STATE_LINE = re.compile(
    r"^STATE\s+(\d+):\s+E=\s+\S+ au\s+\S+ eV\s+(\d+\.\d+) cm\*\*-1.*Mult (\d)\s*$", re.M
)
COMPLEX = r"\(\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\)"
SOCME_ROW = re.compile(rf"^\s*(\d+)\s+(\d+)\s+{COMPLEX}\s+{COMPLEX}\s+{COMPLEX}\s*$")
SOCME_HEADER = "CALCULATED SOCME BETWEEN TRIPLETS AND SINGLETS"
NONZERO_HEADER = "NONZERO SOC MATRIX ELEMENTS (cm**-1)"
NUMBER = r"(-?\d+(?:\.\d+)?)"
NONZERO_ROW = re.compile(
    rf"^\s*(\d+)\s+(\d+)\s+{NUMBER}\s+{NUMBER}\s+(\d+)\s+(\d+)\s+{NUMBER}\s+{NUMBER}"
    rf"\s+{NUMBER}\s+{NUMBER}\s*$"
)


class OrcaOutputError(AssertionError):
    """A block the parser needs is missing or malformed in ORCA's output."""


# --- ORCA ------------------------------------------------------------------------------------


def _atom_lines(molecule: str) -> list[str]:
    """The coordinate lines of the committed XYZ file, verbatim (Angstrom, 10 decimals)."""
    return [line.strip() for line in (DATA / f"{molecule}.xyz").read_text().splitlines()[2:]]


def _orca_input(molecule: str, method: str, operator: str) -> str:
    """Task 1's inputs (`tda_*_soc3/1.inp`, `cas_*_soc3/1_pl3.inp`), TDA with 6 roots."""
    if method == "tda":
        header = [
            "! RHF 6-31G NoRI NoFrozenCore TightSCF",
            f"%tddft nroots {TDA_ROOTS} tda true triplets true dosoc true end",
            ORCA_OPERATOR[operator],
        ]
    else:
        header = [
            "! 6-31G NoRI TightSCF",
            ORCA_OPERATOR[operator],
            "%casscf",
            "  nel 4",
            "  norb 3",
            "  mult 1,3",
            "  nroots 2,2",
            "  bweight 2,2",
            "  weights[0] = 1,1",
            "  weights[1] = 1,1",
            "  etol 1e-10",
            "  gtol 1e-6",
            "  rel",
            "    dosoc true",
            "    printlevel 3",
            "  end",
            "end",
        ]
    return "\n".join([*header, "* xyz 0 1", *_atom_lines(molecule), "*"]) + "\n"


def _run_orca(directory: Path, text: str) -> str:
    (directory / "input.inp").write_text(text)
    done = subprocess.run(
        [os.environ["ORCA_EXE"], "input.inp"],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=900,
    )
    (directory / "output.out").write_text(done.stdout)
    if done.returncode != 0 or "ORCA TERMINATED NORMALLY" not in done.stdout:
        raise RuntimeError(
            f"ORCA failed (exit {done.returncode}) in {directory}; see output.out:\n"
            + done.stdout[-2000:]
            + "\nstderr:\n"
            + done.stderr[-2000:]
        )
    return done.stdout


def _section(out: str, header: str, what: str) -> str:
    """The text after the only occurrence of `header`; a missing or repeated block fails."""
    count = out.count(header)
    if count != 1:
        raise OrcaOutputError(f"ORCA output: expected one '{header}' block ({what}), found {count}")
    return out.split(header, 1)[1]


def _scf_energy(out: str) -> float:
    """The SCF energy (Eh) from `Total Energy       :     -113.80712844433204 Eh`."""
    found = re.findall(r"^Total Energy\s+:\s+(-?\d+\.\d+) Eh", out, re.M)
    if len(found) != 1:
        raise OrcaOutputError(f"ORCA output: expected one 'Total Energy' line, found {len(found)}")
    return float(found[0])


def _tda_states(out: str) -> dict[int, list[float]]:
    """{multiplicity: excitation energies (Eh)} in ORCA's root order, from the `STATE  n:  E= ...
    Mult m` lines of `CIS-EXCITED STATES (SINGLETS)` and `CIS EXCITED STATES (TRIPLETS)`.

    The au column has 6 decimals (5e-7 Eh); the cm**-1 column (0.05 cm-1 = 2.3e-7 Eh) is parsed
    and divided by HARTREE_IN_CM1. The SOC eigenvalue list (` STATE  0:      0.00`) has no `E=`
    and is not matched.
    """
    for header in ("CIS-EXCITED STATES (SINGLETS)", "CIS EXCITED STATES (TRIPLETS)"):
        _section(out, header, "the TDA excited states")
    states: dict[int, list[float]] = {1: [], 3: []}
    for _, cm1, mult in STATE_LINE.findall(out):
        states[int(mult)].append(float(cm1) / HARTREE_IN_CM1)
    for mult, energies in states.items():
        if len(energies) != TDA_ROOTS:
            raise OrcaOutputError(
                f"ORCA output: {len(energies)} TDA states of multiplicity {mult}, "
                f"expected {TDA_ROOTS} ('STATE  n:  E= ... Mult {mult}' lines)"
            )
    return states


def _tda_socme(out: str) -> dict[tuple[int, int], np.ndarray]:
    """{(T, S): the 3 complex components (cm-1)} from `CALCULATED SOCME BETWEEN TRIPLETS AND
    SINGLETS`: T = 1.. the triplet roots, S = 0 the SCF ground state and 1.. the singlet roots,
    columns Z, X, Y as (Re, Im) with 2 decimals; every pair is printed, zeros too."""
    lines = _section(out, SOCME_HEADER, "the TDA spin-orbit couplings").splitlines()
    elements: dict[tuple[int, int], np.ndarray] = {}
    started = False
    for line in lines[1:]:
        match = SOCME_ROW.match(line)
        if match:
            started = True
            t, s, *parts = match.groups()
            values = [float(p) for p in parts]
            elements[int(t), int(s)] = np.array(
                [complex(values[k], values[k + 1]) for k in (0, 2, 4)]
            )
        elif started and not line.strip():
            break
    expected = {(t, s) for t in range(1, TDA_ROOTS + 1) for s in range(TDA_ROOTS + 1)}
    if set(elements) != expected:
        raise OrcaOutputError(
            f"ORCA output: the '{SOCME_HEADER}' block has {len(elements)} rows, expected every "
            f"(T, S) with T = 1..{TDA_ROOTS}, S = 0..{TDA_ROOTS}"
        )
    return elements


def _casscf_states(out: str) -> dict[int, tuple[int, list[float]]]:
    """{multiplicity: (block number, [state energies (Eh)])} from the `CAS-SCF STATES FOR BLOCK
    k MULT= m` blocks after `CASSCF RESULTS` (ORCA re-orders the blocks: block 0 = triplets)."""
    final = _section(out, "CASSCF RESULTS", "the final CASSCF states")
    blocks = re.findall(
        r"CAS-SCF STATES FOR BLOCK\s+(\d+) MULT=\s*(\d+) NROOTS=\s*(\d+)(.*?)(?=\n\n\n)",
        final,
        re.S,
    )
    states = {}
    for block, mult, nroots, body in blocks:
        energies = [
            float(e) for e in re.findall(r"^ROOT\s+\d+:\s+E=\s+(-?\d+\.\d+) Eh", body, re.M)
        ]
        if len(energies) != int(nroots):
            raise OrcaOutputError(
                f"ORCA output: CASSCF block {block} (MULT {mult}) has {len(energies)} "
                f"'ROOT  i:  E=' lines, expected {nroots}"
            )
        states[int(mult)] = (int(block), energies)
    if sorted(states) != [1, 3] or any(len(e) != 2 for _, e in states.values()):
        raise OrcaOutputError(
            "ORCA output: expected CAS-SCF STATES blocks for MULT 1 and 3 with 2 roots each, "
            f"found {sorted(states)}"
        )
    return states


def _casscf_soc_lines(out: str) -> dict[frozenset, complex]:
    """The `NONZERO SOC MATRIX ELEMENTS (cm**-1)` lines (`printlevel 3` in the `rel` sub-block of
    `%casscf`): `Block Root S Ms | Block Root S Ms = Re Im`, 3 decimals, one triangle only.

    Returned as {frozenset of the two (block, root, Ms) states: element}, so an element and its
    conjugate count once. Only elements above ORCA's print threshold are listed (which also lets
    some rounded zeros through); an absent element is below it.
    """
    lines = _section(out, NONZERO_HEADER, "the CASSCF spin-orbit couplings").splitlines()
    elements: dict[frozenset, complex] = {}
    for line in lines[1:]:
        if line.strip().startswith("Note:"):  # the full SOC matrix follows
            break
        match = NONZERO_ROW.match(line)
        if not match:
            if line.strip()[:1].isdigit():
                raise OrcaOutputError(
                    f"ORCA output: unreadable line in the '{NONZERO_HEADER}' block: {line!r}"
                )
            continue
        b1, r1, _, m1, b2, r2, _, m2, re_part, im_part = match.groups()
        key = frozenset({(int(b1), int(r1), float(m1)), (int(b2), int(r2), float(m2))})
        if key in elements:
            raise OrcaOutputError(f"ORCA output: the element {sorted(key)} is printed twice")
        elements[key] = complex(float(re_part), float(im_part))
    if not elements:
        raise OrcaOutputError(f"ORCA output: the '{NONZERO_HEADER}' block has no element line")
    return elements


# --- the comparison --------------------------------------------------------------------------


def _map_by_energy(ours: list[float], theirs: list[float], what: str, first: int) -> list[int]:
    """For each of our excitation energies, the index of ORCA's state at the same energy; the
    energies must agree within ENERGY_TOLERANCE and the map must be one to one. Our states are
    labelled `what` + first, first + 1, ... in the printout."""
    mapping = []
    for k, energy in enumerate(ours, start=first):
        diffs = [abs(energy - t) for t in theirs]
        best = int(np.argmin(diffs))
        print(
            f"  energy {what}{k}: ours {energy:.9f}  ORCA {theirs[best]:.9f}  "
            f"diff {energy - theirs[best]:+.2e} Eh"
        )
        assert diffs[best] <= ENERGY_TOLERANCE, (
            f"{what}{k}: no ORCA state within {ENERGY_TOLERANCE} Eh of {energy:.9f} Eh "
            f"(nearest {theirs[best]:.9f}): the two programs are on different solutions"
        )
        mapping.append(best)
    assert len(set(mapping)) == len(mapping), f"{what}: two of our states map to one ORCA state"
    return mapping


def _check_s0(ours: float, orca: float) -> None:
    print(f"  energy S0 (absolute): ours {ours:.10f}  ORCA {orca:.10f}  diff {ours - orca:+.2e} Eh")
    assert abs(ours - orca) <= ENERGY_TOLERANCE, "S0: the two programs are on different solutions"


def _our_pairs(h: np.ndarray, n_s: int, n_t: int) -> dict[str, float]:
    """Root sum of squares (cm-1) per spin-free pair of our spin-expanded Hamiltonian (canonical
    order S0..S(n-1), then T1..Tm for Ms = -1, 0, +1), each distinct element counted once."""
    cm1 = h * HARTREE_IN_CM1

    def index(j: int, m: int) -> int:
        return n_s + m * n_t + j

    pairs = {}
    for i in range(n_s):
        for j in range(n_t):
            pairs[f"S{i}-T{j + 1}"] = float(
                np.sqrt(sum(abs(cm1[i, index(j, m)]) ** 2 for m in range(3)))
            )
    for j in range(n_t):
        for jb in range(j, n_t):
            total = sum(
                abs(cm1[index(j, m), index(jb, mb)]) ** 2
                for m in range(3)
                for mb in range(3)
                if j != jb or m < mb
            )
            pairs[f"T{j + 1}-T{jb + 1}"] = float(np.sqrt(total))
    return pairs


def _ours(molecule: str, method: str, operator: str):
    """(spin-free energies {0: [E(S0)], 1: singlet and 3: triplet excitation energies} in Eh,
    pair rss {pair: cm-1})."""
    geometry = parse_xyz((DATA / f"{molecule}.xyz").read_text())
    coords = np.array(geometry.coords_angstrom) * ANGSTROM_TO_BOHR
    settings = ElectronicSection.model_validate({**SETTINGS[method], "soc_operator": operator})
    result = PySCFBackend(MoleculeSpec(geometry.symbols), settings).compute(
        Request(coords, soc=True)
    )
    n_s, n_t = settings.singlets, settings.triplets
    e0 = result.energies[StateLabel(1, 0)]
    energies = {
        0: [e0],  # the absolute S0 energy
        1: [result.energies[StateLabel(1, i)] - e0 for i in range(1, n_s)],
        3: [result.energies[StateLabel(3, j)] - e0 for j in range(1, n_t + 1)],
    }
    return energies, _our_pairs(result.hamiltonian, n_s, n_t)


def _orca_tda_pairs(out: str, ours: dict[int, list[float]]) -> dict[str, float]:
    states, socme = _tda_states(out), _tda_socme(out)
    _check_s0(ours[0][0], _scf_energy(out))
    # ORCA's S = 0 is the SCF ground state (our S0); S = k and T = k are its k-th roots
    singlet = [0] + [1 + k for k in _map_by_energy(ours[1], states[1], "S", 1)]
    triplet = [1 + k for k in _map_by_energy(ours[3], states[3], "T", 1)]
    pairs = {}
    for i, s in enumerate(singlet):
        for j, t in enumerate(triplet):
            pairs[f"S{i}-T{j + 1}"] = float(np.sqrt(np.sum(np.abs(socme[t, s]) ** 2)))
    return pairs


def _orca_casscf_pairs(out: str, ours: dict[int, list[float]]) -> dict[str, float]:
    """The pair rss from the printed lines. For T_J-T_J' every printed line of the pair is
    summed: an element absent from the NONZERO list is below its print threshold (0.0005 cm-1),
    so the sum over the printed lines is the 9-element sum to print precision; T_J-T_J has no
    printed line (zero for real states), so ORCA's value is 0."""
    states, lines = _casscf_states(out), _casscf_soc_lines(out)
    s_block, s_energies = states[1]
    t_block, t_energies = states[3]
    s0 = s_energies[0]
    assert min(s_energies) == s0  # ORCA's root 0 of the singlet block is S0
    _check_s0(ours[0][0], s0)
    singlet = [0] + [
        1 + k for k in _map_by_energy(ours[1], [e - s0 for e in s_energies[1:]], "S", 1)
    ]
    triplet = _map_by_energy(ours[3], [e - s0 for e in t_energies], "T", 1)

    used: set[frozenset] = set()

    def rss(first: tuple[int, int], second: tuple[int, int]) -> float:
        """Over the printed elements between spin-free states `first` and `second` ((block,
        root)); for first == second the two components differ in Ms (the key has two members)."""
        total = 0.0
        for key, value in lines.items():
            roots = sorted((block, root) for block, root, _ in key)
            if roots == sorted([first, second]):
                total += abs(value) ** 2
                used.add(key)
        return float(np.sqrt(total))

    pairs = {}
    for i, s in enumerate(singlet):
        for j, t in enumerate(triplet):
            pairs[f"S{i}-T{j + 1}"] = rss((s_block, s), (t_block, t))
    for j, t in enumerate(triplet):
        for jb in range(j, len(triplet)):
            pairs[f"T{j + 1}-T{jb + 1}"] = rss((t_block, t), (t_block, triplet[jb]))
    unused = [sorted(key) for key in lines if key not in used]
    assert not unused, f"ORCA printed elements outside the compared pairs: {unused}"
    return pairs


@pytest.mark.parametrize("operator", list(ORCA_OPERATOR))
@pytest.mark.parametrize("method", list(SETTINGS))
@pytest.mark.parametrize("molecule", MOLECULES)
def test_spin_orbit_couplings_match_orca(molecule, method, operator, tmp_path):
    out = _run_orca(tmp_path, _orca_input(molecule, method, operator))
    energies, ours = _ours(molecule, method, operator)
    print(f"\n{molecule} {method} {operator}")
    if method == "tda":
        theirs = _orca_tda_pairs(out, energies)
    else:
        theirs = _orca_casscf_pairs(out, energies)
    assert set(theirs) <= set(ours)
    misses = []
    for pair, orca in theirs.items():
        diff = abs(ours[pair] - orca)
        allowance = max(FLOOR_CM1, FRACTION * orca)
        print(
            f"  pair {pair}: ours {ours[pair]:.4f}  ORCA {orca:.4f}  |diff| {diff:.4f}  "
            f"allowance {allowance:.4f}  ratio {diff / allowance:.3f}"
        )
        if diff > allowance:
            misses.append(f"{pair}: ours {ours[pair]:.4f}, ORCA {orca:.4f} cm-1")
    assert not misses, f"{molecule} {method} {operator}: " + "; ".join(misses)
    # non-vacuous: the largest pair is a real coupling, not a symmetry zero
    assert max(theirs.values()) > 10.0


@pytest.mark.parametrize(
    "parser, block",
    [
        (_scf_energy, "Total Energy"),
        (_tda_states, "CIS-EXCITED STATES (SINGLETS)"),
        (_tda_socme, SOCME_HEADER),
        (_casscf_states, "CASSCF RESULTS"),
        (_casscf_soc_lines, NONZERO_HEADER),
    ],
)
def test_a_parser_names_the_missing_block(parser, block):
    with pytest.raises(OrcaOutputError, match=re.escape(block)):
        parser("an ORCA output without the block\n")


def test_an_incomplete_socme_block_is_refused():
    row = "   1      0    (   0.00 ,  -63.08 )    (   0.00 ,    0.00 )    (  -0.00 ,    0.00 )"
    text = f"{SOCME_HEADER}\n  T  S\n{row}\n\n"
    with pytest.raises(OrcaOutputError, match="has 1 rows"):
        _tda_socme(text)


def test_an_unreadable_nonzero_line_is_refused():
    good = "   0    1  1.0  1.0              0     0  1.0  1.0            0.000             55.943"
    bad = "   1    0  0.0  0.0              0     0  1.0  0.0           -0.000         ********"
    text = f"{NONZERO_HEADER}\n{good}\n{bad}\n\nNote: the full matrix follows\n"
    with pytest.raises(OrcaOutputError, match="unreadable line"):
        _casscf_soc_lines(text)
    assert len(_casscf_soc_lines(f"{NONZERO_HEADER}\n{good}\n\nNote:\n")) == 1
