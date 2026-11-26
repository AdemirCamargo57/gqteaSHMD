"""Triplet states against SHARC 4.1 (spec 4 item 5): not comparable, recorded as a skip.

SHARC 4.1's SHARC_PYSCF.py refuses a QM.in with non-singlet states when it reads the input
(measured in Task 1 of the triplet plan, by running it; no source read), so there is nothing to
compare. The test stays in the list so that the decision is visible.
"""

import os

import pytest

pytestmark = [
    pytest.mark.sharc,
    pytest.mark.pyscf,
    pytest.mark.skipif(
        not (os.environ.get("SHARC_HOME") and os.environ.get("SHARC_PYTHON")),
        reason="needs SHARC_HOME and SHARC_PYTHON",
    ),
]


def test_triplet_energies_and_gradients_match_sharc_pyscf():
    pytest.skip(
        "SHARC 4.1 SHARC_PYSCF.py refuses non-singlet states (NotImplementedError: Not "
        "implemented for states other than singlets!), measured 2026-11-26"
    )
