import re
from importlib import metadata

import gqteashmd

# a release (0.1.0) or a development version (0.2.0.dev0)
VERSION_RE = re.compile(r"\d+\.\d+\.\d+(\.dev\d+)?")


def test_the_version_is_a_release_or_dev_version_and_matches_the_install():
    assert VERSION_RE.fullmatch(gqteashmd.__version__)
    # differs after a version change until `pip install -e . --no-deps` is run again
    assert metadata.version("gqteashmd") == gqteashmd.__version__


def test_importing_the_cli_and_worker_does_not_import_pyscf():
    import subprocess
    import sys

    code = (
        "import gqteashmd.cli, gqteashmd.worker, sys; "
        "import gqteashmd.overlaps.determinants, gqteashmd.overlaps.tracking; "
        "import gqteashmd.electronic.base, gqteashmd.electronic.factory; "
        "assert not any(m.split('.')[0] in ('pyscf', 'geometric', 'h5py') for m in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
