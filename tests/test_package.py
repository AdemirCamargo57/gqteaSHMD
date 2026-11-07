import gqteashmd


def test_version_is_defined():
    assert gqteashmd.__version__ == "0.1.0.dev0"


def test_importing_the_cli_and_worker_does_not_import_pyscf():
    import subprocess
    import sys

    code = (
        "import gqteashmd.cli, gqteashmd.worker, sys; "
        "import gqteashmd.overlaps.determinants, gqteashmd.overlaps.tracking; "
        "import gqteashmd.electronic.base, gqteashmd.electronic.factory; "
        "assert not any(m == 'pyscf' or m.startswith('pyscf.') for m in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
