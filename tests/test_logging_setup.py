import logging

from gqteashmd.logging_setup import close_logging, configure_task_logging


def test_task_logging_sends_warnings_to_stderr_once(capsys):
    try:
        configure_task_logging()
        configure_task_logging()  # idempotent: no second handler
        log = logging.getLogger("gqteashmd.electronic.pyscf_backend")
        log.info("not shown")
        log.warning("No CASSCF reference solution at the reference geometry (x)")
        err = capsys.readouterr().err
        assert err.count("No CASSCF reference solution") == 1
        assert "WARNING gqteashmd.electronic.pyscf_backend:" in err
        assert "not shown" not in err
    finally:
        close_logging()
