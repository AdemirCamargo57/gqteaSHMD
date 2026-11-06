"""gqteashmd: automated surface-hopping molecular dynamics workflow."""

import logging

__version__ = "0.1.0.dev0"

# Without a handler, Python's last-resort handler would echo our log records to stderr.
logging.getLogger("gqteashmd").addHandler(logging.NullHandler())
