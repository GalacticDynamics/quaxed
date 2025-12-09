#!/usr/bin/env python
# /// script
# dependencies = [
#   "jax>=0.5.3",
#   "hatchling",
# ]
# ///
"""Orchestrate generation of all type stubs for quaxed.

This script coordinates the generation of type stubs from upstream libraries
(e.g., jax.numpy) into quaxed-aware versions. It serves as the main entry
point for stub generation and can be invoked directly or via the
``quaxed-make-stubs`` console script.

"""

import logging
import sys

# Configure logging before importing submodules
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

from .make_numpy_stub import main as make_numpy_stub_main  # noqa: E402


def main() -> int:
    """Generate all type stubs for quaxed.

    Returns
    -------
        Exit code (0 for success, non-zero for failure).

    """
    logger.info("=" * 60)
    logger.info("Generating type stubs for quaxed")
    logger.info("=" * 60)
    logger.info("")

    # Generate numpy stub
    logger.info("[1/1] Generating numpy stub...")
    exit_code = make_numpy_stub_main()

    if exit_code != 0:
        logger.error("")
        logger.error("ERROR: Failed to generate numpy stub")
        return exit_code

    logger.info("")
    logger.info("=" * 60)
    logger.info("All stubs generated successfully!")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
