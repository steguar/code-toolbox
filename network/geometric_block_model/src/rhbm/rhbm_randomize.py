
import os
import sys
import time
import logging
from pathlib import Path

from rhbm.model import randomize
from rhbm.blocks import get_block_sizes
from rhbm.utils import read_matrix, save_data


def run_rhbm_randomize(
    *,
    input_graph: str,
    communities: int,
    assortativity: float,
    order_decay: float,
    output: str,
    n_runs: int = 1,
    log_to_stdout: bool = True,
):
    """
    Programmatic interface for RHBM randomization.
    Replaces CLI usage and subprocess calls.
    """

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    handler_stream = sys.stdout if log_to_stdout else sys.stderr
    handler = logging.StreamHandler(handler_stream)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    os.makedirs(output, exist_ok=True)

    start = time.time()

    randomize(
        input_graph,
        communities,
        assortativity,
        order_decay,
        output,
        n_runs,
        logger=logger
    )

    logger.info(f"Randomization completed in {time.time() - start:.2f} seconds")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-n", "--communities", type=int, required=True)
    parser.add_argument("-p", "--assortativity", type=float, required=True)
    parser.add_argument("-q", "--order_decay", type=float, required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--n_runs", type=int, default=1)

    args = parser.parse_args()

    run_rhbm_randomize(
        input_graph=args.input,
        communities=args.communities,
        assortativity=args.assortativity,
        order_decay=args.order_decay,
        output=args.output,
        n_runs=args.n_runs
    )
