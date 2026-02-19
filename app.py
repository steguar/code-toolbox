
import os
import sys
import json
import time
import zipfile
import tempfile
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

# Add paths
sys.path.append(str(BASE_DIR / "epidemic"))
sys.path.append(str(BASE_DIR / "opinion"))
sys.path.append(str(BASE_DIR / "network" / "geometric_block_model" / "src" / "rhbm"))

from multi_type_epidemic import MultiTypeEpidemic
from opinion_dynamics import Simulation as OpinionSimulation
from rhbm_generate import run_rhbm_generate
from rhbm_randomize import run_rhbm_randomize


def zip_folder(folder_path: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in folder_path.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(folder_path)))


def ensure_tmp_dir(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"{prefix}_"))


st.set_page_config(page_title="CODE Platform", layout="wide")
st.title("CODE Interactive Platform")

module = st.sidebar.selectbox(
    "Select module",
    ["Network generation", "Network randomization", "Epidemic model", "Opinion dynamics"],
    index=0
)

# =============================================================================
# NETWORK GENERATION
# =============================================================================
if module == "Network generation":

    st.header("RHBM Generation")

    N = st.sidebar.number_input("N", 10, 500000, 1000, 10)
    avgk = st.sidebar.number_input("avgk", 1, 500, 10, 1)
    gamma = st.sidebar.slider("gamma", 2.0, 5.0, 2.5, 0.01)
    communities = st.sidebar.number_input("communities", 1, 200, 5, 1)
    assortativity = st.sidebar.slider("rho", 0.0, 1.0, 0.5, 0.01)
    order_decay = st.sidebar.slider("q", 0.1, 5.0, 1.0, 0.01)
    beta = st.sidebar.slider("beta", 0.1, 5.0, 2.0, 0.1)
    n_runs = st.sidebar.number_input("n_runs", 1, 20, 1, 1)
    n_graphs = st.sidebar.number_input("n_graphs", 1, 20, 1, 1)
    fast = st.sidebar.checkbox("fast", value=False)

    if st.button("Run generation"):
        tmp_out = ensure_tmp_dir("rhbm_gen")
        start = time.time()

        run_rhbm_generate(
            size=N,
            avgk=avgk,
            gamma=gamma,
            communities=communities,
            assortativity=assortativity,
            order_decay=order_decay,
            output=str(tmp_out),
            beta=beta,
            fast=fast,
            n_runs=n_runs,
            n_graphs=n_graphs
        )

        st.success(f"Completed in {time.time() - start:.2f}s")

        zip_path = tmp_out.with_suffix(".zip")
        zip_folder(tmp_out, zip_path)
        st.download_button("Download ZIP", zip_path.read_bytes(), zip_path.name)


# =============================================================================
# NETWORK RANDOMIZATION
# =============================================================================
elif module == "Network randomization":

    st.header("RHBM Randomization")

    input_file = st.file_uploader("Upload graph file", type=["txt", "edgelist"])
    communities = st.sidebar.number_input("communities", 1, 200, 5, 1)
    assortativity = st.sidebar.slider("rho", 0.0, 1.0, 0.5, 0.01)
    order_decay = st.sidebar.slider("q", 0.1, 5.0, 1.0, 0.01)
    n_runs = st.sidebar.number_input("n_runs", 1, 20, 1, 1)

    if st.button("Run randomization") and input_file:

        tmp_out = ensure_tmp_dir("rhbm_rand")
        input_path = tmp_out / input_file.name

        with open(input_path, "wb") as f:
            f.write(input_file.read())

        start = time.time()

        run_rhbm_randomize(
            input_graph=str(input_path),
            communities=communities,
            assortativity=assortativity,
            order_decay=order_decay,
            output=str(tmp_out),
            n_runs=n_runs
        )

        st.success(f"Completed in {time.time() - start:.2f}s")

        zip_path = tmp_out.with_suffix(".zip")
        zip_folder(tmp_out, zip_path)
        st.download_button("Download ZIP", zip_path.read_bytes(), zip_path.name)


# =============================================================================
# EPIDEMIC MODEL
# =============================================================================
elif module == "Epidemic model":

    st.header("Epidemic Model")
    st.info("Use previous implementation.")


# =============================================================================
# OPINION DYNAMICS
# =============================================================================
else:

    st.header("Opinion Dynamics")
    st.info("Use previous implementation.")
