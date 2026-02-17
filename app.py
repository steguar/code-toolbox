import os
import sys
import json
import time
import shutil
import zipfile
import tempfile
import subprocess
from pathlib import Path

import streamlit as st

# -----------------------------------------------------------------------------
# Path setup: keep imports working both locally and on hosting platforms.
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "epidemic"))
sys.path.append(str(BASE_DIR / "opinion"))

from multi_type_epidemic import MultiTypeEpidemic
from opinion_dynamics import Simulation as OpinionSimulation, plot_results


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def zip_folder(folder_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in folder_path.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(folder_path)))

def ensure_tmp_dir(name: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix=f"code_{name}_"))
    return d

def save_uploaded_file(uploaded, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(uploaded.getbuffer())

def describe_param(title: str, text: str):
    st.sidebar.markdown(f"**{title}**  \n{text}")

# -----------------------------------------------------------------------------
# Streamlit page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="CODE Interactive Platform", layout="wide")

st.title("CODE Interactive Simulation Platform")
st.markdown(
    """
This web app provides interactive access to the computational tools developed in the CODE project:

- **Network generation** (RHBM: synthetic generation or randomization of an observed graph)
- **Multi-type epidemic model** (mean-field and network-based simulation; SIR/SIS with optional exposed stage)
- **Opinion dynamics model** (community-structured opinion evolution driven by social interaction and external risk)

All models are configured via JSON files in the repository, but parameters can be adjusted interactively in the sidebar.
"""
)

module = st.sidebar.selectbox(
    "Select module",
    ["Network generation", "Epidemic model", "Opinion dynamics"],
    index=0
)

# Shared session state (so you can generate a network and re-use it)
if "last_network_zip" not in st.session_state:
    st.session_state.last_network_zip = None
if "last_nodes_path" not in st.session_state:
    st.session_state.last_nodes_path = None
if "last_edges_path" not in st.session_state:
    st.session_state.last_edges_path = None

# =============================================================================
# 1) NETWORK GENERATION (RHBM)
# =============================================================================
if module == "Network generation":
    st.header("Network generation (Random Hyperbolic Block Model)")

    st.markdown(
        """
Choose one of the following:
- **Generate**: create a synthetic network from scratch with community structure.
- **Randomize**: start from an observed graph (GraphML) and randomize it while preserving model constraints.

The generator writes outputs into an output folder (including a `node_list.txt` membership file).
This app packages the output folder as a **ZIP** for download.
"""
    )

    mode = st.sidebar.radio("Mode", ["Generate synthetic", "Randomize observed"], index=0)

    # Paths to scripts inside your repo
    # You said they are in: network/geometric_block_model/src/rhbm/
    gen_script = BASE_DIR / "network" / "geometric_block_model" / "src" / "rhbm" / "rhbm_generate.py"
    rnd_script = BASE_DIR / "network" / "geometric_block_model" / "src" / "rhbm" / "rhbm_randomize.py"

    if not gen_script.exists() or not rnd_script.exists():
        st.error(
            "RHBM scripts were not found in the expected location:\n"
            f"- {gen_script}\n- {rnd_script}\n\n"
            "Please verify your repository path `network/geometric_block_model/src/rhbm/`."
        )
        st.stop()

    st.sidebar.subheader("Output")
    describe_param("Output ZIP", "After generation, the produced folder is zipped and offered for download.")
    out_name = st.sidebar.text_input("Output folder name (prefix)", value="rhbm_output")

    # Common params
    st.sidebar.subheader("Common parameters")
    describe_param("beta", "Inverse temperature controlling clustering (higher beta typically increases clustering).")
    beta = st.sidebar.slider("beta", 0.1, 5.0, 2.0, 0.1)

    describe_param("n_runs", "Number of internal fitting / adjustment runs (script option `--n_runs`).")
    n_runs = st.sidebar.number_input("n_runs", min_value=1, max_value=100, value=1, step=1)

    describe_param("n_graphs", "Number of synthetic graphs to generate (script option `--n_graphs`).")
    n_graphs = st.sidebar.number_input("n_graphs", min_value=1, max_value=1000, value=10, step=1)

    fast = st.sidebar.checkbox("fast", value=False, help="Skip some adjustment steps (script option `--fast`).")

    dump_p = st.sidebar.checkbox("dump_p", value=False, help="Dump probability matrix; may ignore n_graphs (script option `--dump_p`).")

    # Mode-specific params
    cmd = None
    tmp_out = None
    uploaded_graph_path = None

    if mode == "Generate synthetic":
        st.sidebar.subheader("Synthetic generation parameters")

        describe_param("N", "Network size (number of nodes).")
        N = st.sidebar.number_input("N", min_value=10, max_value=500000, value=1000, step=10)

        describe_param("avgk", "Average degree (expected number of links per node).")
        avgk = st.sidebar.number_input("avgk", min_value=1, max_value=500, value=10, step=1)

        describe_param("gamma", "Power-law exponent of the degree distribution.")
        gamma = st.sidebar.slider("gamma", 2.0, 5.0, 2.5, 0.01)

        describe_param("communities", "Number of communities.")
        communities = st.sidebar.number_input("communities", min_value=1, max_value=200, value=5, step=1)

        describe_param("assortativity (rho)", "Controls assortativity of the mixing matrix (script option `-p/--assortativity`).")
        assortativity = st.sidebar.slider("assortativity", 0.0, 1.0, 0.5, 0.01)

        describe_param("order_decay (q)", "Controls how connectivity decays with (community) distance (script option `-q/--order_decay`).")
        order_decay = st.sidebar.slider("order_decay", 0.1, 5.0, 1.0, 0.01)

        tmp_out = ensure_tmp_dir(out_name)

        cmd = [
            sys.executable, "-u", str(gen_script),
            "-N", str(int(N)),
            "-k", str(int(avgk)),
            "-g", str(float(gamma)),
            "-n", str(int(communities)),
            "-p", str(float(assortativity)),
            "-q", str(float(order_decay)),
            "-b", str(float(beta)),
            "--n_runs", str(int(n_runs)),
            "--n_graphs", str(int(n_graphs)),
            "-o", str(tmp_out),
        ]
        if fast:
            cmd.append("-f")
        if dump_p:
            cmd.append("--dump_p")

    else:
        st.sidebar.subheader("Randomization parameters")

        describe_param("input graph (GraphML)", "Upload an observed graph in GraphML format.")
        uploaded = st.sidebar.file_uploader("Upload GraphML", type=["graphml"])

        describe_param("membership_attribute", "Node attribute that stores community membership in the input graph.")
        membership_attr = st.sidebar.text_input("membership_attribute", value="community")

        describe_param("communities (optional)", "Alternative to membership_attribute: explicitly set number of communities.")
        comm_override = st.sidebar.text_input("communities (optional)", value="")

        if uploaded is None:
            st.info("Upload a GraphML file to enable randomization.")
        else:
            tmp_dir = ensure_tmp_dir("observed_graph")
            uploaded_graph_path = tmp_dir / "input.graphml"
            save_uploaded_file(uploaded, uploaded_graph_path)

        tmp_out = ensure_tmp_dir(out_name)

        # Build command only if we have an uploaded graph
        if uploaded_graph_path is not None:
            cmd = [
                sys.executable, str(rnd_script),
                "-i", str(uploaded_graph_path),
                "-o", str(tmp_out),
                "-b", str(float(beta)),
                "--n_runs", str(int(n_runs)),
                "--n_graphs", str(int(n_graphs)),
            ]
            # Two mutually exclusive ways (script supports both; we let user choose)
            if comm_override.strip():
                cmd += ["-n", str(int(comm_override.strip()))]
            else:
                cmd += ["-m", membership_attr]
            if fast:
                cmd.append("-f")
            if dump_p:
                cmd.append("--dump_p")

    st.subheader("Run")
    col1, col2 = st.columns([1, 1])

    with col1:
        run = st.button("Run network tool")

    with col2:
        use_last = st.button("Use last generated network (if any)")

    if use_last:
        if st.session_state.last_network_zip is None:
            st.warning("No generated network is available in this session yet.")
        else:
            st.success("Last generated network is already stored in this session. You can download it below.")

    if run:
        if cmd is None:
            st.error("Missing inputs to build the command (e.g., GraphML not uploaded).")
        else:
            st.code(" ".join(cmd), language="bash")
            start = time.time()
            try:
                st.write("Executable:", sys.executable)
                completed = subprocess.run(
                    cmd,
                    cwd=str(BASE_DIR),
                    capture_output=True,
                    text=True,
                    # check=True
                )
                elapsed = time.time() - start
                st.success(f"Done in {elapsed:.2f}s")

                # Show logs
                if completed.stdout.strip():
                    st.text_area("stdout", completed.stdout, height=200)
                if completed.stderr.strip():
                    st.text_area("stderr", completed.stderr, height=200)

                # Zip output
                zip_path = tmp_out.with_suffix(".zip")
                zip_folder(tmp_out, zip_path)

                st.session_state.last_network_zip = zip_path
                # Best-effort: detect nodes/edges files commonly produced
                # RHBM scripts dump node_list.txt; edges may be in output depending on library.
                node_list = tmp_out / "node_list.txt"
                if node_list.exists():
                    st.session_state.last_nodes_path = node_list
                else:
                    st.session_state.last_nodes_path = None

                # Try to find an edge list file
                edge_candidates = list(tmp_out.glob("*.txt")) + list(tmp_out.glob("*.csv")) + list(tmp_out.glob("*.edges"))
                edges_guess = None
                for p in edge_candidates:
                    if "edge" in p.name.lower():
                        edges_guess = p
                        break
                st.session_state.last_edges_path = edges_guess

            except subprocess.CalledProcessError as e:
                st.error("Network tool failed.")
                st.text_area("stdout", e.stdout or "", height=200)
                st.text_area("stderr", e.stderr or "", height=200)

    # Download section
    st.subheader("Download outputs")
    if st.session_state.last_network_zip and Path(st.session_state.last_network_zip).exists():
        zip_bytes = Path(st.session_state.last_network_zip).read_bytes()
        st.download_button(
            "Download last output ZIP",
            data=zip_bytes,
            file_name=Path(st.session_state.last_network_zip).name,
            mime="application/zip"
        )
        if st.session_state.last_nodes_path:
            st.caption(f"Detected membership file: {Path(st.session_state.last_nodes_path).name}")
        if st.session_state.last_edges_path:
            st.caption(f"Detected edge-like file: {Path(st.session_state.last_edges_path).name}")
        else:
            st.caption("Edge list file not auto-detected in the output folder (it may be inside subfolders).")
    else:
        st.info("Run the tool to generate outputs, then download the ZIP here.")

# =============================================================================
# 2) EPIDEMIC MODEL
# =============================================================================
elif module == "Epidemic model":
    st.header("Multi-type epidemic model (SIR / SIS with optional exposed stage)")

    cfg_path = BASE_DIR / "epidemic" / "config.json"
    config = load_json(cfg_path)

    st.sidebar.header("Simulation set-up")
    describe_param("Mode",
                   "mean_field = deterministic ODE approximation; "
                   "network = stochastic simulation on an undirected contact network.")
    config["model"]["mode"] = st.sidebar.selectbox("mode", ["mean_field", "network"], index=0)

    describe_param("Family",
                   "SIR = immunity after recovery; SIS = recovered return to susceptible.")
    config["model"]["family"] = st.sidebar.selectbox("family", ["SIR", "SIS"], index=0)

    describe_param("use_exposed",
                   "If enabled, the model includes an Exposed (E) compartment. "
                   "Set sigma very large to approximate instantaneous incubation (SIR/SIS).")
    config["model"]["use_exposed"] = st.sidebar.checkbox("use_exposed", value=bool(config["model"].get("use_exposed", True)))

    st.sidebar.header("Rates (all in config, editable here)")
    describe_param("beta0", "Baseline transmission rate (before any modulation).")
    config["rates"]["beta0"] = st.sidebar.slider("beta0", 0.0, 2.0, float(config["rates"]["beta0"]), 0.001)

    describe_param("gamma", "Recovery rate (I→R for SIR; I→S for SIS).")
    config["rates"]["gamma"] = st.sidebar.slider("gamma", 0.0, 1.0, float(config["rates"]["gamma"]), 0.001)

    describe_param("sigma", "Incubation completion rate (E→I). Use a large value to approximate no incubation.")
    config["rates"]["sigma"] = st.sidebar.slider("sigma", 0.0, 500.0, float(config["rates"]["sigma"]), 0.5)

    st.sidebar.header("Media / awareness modulation")
    describe_param("epsilon_media", "Strength of media intensity M(t) in reducing effective transmission.")
    config["behavior_modulation"]["epsilon_media"] = st.sidebar.slider(
        "epsilon_media", 0.0, 2.0, float(config["behavior_modulation"]["epsilon_media"]), 0.01
    )
    describe_param("epsilon_local", "Strength of local awareness (prevalence) feedback.")
    config["behavior_modulation"]["epsilon_local"] = st.sidebar.slider(
        "epsilon_local", 0.0, 2.0, float(config["behavior_modulation"]["epsilon_local"]), 0.01
    )
    describe_param("alpha_risk", "Ideological modulation factor based on theta_i (community risk ideology).")
    config["behavior_modulation"]["alpha_risk"] = st.sidebar.slider(
        "alpha_risk", 0.0, 2.0, float(config["behavior_modulation"]["alpha_risk"]), 0.01
    )

    st.sidebar.header("Media intensity M(t)")
    describe_param("delay_steps", "Discrete delay (in time steps) between true prevalence and media response.")
    config["media_intensity"]["delay_steps"] = st.sidebar.slider(
        "delay_steps", 0, 300, int(config["media_intensity"]["delay_steps"]), 1
    )
    describe_param("tau", "Time constant for the exponential smoothing of media intensity.")
    config["media_intensity"]["tau"] = st.sidebar.slider(
        "tau", 0.1, 50.0, float(config["media_intensity"]["tau"]), 0.1
    )
    describe_param("a", "Media bias amplitude: underestimation uses (1-a), amplification uses (1+a).")
    config["media_intensity"]["a"] = st.sidebar.slider(
        "a", 0.0, 1.0, float(config["media_intensity"]["a"]), 0.01
    )
    describe_param("g_threshold", "Growth threshold on I_total(t)-I_total(t-1) to switch under/over-estimation.")
    config["media_intensity"]["g_threshold"] = st.sidebar.slider(
        "g_threshold", 0.0, 0.02, float(config["media_intensity"]["g_threshold"]), 0.0001
    )

    st.sidebar.header("Run control")
    describe_param("dt", "Simulation time step.")
    config["simulation"]["dt"] = st.sidebar.slider("dt", 0.01, 1.0, float(config["simulation"]["dt"]), 0.01)

    describe_param("n_steps", "Number of simulation steps.")
    config["simulation"]["n_steps"] = st.sidebar.slider("n_steps", 50, 10000, int(config["simulation"]["n_steps"]), 10)

    describe_param("seed", "Random seed (used in network mode and for initial seeding).")
    config["simulation"]["seed"] = st.sidebar.number_input("seed", min_value=0, max_value=10_000_000, value=int(config["simulation"]["seed"]), step=1)

    st.sidebar.header("Network input")
    st.sidebar.markdown("Provide a contact network (nodes + edges). You can upload files or reuse the last generated network ZIP outputs if compatible.")

    uploaded_nodes = st.sidebar.file_uploader("Upload nodes.txt (node_id com_id)", type=["txt"])
    uploaded_edges = st.sidebar.file_uploader("Upload edges.txt (source_id target_id)", type=["txt"])

    use_generated = st.sidebar.checkbox("Use last generated membership file (node_list.txt) if available", value=False)

    tmp_input = None
    if uploaded_nodes and uploaded_edges:
        tmp_input = ensure_tmp_dir("epi_input")
        nodes_path = tmp_input / "nodes.txt"
        edges_path = tmp_input / "edges.txt"
        save_uploaded_file(uploaded_nodes, nodes_path)
        save_uploaded_file(uploaded_edges, edges_path)
        config["input_files"]["nodes_file"] = str(nodes_path)
        config["input_files"]["edges_file"] = str(edges_path)
        st.sidebar.success("Uploaded network will be used.")
    elif use_generated and st.session_state.last_nodes_path and st.session_state.last_edges_path:
        config["input_files"]["nodes_file"] = str(st.session_state.last_nodes_path)
        config["input_files"]["edges_file"] = str(st.session_state.last_edges_path)
        st.sidebar.success("Using last generated network files (best-effort).")
    else:
        st.sidebar.info("Using paths defined in epidemic/config.json (server-side).")

    if st.button("Run epidemic simulation"):
        model = MultiTypeEpidemic(config)
        results = model.run()
        fig = MultiTypeEpidemic.plot_results(results)
        st.pyplot(fig, clear_figure=True)

        # quick table
        st.subheader("Final infected fractions by community")
        final_I = results["I_comm"][-1, :]
        st.write({f"community_{i}": float(final_I[i]) for i in range(len(final_I))})

# =============================================================================
# 3) OPINION DYNAMICS
# =============================================================================
else:
    st.header("Opinion dynamics model")

    cfg_path = BASE_DIR / "opinion" / "config.json"
    config = load_json(cfg_path)

    st.sidebar.header("Opinion dynamics parameters")
    describe_param("alpha_social", "Strength of social influence (neighbor interactions).")
    config["opinion_dynamics"]["alpha_social"] = st.sidebar.slider(
        "alpha_social", 0.0, 3.0, float(config["opinion_dynamics"]["alpha_social"]), 0.01
    )

    describe_param("alpha_community", "Strength of attraction to the community baseline opinion.")
    config["opinion_dynamics"]["alpha_community"] = st.sidebar.slider(
        "alpha_community", 0.0, 3.0, float(config["opinion_dynamics"]["alpha_community"]), 0.01
    )

    st.sidebar.header("External risk effect")
    describe_param("alpha_R", "Strength of the external field driven by perceived risk R(t).")
    config["external_effect"]["alpha_R"] = st.sidebar.slider(
        "alpha_R", 0.0, 3.0, float(config["external_effect"]["alpha_R"]), 0.01
    )

    describe_param("mode (prudence vs radicalization)",
                   "Discrete switch: 0 = prudence for all; 1 = radicalization (strength driven by opinion magnitude).")
    config["external_effect"]["mode"] = st.sidebar.selectbox(
        "mode", [0, 1], index=int(config["external_effect"].get("mode", 0))
    )

    describe_param("p", "Exponent controlling nonlinearity of radicalization response (used when mode=1).")
    config["external_effect"]["p"] = st.sidebar.slider(
        "p", 1, 7, int(config["external_effect"]["p"]), 1
    )

    st.sidebar.header("Media intensity / perceived risk")
    describe_param("a", "Media bias amplitude: underestimation (1-a) vs amplification (1+a).")
    config["media_bias"]["a"] = st.sidebar.slider(
        "a", 0.0, 1.0, float(config["media_bias"]["a"]), 0.01
    )

    describe_param("g_threshold", "Growth threshold to switch under/over-estimation in media.")
    config["media_bias"]["g_threshold"] = st.sidebar.slider(
        "g_threshold", 0.0, 0.02, float(config["media_bias"]["g_threshold"]), 0.0001
    )

    describe_param("delay_steps", "Discrete delay (in steps) in risk perception.")
    config["risk_perception"]["delay_steps"] = st.sidebar.slider(
        "delay_steps", 0, 300, int(config["risk_perception"]["delay_steps"]), 1
    )

    describe_param("tau", "Time constant for risk perception smoothing.")
    config["risk_perception"]["tau"] = st.sidebar.slider(
        "tau", 0.1, 50.0, float(config["risk_perception"]["tau"]), 0.1
    )

    st.sidebar.header("Network input")
    uploaded_nodes = st.sidebar.file_uploader("Upload nodes.txt (node_id com_id)", type=["txt"], key="op_nodes")
    uploaded_edges = st.sidebar.file_uploader("Upload edges.txt (source_id target_id)", type=["txt"], key="op_edges")

    tmp_input = None
    if uploaded_nodes and uploaded_edges:
        tmp_input = ensure_tmp_dir("op_input")
        nodes_path = tmp_input / "nodes.txt"
        edges_path = tmp_input / "edges.txt"
        save_uploaded_file(uploaded_nodes, nodes_path)
        save_uploaded_file(uploaded_edges, edges_path)
        config["input_files"]["nodes_file"] = str(nodes_path)
        config["input_files"]["edges_file"] = str(edges_path)
        st.sidebar.success("Uploaded network will be used.")
    else:
        st.sidebar.info("Using paths defined in opinion/config.json (server-side).")

    if st.button("Run opinion simulation"):
        sim = OpinionSimulation(config)
        results = sim.run()
        fig = plot_results(results)
        st.pyplot(fig, clear_figure=True)
