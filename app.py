import streamlit as st
import json
import sys
import os
import subprocess

# Add project folders to path
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "epidemic"))
sys.path.append(os.path.join(BASE_DIR, "opinion"))

from multi_type_epidemic import MultiTypeEpidemic
from opinion_dynamics import Simulation as OpinionSimulation


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="CODE Interactive Platform",
    layout="wide"
)

st.title("CODE Interactive Simulation Platform")
st.markdown(
    """
This platform allows interactive exploration of:

- Community-based epidemic dynamics
- Opinion formation under media influence
- Synthetic network generation

All models are fully parameter-driven and reproducible.
"""
)

# ------------------------------------------------------------
# MODULE SELECTOR
# ------------------------------------------------------------

module = st.sidebar.selectbox(
    "Select Module",
    ["Epidemic Model", "Opinion Dynamics", "Network Generation"]
)

# ============================================================
# 1️⃣ EPIDEMIC MODEL
# ============================================================

if module == "Epidemic Model":

    st.header("Multi-Type Epidemic Model")

    with open("epidemic/config.json") as f:
        config = json.load(f)

    st.sidebar.header("Model Structure")

    st.sidebar.markdown(
        """
**Model type**
- *mean_field*: deterministic ODE approximation.
- *network*: stochastic simulation on the provided network.
        
**Family**
- *SIR*: recovered individuals gain immunity.
- *SIS*: recovered individuals return to susceptible.
"""
    )

    config["model"]["mode"] = st.sidebar.selectbox(
        "Simulation Mode",
        ["mean_field", "network"],
        index=0
    )

    config["model"]["family"] = st.sidebar.selectbox(
        "Disease Family",
        ["SIR", "SIS"],
        index=0
    )

    st.sidebar.header("Transmission Parameters")

    st.sidebar.markdown(
        """
**beta0**: Baseline transmission rate.

**gamma**: Recovery rate.

**sigma**: Rate from Exposed to Infected.
Large values approximate no incubation period.
"""
    )

    config["rates"]["beta0"] = st.sidebar.slider(
        "beta0", 0.0, 2.0, float(config["rates"]["beta0"])
    )

    config["rates"]["gamma"] = st.sidebar.slider(
        "gamma", 0.0, 1.0, float(config["rates"]["gamma"])
    )

    config["rates"]["sigma"] = st.sidebar.slider(
        "sigma", 0.0, 200.0, float(config["rates"]["sigma"])
    )

    st.sidebar.header("Behavioral Modulation")

    st.sidebar.markdown(
        """
**epsilon_media**: Strength of media influence on transmission.

**epsilon_local**: Strength of local awareness effect.

**alpha_risk**: Ideological modulation of risk perception.
"""
    )

    config["behavior_modulation"]["epsilon_media"] = st.sidebar.slider(
        "epsilon_media", 0.0, 2.0,
        float(config["behavior_modulation"]["epsilon_media"])
    )

    config["behavior_modulation"]["epsilon_local"] = st.sidebar.slider(
        "epsilon_local", 0.0, 2.0,
        float(config["behavior_modulation"]["epsilon_local"])
    )

    config["behavior_modulation"]["alpha_risk"] = st.sidebar.slider(
        "alpha_risk", 0.0, 2.0,
        float(config["behavior_modulation"]["alpha_risk"])
    )

    if st.button("Run Epidemic Simulation"):
        model = MultiTypeEpidemic(config)
        results = model.run()
        fig = MultiTypeEpidemic.plot_results(results)
        st.pyplot(fig)

# ============================================================
# 2️⃣ OPINION DYNAMICS
# ============================================================

elif module == "Opinion Dynamics":

    st.header("Opinion Dynamics Model")

    with open("opinion/config.json") as f:
        config = json.load(f)

    st.sidebar.header("Opinion Interaction Parameters")

    st.sidebar.markdown(
        """
**alpha_social**: Strength of peer-to-peer influence.

**alpha_community**: Pull toward community mean opinion.

**alpha_R**: Strength of external risk-driven influence.
"""
    )

    config["opinion_dynamics"]["alpha_social"] = st.sidebar.slider(
        "alpha_social", 0.0, 3.0,
        float(config["opinion_dynamics"]["alpha_social"])
    )

    config["opinion_dynamics"]["alpha_community"] = st.sidebar.slider(
        "alpha_community", 0.0, 3.0,
        float(config["opinion_dynamics"]["alpha_community"])
    )

    config["external_effect"]["alpha_R"] = st.sidebar.slider(
        "alpha_R", 0.0, 3.0,
        float(config["external_effect"]["alpha_R"])
    )

    st.sidebar.header("Media Perception")

    st.sidebar.markdown(
        """
**a**: Media amplification / underestimation factor.

**tau**: Time scale of risk perception.

**delay_steps**: Information delay in perception.
"""
    )

    config["media_bias"]["a"] = st.sidebar.slider(
        "media amplification (a)", 0.0, 1.0,
        float(config["media_bias"]["a"])
    )

    config["risk_perception"]["tau"] = st.sidebar.slider(
        "tau", 0.1, 20.0,
        float(config["risk_perception"]["tau"])
    )

    config["risk_perception"]["delay_steps"] = st.sidebar.slider(
        "delay_steps", 0, 100,
        int(config["risk_perception"]["delay_steps"])
    )

    if st.button("Run Opinion Simulation"):
        sim = OpinionSimulation(config)
        results = sim.run()
        fig = sim.plot_results(results)
        st.pyplot(fig)

# ============================================================
# 3️⃣ NETWORK GENERATION
# ============================================================

elif module == "Network Generation":

    st.header("Random Hyperbolic Block Model")

    st.markdown(
        """
This module generates synthetic networks or randomizes
an observed network using the Random Hyperbolic Block Model.
"""
    )

    mode = st.selectbox(
        "Generation Mode",
        ["Generate Synthetic Network", "Randomize Observed Network"]
    )

    if mode == "Generate Synthetic Network":

        st.sidebar.header("Synthetic Network Parameters")

        N = st.sidebar.number_input("Number of nodes (N)", 100, 10000, 1000)
        avgk = st.sidebar.number_input("Average degree", 2, 100, 10)
        gamma = st.sidebar.slider("Power-law exponent (gamma)", 2.0, 4.0, 2.5)
        communities = st.sidebar.number_input("Communities", 1, 10, 3)

        st.info("Network generation is currently server-side only.")

    else:

        st.sidebar.header("Randomization Parameters")

        beta = st.sidebar.slider("Beta (temperature)", 0.1, 5.0, 2.0)
        st.info("Upload-based randomization can be implemented if needed.")

    st.warning(
        "For large networks, generation may take significant computation time."
    )

