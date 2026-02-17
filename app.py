import streamlit as st
import json
import sys
import os

# Add epidemic folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), "epidemic"))

from multi_type_epidemic import MultiTypeEpidemic

st.title("Multi-Type Epidemic Simulator")

config_path = os.path.join("epidemic", "config.json")

with open(config_path) as f:
    config = json.load(f)
    st.sidebar.header("Global Parameters")
    config["rates"]["beta0"] = st.sidebar.slider("beta0", 0.0, 2.0, float(config["rates"]["beta0"]))
    config["rates"]["gamma"] = st.sidebar.slider("gamma", 0.0, 1.0, float(config["rates"]["gamma"]))
    config["rates"]["sigma"] = st.sidebar.slider("sigma", 0.0, 200.0, float(config["rates"]["sigma"]))
    config["model"]["mode"] = st.sidebar.selectbox("Mode", ["mean_field", "network"])
    config["model"]["family"] = st.sidebar.selectbox("Family", ["SIR", "SIS"])
    if st.button("Run Simulation"):
        model = MultiTypeEpidemic(config)
        results = model.run()
        fig = MultiTypeEpidemic.plot_results(results)
        st.pyplot(fig)

