"""
Integrated Epidemic–Opinion Dynamics Model
===========================================

This model couples:

1. A macro SIR epidemic model (mean-field)
2. A static social network with ideological communities
3. A perceived-risk dynamics with:
   - discrete delay
   - media amplification/underestimation
4. An external opinion driver with two discrete regimes:
   mode = 0 → prudence for all
   mode = 1 → intensity-driven ideological radicalization

All modeling assumptions are documented explicitly below.
"""
import sys
import os
import shutil
import pickle
import datetime
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque


# ============================================================
# CONFIG
# ============================================================

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# GRAPH LOADING
# ============================================================

def load_graph(nodes_file, edges_file):
    """
    Nodes file format:
        node_id  community_id
    Edges file format:
        source_id  target_id

    Graph is assumed undirected.
    Communities are static.
    """

    G = nx.Graph()
    communities = {}

    with open(nodes_file, "r") as f:
        for line in f:
            node_id, com_id = line.strip().split()
            node_id = int(node_id)
            com_id = int(com_id)
            G.add_node(node_id)
            communities[node_id] = com_id

    with open(edges_file, "r") as f:
        for line in f:
            u, v = line.strip().split()
            G.add_edge(int(u), int(v))

    return G, communities


# ============================================================
# EPIDEMIC MODEL
# ============================================================

class SIRModel:
    """
    Standard mean-field SIR model.

    dS/dt = -β S I
    dI/dt = β S I - μ I
    dR/dt = μ I

    All variables are normalized so that S+I+R = 1.
    """

    def __init__(self, params):
        self.beta = params["beta"]
        self.mu = params["mu"]
        self.S = params["S0"]
        self.I = params["I0"]
        self.R = params["R0"]

    def step(self, dt):
        dS = -self.beta * self.S * self.I
        dI = self.beta * self.S * self.I - self.mu * self.I
        dR = self.mu * self.I

        self.S += dt * dS
        self.I += dt * dI
        self.R += dt * dR

        # numerical renormalization
        total = self.S + self.I + self.R
        self.S /= total
        self.I /= total
        self.R /= total

    def state(self):
        return self.S, self.I, self.R


# ============================================================
# OPINION MODEL
# ============================================================

class OpinionModel:
    """
    Opinion dynamics on static network with communities.

    Dynamics:

    do_i/dt =
        α_social * (mean_neighbor_opinion - o_i)
      + α_comm * (community_mean - o_i)
      + External_i(t)

    External term has two regimes:

    mode = 0:
        Prudence for all:
            External_i = α_R * R(t)

    mode = 1:
        Intensity-driven radicalization:
            External_i = α_R * R(t) * sign(o_i) * |o_i|^p

    Where:
        R(t) = perceived risk (see RiskModel)
    """

    def __init__(self, G, communities_raw, config):
        self.G = G
        self.nodes = list(G.nodes())
        self.N = len(self.nodes)
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}

        self.dt = config["simulation"]["dt"]
        self.rng = np.random.default_rng(config["simulation"]["seed"])

        # --- remap communities to 0..K-1
        raw = np.array([communities_raw[n] for n in self.nodes])
        unique = sorted(set(raw))
        self.map = {lab: k for k, lab in enumerate(unique)}
        self.c = np.array([self.map[x] for x in raw])
        self.K = len(unique)

        # --- initialize opinions per community
        init_cfg = config["initial_conditions"]["opinions_by_community"]
        clip_lo, clip_hi = config["initial_conditions"]["clip_opinions_to"]

        self.o = np.zeros(self.N)

        for k in range(self.K):
            mean = init_cfg[str(k)]["mean"]
            std = init_cfg[str(k)]["std"]
            idx = np.where(self.c == k)[0]
            self.o[idx] = self.rng.normal(mean, std, len(idx))

        self.o = np.clip(self.o, clip_lo, clip_hi)

        # parameters
        self.alpha_s = config["opinion_dynamics"]["alpha_social"]
        self.alpha_c = config["opinion_dynamics"]["alpha_community"]

        self.ext_cfg = config["external_effect"]

    # ----------------------------------------

    def community_stats(self):
        means = np.zeros(self.K)
        stds = np.zeros(self.K)
        for k in range(self.K):
            idx = np.where(self.c == k)[0]
            means[k] = self.o[idx].mean()
            stds[k] = self.o[idx].std()
        return means, stds

    # ----------------------------------------

    def step(self, R_value):
        """
        Perform one Euler step for opinion dynamics.
        """

        social = np.zeros(self.N)

        for node in self.nodes:
            i = self.node_to_idx[node]
            neigh = list(self.G.neighbors(node))
            if len(neigh) == 0:
                continue
            neigh_idx = [self.node_to_idx[n] for n in neigh]
            social[i] = self.o[neigh_idx].mean() - self.o[i]

        comm_means, _ = self.community_stats()
        Fcomm = comm_means[self.c] - self.o

        # ------------------------
        # External term
        # ------------------------

        mode = self.ext_cfg["mode"]
        alpha_R = self.ext_cfg["alpha_R"]
        p = self.ext_cfg.get("p", 1)

        if mode == 0:
            # Prudence for all
            external = alpha_R * R_value

        elif mode == 1:
            # Intensity-driven radicalization
            external = (
                alpha_R
                * R_value
                * np.sign(self.o)
                * (np.abs(self.o) ** p)
            )
        else:
            raise ValueError("external_effect.mode must be 0 or 1")

        # Euler update
        do_dt = self.alpha_s * social + self.alpha_c * Fcomm + external
        self.o = np.clip(self.o + self.dt * do_dt, -1, 1)

        return self.community_stats(), self.o.mean(), self.o.std()


# ============================================================
# RISK PERCEPTION MODEL
# ============================================================

class RiskModel:
    """
    Perceived risk R(t) is a delayed and filtered version of epidemic I(t).

    R evolves as:

        dR/dt = ( A(t) * I(t-Δ) - R ) / τ

    Media bias:
        A(t) =
            1 - a   if growth < threshold
            1 + a   otherwise

    Growth computed with δ = 1 step:
        g(t) = I(t) - I(t-1)
    """

    def __init__(self, config):
        self.dt = config["simulation"]["dt"]

        self.delay_steps = config["risk_perception"]["delay_steps"]
        self.tau = config["risk_perception"]["tau"]

        self.a = config["media_bias"]["a"]
        self.g_threshold = config["media_bias"]["g_threshold"]

        self.R = 0.0

        # memory buffer for delay
        self.buffer = deque(maxlen=self.delay_steps + 1)

    # ----------------------------------------

    def step(self, I_current):
        """
        Update perceived risk R(t).
        """

        self.buffer.append(I_current)

        # compute growth with δ = 1
        if len(self.buffer) >= 2:
            g = self.buffer[-1] - self.buffer[-2]
        else:
            g = 0

        # media bias
        if g < self.g_threshold:
            A = 1 - self.a
        else:
            A = 1 + self.a

        # delayed epidemic value
        if len(self.buffer) > self.delay_steps:
            I_delayed = self.buffer[0]
        else:
            I_delayed = self.buffer[-1]

        # Euler update
        dR = (A * I_delayed - self.R) / self.tau
        self.R += self.dt * dR

        return self.R


# ============================================================
# SIMULATION CONTROLLER
# ============================================================

class Simulation:

    def __init__(self, config):
        self.config = config

        G, communities = load_graph(
            config["input_files"]["nodes_file"],
            config["input_files"]["edges_file"]
        )

        self.epi = SIRModel(config["epidemic_model"]["parameters"])
        self.opinion = OpinionModel(G, communities, config)
        self.risk = RiskModel(config)

        self.n_steps = config["simulation"]["n_steps"]
        self.dt = config["simulation"]["dt"]

    # ----------------------------------------

    def run(self):

        t = np.arange(self.n_steps) * self.dt

        S_hist, I_hist, R_epi_hist = [], [], []
        R_perceived_hist = []

        o_mean_hist = []
        o_std_hist = []

        o_mean_c_hist = []
        o_std_c_hist = []

        for _ in range(self.n_steps):

            # epidemic
            self.epi.step(self.dt)
            S, I, R_epi = self.epi.state()

            # perceived risk
            R_perc = self.risk.step(I)

            # opinions
            (mean_c, std_c), mean_o, std_o = self.opinion.step(R_perc)

            # store
            S_hist.append(S)
            I_hist.append(I)
            R_epi_hist.append(R_epi)
            R_perceived_hist.append(R_perc)

            o_mean_hist.append(mean_o)
            o_std_hist.append(std_o)
            o_mean_c_hist.append(mean_c)
            o_std_c_hist.append(std_c)

        return {
            "t": t,
            "S": np.array(S_hist),
            "I": np.array(I_hist),
            "R_epi": np.array(R_epi_hist),
            "R_perceived": np.array(R_perceived_hist),
            "o_mean": np.array(o_mean_hist),
            "o_std": np.array(o_std_hist),
            "o_mean_c": np.array(o_mean_c_hist),
            "o_std_c": np.array(o_std_c_hist)
        }


# ============================================================
# PLOTTING
# ============================================================
def plot_results(results):

    t = results["t"]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    # --- Epidemic and perceived risk ---
    ax1.plot(t, results["I"], label="I(t) - Epidemic prevalence")
    ax1.plot(t, results["R_perceived"], linestyle="--",
             label="R(t) - Perceived risk")

    ax1.set_xlabel("time")
    ax1.set_ylabel("Epidemic / Perceived risk")

    # --- Opinion dynamics ---
    ax2 = ax1.twinx()

    ax2.plot(t, results["o_mean"],
             label="Global opinion mean")

    ax2.fill_between(
        t,
        results["o_mean"] - results["o_std"],
        results["o_mean"] + results["o_std"],
        alpha=0.2,
        label="Global opinion ±1 std"
    )

    # community means
    for k in range(results["o_mean_c"].shape[1]):
        ax2.plot(
            t,
            results["o_mean_c"][:, k],
            linestyle=":",
            label=f"Community {k} mean"
        )

    ax2.set_ylabel("Opinion")

    # --- Combined legend ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper center",
               fontsize=8)

    fig.tight_layout()
    return fig



if __name__ == "__main__":


    # 1. Get config path from command line arguments, default to "config.json"
    config_path = sys.argv[1] if len(sys.argv) > 1 else "opinion/config.json"

    # Load config and run simulation
    config = load_config(config_path)
    sim = Simulation(config)
    results = sim.run()

    # 2. Create a unique output folder using a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"simulation_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 3. Save a copy of the configuration file used
    shutil.copy(config_path, os.path.join(output_dir, "config_used.json"))

    # 4. Generate and save the plot in the new folder
    fig = plot_results(results)
    
    # Note: Swapped inner double quotes to single quotes to fix f-string syntax
    plot_filename = f"opinions_alphaS_{config['opinion_dynamics']['alpha_social']}_alphaC_{config['opinion_dynamics']['alpha_community']}_external_type_{config['external_effect']['mode']}_alphaR_{config['external_effect']['alpha_R']}_mediabias_a_{config['media_bias']['a']}_mediabias_gthreshold_{config['media_bias']['g_threshold']}_risk_perception_delay_{config['risk_perception']['delay_steps']}_risk_perception_tau_{config['risk_perception']['tau']}.png"
    plt.savefig(os.path.join(output_dir, plot_filename))

    # 5. Save the 'results' variable as a pickle file
    with open(os.path.join(output_dir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    print(f"Simulation complete! All files saved to the '{output_dir}' directory.")
    
    # Optional: Keep this if you still want the plot to pop up on your screen when running
    plt.show()

