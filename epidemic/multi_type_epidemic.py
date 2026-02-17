"""
multi_type_epidemic.py
======================

Implements a "multi-type SIR" epidemic model (renamed from HeSIR in your request),
with a generic exposed stage E included (SEIR/SEIS), and two simulation backends:

A) Mean-field (ODE-like Euler discretization)
B) Network-based stochastic simulation on an undirected graph read from files

Key modeling choices (documented for reproducibility)
-----------------------------------------------------

1) Communities / types:
   - Each node belongs to a community (type) read from nodes.txt: "node_id com_id".
   - Communities are static.
   - In mean-field, each community i has compartment fractions S_i, E_i, I_i, R_i.

2) Transmission intensity:
   We use a community-specific effective transmission rate beta_i(t) that includes
   three multiplicative modulation factors:

      beta_i(t) = beta0 * f_media_i(t) * f_local_i(t) * g(theta_i)

   where:
   - beta0: baseline transmissibility (global rate in config)
   - f_media_i(t): reduction due to media intensity M(t), scaled by phi_i and epsilon_media
   - f_local_i(t): reduction due to local epidemic prevalence, scaled by psi_i and epsilon_local
   - g(theta_i): risk/ideology factor, controlled by theta_i and alpha_risk

   IMPORTANT: These forms match the spirit of §8.3 (media + awareness + ideology modulation),
   while keeping the code modular and usable as a standalone epidemic module.

3) Media intensity M(t):
   We reuse exactly the same structure you used in the other model:
   - Compute total prevalence I_total(t)
   - Compute growth g = I_total(t) - I_total(t-1)  (delta fixed to 1)
   - Media bias A(t) is a 2-level function:
        A(t)=1-a  if growth < g_threshold  (underestimation, even for slow growth)
        A(t)=1+a  otherwise                (amplification)
   - Media intensity M(t) is a delayed + filtered version of I_total:
        dM/dt = ( A(t) * I_total(t-delay) - M ) / tau

4) Force of infection:
   - Mean-field:
        lambda_i(t) = beta_i(t) * [ h_i * I_i + (1-h_i) * sum_{j!=i} w_ij * I_j ]
     with w_ij estimated from the input graph edges between communities (normalized per i over j!=i).
     NOTE: I_i here is community prevalence fraction (I_i in [0,1]).

   - Network-based:
     We simulate individual node states on the provided graph.
     For a susceptible node u in community i, we use neighbor infection pressure:
        pressure(u) = (# infected neighbors of u) / max(deg(u), 1)
     and infection hazard:
        lambda_u(t) = beta_i(t) * pressure(u)
     Then infection probability over dt:
        p_inf = 1 - exp(-lambda_u dt)

     This makes the network mode genuinely depend on the graph topology.
     The same beta_i(t) modulation is used, computed from community-level prevalences.

5) Disease families and exposed stage:
   - We always keep compartments S, E, I, R in code.
   - family = "SIR": recovery is I -> R at rate gamma
   - family = "SIS": recovery is I -> S at rate gamma (R stays 0)
   - E stage is included if use_exposed=true, with E -> I at rate sigma.
     Setting sigma very large approximates instantaneous incubation (i.e., no exposed stage),
     making SEIR ~ SIR and SEIS ~ SIS.

Outputs
-------
- Time series of I_total(t)
- Time series of I_i(t) for each community i
- Also stores S,E,I,R per community for mean-field (and counts for network mode)
- A final plot: I_total and I_i curves
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ============================================================
# Utilities: config and graph loading
# ============================================================

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_graph(nodes_file: str, edges_file: str) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    nodes_file (no header):
        node_id com_id

    edges_file (no header):
        source_id target_id

    Returns:
      - undirected NetworkX graph G with node ids as integers
      - mapping node_id -> com_id (raw labels)
    """
    G = nx.Graph()
    node_to_com = {}

    with open(nodes_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            nid, cid = line.strip().split()
            nid = int(nid)
            cid = int(cid)
            G.add_node(nid)
            node_to_com[nid] = cid

    with open(edges_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            u, v = line.strip().split()
            G.add_edge(int(u), int(v))

    return G, node_to_com


def remap_communities(node_to_com: Dict[int, int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Remap arbitrary community labels to 0..K-1.
    Returns:
      - com_map: raw_label -> new_label
      - node_to_type: node -> new_label
    """
    raw_labels = sorted(set(node_to_com.values()))
    com_map = {lab: i for i, lab in enumerate(raw_labels)}
    node_to_type = {n: com_map[c] for n, c in node_to_com.items()}
    return com_map, node_to_type


def compute_mixing_matrix_from_graph(G: nx.Graph, node_to_type: Dict[int, int], K: int) -> np.ndarray:
    """
    Estimate w_ij from the network: count edges between communities.

    For mean-field we need weights over j != i:
      sum_{j != i} w_ij = 1  (if i has any external edges)
    If a community has no external edges, we set uniform over other communities.

    Returns:
      W (KxK) with W[i,i] = 0 and each row i normalized over j!=i.
    """
    counts = np.zeros((K, K), dtype=float)
    for u, v in G.edges():
        i = node_to_type[u]
        j = node_to_type[v]
        if i != j:
            counts[i, j] += 1.0
            counts[j, i] += 1.0

    W = np.zeros((K, K), dtype=float)
    for i in range(K):
        row = counts[i].copy()
        row[i] = 0.0
        s = row.sum()
        if s > 0:
            W[i] = row / s
        else:
            # no external edges: fallback uniform over j != i
            if K > 1:
                W[i] = 1.0 / (K - 1)
                W[i, i] = 0.0
            else:
                W[i, i] = 0.0
    return W


# ============================================================
# Media intensity module (same structure as before)
# ============================================================

@dataclass
class MediaIntensityParams:
    delay_steps: int
    tau: float
    a: float
    g_threshold: float
    dt: float


class MediaIntensity:
    """
    Implements the "media intensity" signal M(t) using:
      - growth threshold on I_total(t) - I_total(t-1)
      - under/over bias A(t) = 1-a or 1+a
      - delayed + filtered target: A(t) * I_total(t-delay)
      - first-order lag with time constant tau

    This is identical in structure to what you already used.
    """

    def __init__(self, params: MediaIntensityParams):
        self.p = params
        self.M = 0.0
        self.buffer = deque(maxlen=self.p.delay_steps + 1)

    def step(self, I_total: float) -> float:
        self.buffer.append(I_total)

        # delta fixed to 1 step
        if len(self.buffer) >= 2:
            growth = self.buffer[-1] - self.buffer[-2]
        else:
            growth = 0.0

        # bias rule: underestimation if growth is below threshold (including slow growth)
        if growth < self.p.g_threshold:
            A = 1.0 - self.p.a
        else:
            A = 1.0 + self.p.a

        # delayed target
        if len(self.buffer) > self.p.delay_steps:
            I_delayed = self.buffer[0]
        else:
            I_delayed = self.buffer[-1]

        target = A * I_delayed
        dM = (target - self.M) / self.p.tau
        self.M += self.p.dt * dM
        return self.M


# ============================================================
# MultiType epidemic model
# ============================================================

class MultiTypeEpidemic:
    """
    Multi-type epidemic model with:
      - families: SIR or SIS
      - optional exposed stage E (SEIR/SEIS)
      - mean-field or network simulation

    Config-driven: all rates in config.

    Public API:
      - run() -> results dict
      - plot(results) -> matplotlib figure
    """

    def __init__(self, config: dict):
        self.cfg = config
        sim = config["simulation"]
        self.dt = float(sim["dt"])
        self.n_steps = int(sim["n_steps"])
        self.seed = int(sim.get("seed", 0))
        self.rng = np.random.default_rng(self.seed)

        model_cfg = config["model"]
        self.mode = model_cfg["mode"]          # "mean_field" or "network"
        self.family = model_cfg["family"]      # "SIR" or "SIS"
        self.use_exposed = bool(model_cfg.get("use_exposed", True))

        rates = config["rates"]
        self.beta0 = float(rates["beta0"])
        self.gamma = float(rates["gamma"])
        self.sigma = float(rates["sigma"]) if self.use_exposed else 0.0

        mod = config["behavior_modulation"]
        self.eps_media = float(mod["epsilon_media"])
        self.eps_local = float(mod["epsilon_local"])
        self.alpha_risk = float(mod["alpha_risk"])

        # Load graph and communities
        files = config["input_files"]
        self.G, node_to_com_raw = load_graph(files["nodes_file"], files["edges_file"])
        self.com_map, self.node_to_type = remap_communities(node_to_com_raw)

        self.nodes = list(self.G.nodes())
        self.N = len(self.nodes)
        if self.N == 0:
            raise ValueError("Graph has no nodes. Check nodes_file.")

        # Types / communities
        self.K = len(set(self.node_to_type.values()))
        self.type_to_nodes = [[] for _ in range(self.K)]
        for n in self.nodes:
            self.type_to_nodes[self.node_to_type[n]].append(n)

        self.Ni = np.array([len(self.type_to_nodes[i]) for i in range(self.K)], dtype=int)
        if np.any(self.Ni == 0):
            # Allowed, but mean-field formulas become tricky; we guard in stats computations.
            pass

        # Community parameters
        # Expected keys per community i:
        # h, phi, psi, theta, I0, E0
        cpars = config["community_parameters"]
        self.h = np.zeros(self.K, dtype=float)
        self.phi = np.zeros(self.K, dtype=float)
        self.psi = np.zeros(self.K, dtype=float)
        self.theta = np.zeros(self.K, dtype=float)
        self.I0 = np.zeros(self.K, dtype=float)
        self.E0 = np.zeros(self.K, dtype=float)

        for i in range(self.K):
            d = cpars.get(str(i))
            if d is None:
                raise ValueError(f"Missing community_parameters['{i}'] in config.")
            self.h[i] = float(d["h"])
            self.phi[i] = float(d["phi"])
            self.psi[i] = float(d["psi"])
            self.theta[i] = float(d["theta"])
            self.I0[i] = float(d.get("I0", 0.0))
            self.E0[i] = float(d.get("E0", 0.0))

        # Mixing matrix for mean-field
        self.W = compute_mixing_matrix_from_graph(self.G, self.node_to_type, self.K)

        # Media intensity (same structure as before)
        mi = config["media_intensity"]
        self.media = MediaIntensity(MediaIntensityParams(
            delay_steps=int(mi["delay_steps"]),
            tau=float(mi["tau"]),
            a=float(mi["a"]),
            g_threshold=float(mi["g_threshold"]),
            dt=self.dt
        ))

        # Initialize state for mean-field and network
        self._init_state()

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------

    def _init_state(self):
        """
        Initialize compartments.

        Mean-field: fractions per community.
        Network: discrete states per node.
        """
        # Mean-field fractions
        self.S = np.zeros(self.K, dtype=float)
        self.E = np.zeros(self.K, dtype=float)
        self.I = np.zeros(self.K, dtype=float)
        self.R = np.zeros(self.K, dtype=float)

        for i in range(self.K):
            i0 = self.I0[i]
            e0 = self.E0[i] if self.use_exposed else 0.0
            # Keep everything normalized in [0,1] per community
            s0 = max(0.0, 1.0 - i0 - e0)
            self.S[i], self.E[i], self.I[i], self.R[i] = s0, e0, i0, 0.0

        # Network discrete states
        # 0=S, 1=E, 2=I, 3=R
        self.node_index = {n: idx for idx, n in enumerate(self.nodes)}
        self.type_of_idx = np.array([self.node_to_type[n] for n in self.nodes], dtype=int)

        self.state = np.zeros(self.N, dtype=np.int8)  # start all S
        if self.use_exposed:
            # seed E first (optional)
            pass

        # Seed initial infected per community by sampling nodes
        for i in range(self.K):
            nodes_i = self.type_to_nodes[i]
            if not nodes_i:
                continue
            n_i = len(nodes_i)
            n_inf = int(round(self.I0[i] * n_i))
            if n_inf <= 0 and self.I0[i] > 0:
                n_inf = 1  # ensure at least one if nonzero fraction requested
            n_inf = min(n_inf, n_i)
            if n_inf > 0:
                chosen = self.rng.choice(nodes_i, size=n_inf, replace=False)
                for n in chosen:
                    self.state[self.node_index[n]] = 2  # I

            if self.use_exposed and self.E0[i] > 0:
                n_exp = int(round(self.E0[i] * n_i))
                n_exp = min(n_exp, n_i - n_inf)
                if n_exp > 0:
                    remaining = [n for n in nodes_i if self.state[self.node_index[n]] == 0]
                    chosen_e = self.rng.choice(remaining, size=n_exp, replace=False)
                    for n in chosen_e:
                        self.state[self.node_index[n]] = 1  # E

    # --------------------------------------------------------
    # Shared helpers
    # --------------------------------------------------------

    def _I_total_from_fractions(self) -> float:
        if self.N == 0:
            return 0.0
        # weighted average by community sizes
        return float(np.sum(self.I * self.Ni) / max(1, np.sum(self.Ni)))

    def _community_prevalences_from_network(self) -> np.ndarray:
        """
        Returns I_i fractions computed from node states.
        """
        I_counts = np.zeros(self.K, dtype=float)
        for i in range(self.K):
            if self.Ni[i] == 0:
                I_counts[i] = 0.0
                continue
            idxs = [self.node_index[n] for n in self.type_to_nodes[i]]
            I_counts[i] = float(np.sum(self.state[idxs] == 2)) / float(self.Ni[i])
        return I_counts

    def _beta_i(self, M: float, local_prev: np.ndarray) -> np.ndarray:
        """
        Compute beta_i(t) for each community i:
          beta_i = beta0 * f_media_i * f_local_i * g(theta_i)

        f_media_i = 1 - phi_i * M * epsilon_media
        f_local_i = 1 - psi_i * local_prev_i * epsilon_local
        g(theta_i) = 1 + alpha_risk * (2*theta_i - 1)

        Clipped below at 0 to avoid negative transmission.
        """
        f_media = 1.0 - self.phi * M * self.eps_media
        f_local = 1.0 - self.psi * local_prev * self.eps_local
        g_theta = 1.0 + self.alpha_risk * (2.0 * self.theta - 1.0)

        beta = self.beta0 * f_media * f_local * g_theta
        return np.clip(beta, 0.0, None)

    # --------------------------------------------------------
    # Mean-field step
    # --------------------------------------------------------

    def step_mean_field(self):
        """
        One Euler step of the mean-field (deterministic) system.

        Force of infection:
          mix_i = h_i * I_i + (1-h_i) * sum_{j!=i} W_ij I_j
          lambda_i = beta_i * mix_i
        """
        I_total = self._I_total_from_fractions()
        M = self.media.step(I_total)

        # cross mixing term per i
        I_vec = self.I.copy()
        cross = self.W @ I_vec  # since W[i,i]=0
        mix = self.h * I_vec + (1.0 - self.h) * cross

        beta_i = self._beta_i(M, local_prev=mix)  # local_prev uses same mix proxy
        lam = beta_i * mix

        # Compartments dynamics (SEIR/SEIS with sigma; sigma large approximates SIR/SIS)
        if self.use_exposed:
            dS = -lam * self.S
            dE = lam * self.S - self.sigma * self.E
            dI = self.sigma * self.E - self.gamma * self.I
        else:
            # direct infection S->I
            dS = -lam * self.S
            dE = 0.0 * self.E
            dI = lam * self.S - self.gamma * self.I

        if self.family.upper() == "SIR":
            dR = self.gamma * self.I
            self.S += self.dt * dS
            self.E += self.dt * dE
            self.I += self.dt * dI
            self.R += self.dt * dR
        elif self.family.upper() == "SIS":
            # recovery returns to S; R stays 0
            self.S += self.dt * (dS + self.gamma * self.I)
            self.E += self.dt * dE
            self.I += self.dt * dI
            self.R[:] = 0.0
        else:
            raise ValueError("model.family must be 'SIR' or 'SIS'")

        # Clip to [0,1] to limit numerical drift (small dt recommended)
        self.S = np.clip(self.S, 0.0, 1.0)
        self.E = np.clip(self.E, 0.0, 1.0)
        self.I = np.clip(self.I, 0.0, 1.0)
        if self.family.upper() == "SIR":
            self.R = np.clip(self.R, 0.0, 1.0)

        # Optional renormalization per community (helps with very large sigma)
        if self.family.upper() == "SIR":
            total = self.S + self.E + self.I + self.R
        else:
            total = self.S + self.E + self.I
        total = np.where(total > 0, total, 1.0)
        self.S /= total
        self.E /= total
        self.I /= total
        if self.family.upper() == "SIR":
            self.R /= total

    # --------------------------------------------------------
    # Network step
    # --------------------------------------------------------

    def step_network(self):
        """
        One stochastic step on the graph.

        For each time step:
        - Compute community-level prevalence I_i (from node states)
        - Compute total I_total and update media intensity M(t)
        - Compute beta_i(t) for each community i using local_prev = I_i (or a proxy)
        - For each node:
            if S: compute infected neighbor fraction -> hazard -> possibly S->E (or S->I if no exposed)
            if E: E->I with prob 1-exp(-sigma dt)
            if I: recover with prob 1-exp(-gamma dt) to R (SIR) or S (SIS)
            if R: stays R

        This makes the network simulation genuinely depend on topology via infected neighbors.
        """
        I_frac = self._community_prevalences_from_network()
        I_total = float(np.sum(I_frac * self.Ni) / max(1, np.sum(self.Ni)))
        M = self.media.step(I_total)

        # For beta_i local_prev, we use I_frac directly (community prevalence),
        # which is a simple and interpretable "local awareness" proxy.
        beta_i = self._beta_i(M, local_prev=I_frac)

        # Precompute infection indicator for neighbors
        infected = (self.state == 2)

        # Transition probabilities for E->I and I->(R or S)
        p_EI = 1.0 - math.exp(-self.sigma * self.dt) if self.use_exposed else 1.0
        p_rec = 1.0 - math.exp(-self.gamma * self.dt)

        # We'll update into a copy (synchronous update)
        new_state = self.state.copy()

        # Iterate nodes (O(E) via neighbor scans overall; ok for medium graphs)
        for idx, node in enumerate(self.nodes):
            st = self.state[idx]
            typ = self.type_of_idx[idx]

            if st == 0:  # S
                deg = self.G.degree(node)
                if deg <= 0:
                    continue
                inf_neigh = 0
                for nb in self.G.neighbors(node):
                    j = self.node_index[nb]
                    if infected[j]:
                        inf_neigh += 1
                pressure = inf_neigh / deg
                lam = beta_i[typ] * pressure
                p_inf = 1.0 - math.exp(-lam * self.dt)

                if self.rng.random() < p_inf:
                    if self.use_exposed:
                        new_state[idx] = 1  # E
                    else:
                        new_state[idx] = 2  # I

            elif st == 1:  # E
                # E->I
                if self.rng.random() < p_EI:
                    new_state[idx] = 2

            elif st == 2:  # I
                # recovery
                if self.rng.random() < p_rec:
                    if self.family.upper() == "SIR":
                        new_state[idx] = 3  # R
                    else:  # SIS
                        new_state[idx] = 0  # S

            else:
                # R stays R for SIR; for SIS we keep R unused
                pass

        self.state = new_state

        # Keep mean-field fractions in sync for reporting convenience (optional)
        # We compute S/E/I/R fractions per community from node states.
        for i in range(self.K):
            if self.Ni[i] == 0:
                self.S[i] = self.E[i] = self.I[i] = self.R[i] = 0.0
                continue
            idxs = [self.node_index[n] for n in self.type_to_nodes[i]]
            s = np.mean(self.state[idxs] == 0)
            e = np.mean(self.state[idxs] == 1)
            inf = np.mean(self.state[idxs] == 2)
            r = np.mean(self.state[idxs] == 3) if self.family.upper() == "SIR" else 0.0
            self.S[i], self.E[i], self.I[i], self.R[i] = float(s), float(e), float(inf), float(r)

    # --------------------------------------------------------
    # Run + results
    # --------------------------------------------------------

    def run(self) -> dict:
        """
        Runs the simulation according to config.model.mode.
        Returns a results dict with time series:
          - t
          - I_total
          - I_by_comm (T x K)
          - also S/E/I/R by comm
          - M(t)
        """
        T = self.n_steps + 1
        t = np.arange(T) * self.dt

        I_total = np.zeros(T, dtype=float)
        M_hist = np.zeros(T, dtype=float)

        I_comm = np.zeros((T, self.K), dtype=float)
        S_comm = np.zeros((T, self.K), dtype=float)
        E_comm = np.zeros((T, self.K), dtype=float)
        R_comm = np.zeros((T, self.K), dtype=float)

        # initial record
        I_total[0] = self._I_total_from_fractions()
        I_comm[0, :] = self.I
        S_comm[0, :] = self.S
        E_comm[0, :] = self.E
        R_comm[0, :] = self.R
        M_hist[0] = self.media.M

        for k in range(1, T):
            if self.mode == "mean_field":
                self.step_mean_field()
            elif self.mode == "network":
                self.step_network()
            else:
                raise ValueError("model.mode must be 'mean_field' or 'network'")

            I_total[k] = self._I_total_from_fractions()
            I_comm[k, :] = self.I
            S_comm[k, :] = self.S
            E_comm[k, :] = self.E
            R_comm[k, :] = self.R
            M_hist[k] = self.media.M

        return {
            "t": t,
            "I_total": I_total,
            "M": M_hist,
            "I_comm": I_comm,
            "S_comm": S_comm,
            "E_comm": E_comm,
            "R_comm": R_comm,
            "K": self.K,
            "mode": self.mode,
            "family": self.family,
            "use_exposed": self.use_exposed
        }

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    @staticmethod
    def plot_results(results: dict, title: Optional[str] = None):
        """
        Final plot: total infected + infected per community.
        """
        t = results["t"]
        I_total = results["I_total"]
        I_comm = results["I_comm"]
        K = results["K"]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t, I_total, label="I_total")

        for i in range(K):
            ax.plot(t, I_comm[:, i], linestyle="--", label=f"I_comm {i}")

        ax.set_xlabel("time")
        ax.set_ylabel("infected fraction")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(title if title else f"MultiType {results['family']} ({results['mode']})")
        fig.tight_layout()
        return fig


# ============================================================
# Standalone usage
# ============================================================

if __name__ == "__main__":
    cfg = load_config("config.json")
    model = MultiTypeEpidemic(cfg)
    res = model.run()
    fig = MultiTypeEpidemic.plot_results(res)
    plt.show()

