"""
S3I2 Network Epidemic Simulator
===============================

This module implements a network-based stochastic simulator for the
S3I2 epidemic model described in:

    "When to boost: How dose timing determines the epidemic threshold"

The model describes epidemic spreading with:

    - multi-dose vaccination
    - imperfect protection
    - waning immunity
    - heterogeneous vaccination rates across communities

The epidemiological states are:

    S0 : fully susceptible individuals
    S1 : partially immunized individuals
    S2 : maximally immunized individuals
    I1 : infected after 0 or 1 immunization events
    I2 : infected after ≥2 immunization events


Transitions
-----------

S0 → I1        infection
S0 → S1        vaccination

S1 → I2        infection
S1 → S2        booster vaccination
S1 → S0        waning immunity

S2 → I2        infection
S2 → S0        waning immunity

I1 → S1        recovery
I2 → S2        recovery


Network model
-------------

The population is represented by an undirected graph.

Nodes represent individuals.
Edges represent potentially infectious contacts.

Nodes belong to communities:

    node_id com_id


Each community has its own vaccination rate.


Configuration File
------------------

JSON parameters:

node_file : path to node list

edge_file : path to edge list

T : number of time steps

beta : transmission rate

gamma : recovery rate

sigma1 : susceptibility reduction after first dose

sigma2 : susceptibility reduction after second dose

eta1 : waning rate from S1

eta2 : waning rate from S2

initial_infected_fraction

vaccination_rates :

    {
        "0": 0.01,
        "1": 0.02
    }


Usage
-----

Command line:

    python s3i2_simulator.py config.json


Python import:

    from s3i2_simulator import run_simulation

    history = run_simulation(config)


Output
------

history[t] = dictionary with counts of states:

    S0, S1, S2, I1, I2
"""


import os
import json
import argparse
import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

########################################
# Epidemiological States
########################################

S0 = 0
S1 = 1
S2 = 2
I1 = 3
I2 = 4

STATE_NAMES = {
    S0: "S0",
    S1: "S1",
    S2: "S2",
    I1: "I1",
    I2: "I2"
}


########################################
# Graph Loading
########################################

def load_graph(node_file, edge_file):
    """
    Load graph and community structure.

    Parameters
    ----------
    node_file : str

        Text file with format:

            node_id  community_id

        No header.

    edge_file : str

        Text file with format:

            source_id  target_id

        No header.

    Returns
    -------

    G : networkx.Graph

        Undirected graph where each node has attribute:

            G.nodes[i]["com"]
    """

    G = nx.Graph()

    with open(node_file) as f:
        for line in f:
            node_id, com_id = line.strip().split()
            G.add_node(int(node_id), com=int(com_id))

    with open(edge_file) as f:
        for line in f:
            s, t = line.strip().split()
            G.add_edge(int(s), int(t))

    return G


########################################
# Initialization
########################################

def initialize_states(G, config):
    """
    Initialize epidemiological states.

    All individuals start in S0 except for a
    fraction initialized as infected.

    Parameters
    ----------

    G : networkx.Graph

    config : dict

    Returns
    -------

    states : dict

        node_id → state
    """

    states = {}

    frac = config["initial_infected_fraction"]

    for n in G.nodes():

        if random.random() < frac:
            states[n] = I1
        else:
            states[n] = S0

    return states


########################################
# Infection Pressure
########################################

def infection_pressure(node, G, states):
    """
    Compute local infection pressure.

    Defined as fraction of infected neighbors.

    Parameters
    ----------

    node : int

    G : graph

    states : dict

    Returns
    -------

    float
    """

    neighbors = G[node]

    if len(neighbors) == 0:
        return 0.0

    infected = 0

    for j in neighbors:

        if states[j] in (I1, I2):
            infected += 1

    return infected / len(neighbors)


########################################
# One Simulation Step
########################################

def step(G, states, config):
    """
    Perform one synchronous update step.

    All transition probabilities are evaluated
    using the state at time t and applied
    simultaneously.

    Returns
    -------

    new_states : dict
    """

    beta = config["beta"]
    gamma = config["gamma"]

    sigma1 = config["sigma1"]
    sigma2 = config["sigma2"]

    eta1 = config["eta1"]
    eta2 = config["eta2"]

    vacc_rates = config["vaccination_rates"]

    new_states = states.copy()

    for i in G.nodes():

        state = states[i]

        com = G.nodes[i]["com"]

        c = vacc_rates[str(com)]

        lam = beta * infection_pressure(i, G, states)

        r = random.random()

        ###################################
        # S0
        ###################################

        if state == S0:

            if r < lam:
                new_states[i] = I1

            elif r < lam + c:
                new_states[i] = S1

        ###################################
        # S1
        ###################################

        elif state == S1:

            p_inf = sigma1 * lam

            if r < p_inf:
                new_states[i] = I2

            elif r < p_inf + c:
                new_states[i] = S2

            elif r < p_inf + c + eta1:
                new_states[i] = S0

        ###################################
        # S2
        ###################################

        elif state == S2:

            p_inf = sigma2 * lam

            if r < p_inf:
                new_states[i] = I2

            elif r < p_inf + eta2:
                new_states[i] = S0

        ###################################
        # I1
        ###################################

        elif state == I1:

            if r < gamma:
                new_states[i] = S1

        ###################################
        # I2
        ###################################

        elif state == I2:

            if r < gamma:
                new_states[i] = S2

    return new_states


########################################
# Observables
########################################

def count_states(states):
    """
    Count individuals in each state.

    Returns
    -------

    dict
    """

    counts = {name: 0 for name in STATE_NAMES.values()}

    for s in states.values():
        counts[STATE_NAMES[s]] += 1

    return counts

def count_states_by_community(G, states):
    """
    Count states separately for each community.

    Returns
    -------

    dict:

        community_id -> state_counts
    """

    counts = {}

    for n in G.nodes():

        com = G.nodes[n]["com"]

        if com not in counts:

            counts[com] = {
                name: 0
                for name in STATE_NAMES.values()
            }

        s = states[n]

        counts[com][STATE_NAMES[s]] += 1

    return counts

def plot_history(history, history_com, output_folder="output"):
    """
    Generate plots.

    Produces:

        output/total.png
        output/communities.png
    """

    T = len(history)

    t = range(T)

    ###################################
    # Global plot
    ###################################

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    plt.figure(figsize=(10,6))

    for state in STATE_NAMES.values():

        series = [h[state] for h in history]

        plt.plot(t, series, label=state)

    plt.xlabel("Time")
    plt.ylabel("Individuals")

    plt.legend()

    plt.title("Global epidemic dynamics")

    plt.tight_layout()

    plt.savefig(output_folder + "/total.png")

    plt.close()


    ###################################
    # Community plots
    ###################################

    communities = sorted(history_com[0].keys())

    ncom = len(communities)

    fig, axes = plt.subplots(
        ncom,
        1,
        figsize=(10,4*ncom),
        sharex=True
    )

    if ncom == 1:
        axes = [axes]

    for ax, com in zip(axes, communities):

        for state in STATE_NAMES.values():

            series = [
                h[com][state]
                for h in history_com
            ]

            ax.plot(t, series, label=state)

        ax.set_title(f"Community {com}")

        ax.set_ylabel("Individuals")

        ax.legend()

    axes[-1].set_xlabel("Time")

    plt.tight_layout()

    plt.savefig(output_folder + "/communities.png")

    plt.close()

    return fig

########################################
# Core Simulator
########################################

def run_simulation(config):
    """
    Run full simulation.

    Returns
    -------

    history_global
    history_by_community
    """

    G = load_graph(
        config["node_file"],
        config["edge_file"]
    )

    states = initialize_states(G, config)

    T = config["T"]

    history = []
    history_com = []

    for t in range(T):

        history.append(
            count_states(states)
        )

        history_com.append(
            count_states_by_community(G, states)
        )

        states = step(G, states, config)

    if config.get("plot", False):

        output_folder = config.get(
            "plot_folder",
            "simulation"
        )

        fig = plot_history(
            history,
            history_com,
            output_folder
        )

    return history, history_com, fig


########################################
# Command Line Interface
########################################

def main():
    """
    Command-line entry point.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config",
        help="JSON configuration file"
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    history, history_com, fig = run_simulation(config)

    for t, counts in enumerate(history):
        print(t, counts)


if __name__ == "__main__":
    main()
