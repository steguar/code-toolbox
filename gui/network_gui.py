import tkinter as tk
from tkinter import filedialog, messagebox
import json
import subprocess


class NetworkGenerationGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Network Generation")

        self.config_data = None
        self.config_path = None

        self.build_layout()

    # -------------------------------------------------

    def build_layout(self):

        top = tk.Frame(self.root)
        top.pack(pady=5)

        tk.Button(top, text="Load Config", command=self.load_config).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Run Network Generation", command=self.run_network).pack(side=tk.LEFT, padx=5)

        # mode selector
        mode_frame = tk.LabelFrame(self.root, text="Mode")
        mode_frame.pack(fill="x", padx=10, pady=5)

        self.mode_var = tk.StringVar(value="generate")

        tk.Radiobutton(mode_frame, text="Generate synthetic",
                       variable=self.mode_var, value="generate").pack(anchor="w")

        tk.Radiobutton(mode_frame, text="Randomize observed",
                       variable=self.mode_var, value="randomize").pack(anchor="w")

        # parameters frame
        self.param_frame = tk.LabelFrame(self.root, text="Parameters")
        self.param_frame.pack(fill="x", padx=10, pady=5)

        self.entries = {}

        param_list = [
            "N", "avgk", "gamma", "communities",
            "assortativity", "order_decay",
            "beta", "n_runs", "n_graphs"
        ]

        for i, p in enumerate(param_list):
            tk.Label(self.param_frame, text=p).grid(row=i, column=0, sticky="w")
            e = tk.Entry(self.param_frame)
            e.grid(row=i, column=1)
            self.entries[p] = e

        # input graph selector for randomize
        tk.Label(self.param_frame, text="input_graph").grid(row=10, column=0, sticky="w")
        self.input_graph_entry = tk.Entry(self.param_frame)
        self.input_graph_entry.grid(row=10, column=1)
        tk.Button(self.param_frame, text="Browse",
                  command=self.browse_graph).grid(row=10, column=2)

    # -------------------------------------------------

    def browse_graph(self):
        f = filedialog.askopenfilename(filetypes=[("GraphML", "*.graphml")])
        if f:
            self.input_graph_entry.delete(0, tk.END)
            self.input_graph_entry.insert(0, f)

    # -------------------------------------------------

    def load_config(self):
        p = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not p:
            return

        with open(p, "r") as f:
            self.config_data = json.load(f)

        self.config_path = p

        net_cfg = self.config_data

        self.mode_var.set(net_cfg["mode"])

        gen = net_cfg["generate"]
        for k in self.entries:
            if k in gen:
                self.entries[k].delete(0, tk.END)
                self.entries[k].insert(0, str(gen[k]))

        rand = net_cfg["randomize"]
        if "input_graph" in rand:
            self.input_graph_entry.delete(0, tk.END)
            self.input_graph_entry.insert(0, rand["input_graph"])

    # -------------------------------------------------

    def save_config(self):
        if not self.config_data:
            return

        self.update_config()

        with open(self.config_path, "w") as f:
            json.dump(self.config_data, f, indent=4)

        messagebox.showinfo("Saved", "Config updated")

    # -------------------------------------------------

    def update_config(self):

        net_cfg = self.config_data
        net_cfg["mode"] = self.mode_var.get()

        for k, e in self.entries.items():
            try:
                val = float(e.get())
                if val.is_integer():
                    val = int(val)
            except:
                val = e.get()

            net_cfg["generate"][k] = val

        net_cfg["randomize"]["input_graph"] = self.input_graph_entry.get()

    # -------------------------------------------------

    def run_network(self):

        if not self.config_data:
            messagebox.showerror("Error", "Load config first")
            return

        self.update_config()

        net_cfg = self.config_data
        mode = net_cfg["mode"]
        output_folder = net_cfg["output_folder"]

        if mode == "generate":
            g = net_cfg["generate"]

            cmd = [
                "python", "geometric_block_model/src/rhbm/rhbm_generate.py",
                "-N", str(g["N"]),
                "-k", str(g["avgk"]),
                "-g", str(g["gamma"]),
                "-n", str(g["communities"]),
                "-p", str(g["assortativity"]),
                "-q", str(g["order_decay"]),
                "-b", str(g["beta"]),
                "--n_runs", str(g["n_runs"]),
                "--n_graphs", str(g["n_graphs"]),
                "-o", output_folder
            ]

        else:
            r = net_cfg["randomize"]

            cmd = [
                "python", "geometric_block_model/src/rhbm/rhbm_randomize.py",
                "-i", r["input_graph"],
                "-b", str(r["beta"]),
                "--n_runs", str(r["n_runs"]),
                "--n_graphs", str(r["n_graphs"]),
                "-o", output_folder
            ]

        try:
            subprocess.run(cmd, check=True)
            messagebox.showinfo("Done", "Network generation completed")
        except subprocess.CalledProcessError:
            messagebox.showerror("Error", "Network generation failed")


if __name__ == "__main__":
    root = tk.Tk()
    app = NetworkGenerationGUI(root)
    root.mainloop()

