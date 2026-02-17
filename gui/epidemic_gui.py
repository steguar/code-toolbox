import tkinter as tk
from tkinter import filedialog, messagebox
import json

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from multi_type_epidemic import MultiTypeEpidemic


class MultiTypeEpidemicGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MultiType SIR/SIS (Mean-field / Network)")

        self.config_data = None
        self.config_path = None
        self.canvas = None

        self.build_layout()

    # ----------------------------
    # Layout
    # ----------------------------
    def build_layout(self):
        top = tk.Frame(self.root)
        top.pack(pady=5)

        tk.Button(top, text="Load Config", command=self.load_config).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=5)
        tk.Button(top, text="Run", command=self.run).pack(side=tk.LEFT, padx=5)

        # model selectors
        sel = tk.LabelFrame(self.root, text="Model")
        sel.pack(fill="x", padx=10, pady=5)

        tk.Label(sel, text="mode").grid(row=0, column=0, sticky="w")
        self.mode_var = tk.StringVar(value="mean_field")
        self.mode_menu = tk.OptionMenu(sel, self.mode_var, "mean_field", "network")
        self.mode_menu.grid(row=0, column=1, sticky="w")

        tk.Label(sel, text="family").grid(row=0, column=2, sticky="w")
        self.family_var = tk.StringVar(value="SIR")
        self.family_menu = tk.OptionMenu(sel, self.family_var, "SIR", "SIS")
        self.family_menu.grid(row=0, column=3, sticky="w")

        # sliders
        self.param_frame = tk.LabelFrame(self.root, text="Global parameters (sliders)")
        self.param_frame.pack(fill="x", padx=10, pady=5)

        # param specs: key -> (path_list, min, max, resolution, cast)
        self.param_specs = {
            "beta0": (["rates", "beta0"], 0.0, 2.0, 0.001, float),
            "gamma": (["rates", "gamma"], 0.0, 1.0, 0.001, float),
            "sigma": (["rates", "sigma"], 0.0, 200.0, 0.5, float),  # if you want "infinite", set big in config

            "epsilon_media": (["behavior_modulation", "epsilon_media"], 0.0, 2.0, 0.01, float),
            "epsilon_local": (["behavior_modulation", "epsilon_local"], 0.0, 2.0, 0.01, float),
            "alpha_risk": (["behavior_modulation", "alpha_risk"], 0.0, 2.0, 0.01, float),

            "delay_steps": (["media_intensity", "delay_steps"], 0, 200, 1, int),
            "tau": (["media_intensity", "tau"], 0.1, 50.0, 0.1, float),
            "a": (["media_intensity", "a"], 0.0, 1.0, 0.01, float),
            "g_threshold": (["media_intensity", "g_threshold"], 0.0, 0.02, 0.0001, float),

            "dt": (["simulation", "dt"], 0.01, 1.0, 0.01, float),
            "n_steps": (["simulation", "n_steps"], 50, 5000, 10, int),
            "seed": (["simulation", "seed"], 0, 9999, 1, int),
        }

        self.sliders = {}
        for r, name in enumerate(self.param_specs.keys()):
            tk.Label(self.param_frame, text=name).grid(row=r, column=0, sticky="w")
            scale = tk.Scale(
                self.param_frame,
                from_=self.param_specs[name][1],
                to=self.param_specs[name][2],
                resolution=self.param_specs[name][3],
                orient=tk.HORIZONTAL,
                length=300
            )
            scale.grid(row=r, column=1, padx=5)
            val_lbl = tk.Label(self.param_frame, text="")
            val_lbl.grid(row=r, column=2, sticky="w")

            scale.config(command=lambda v, lbl=val_lbl: lbl.config(text=str(v)))
            self.sliders[name] = (scale, val_lbl)

        # plot
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(fill="both", expand=True)

    # ----------------------------
    # Helpers to get/set config paths
    # ----------------------------
    def _get_by_path(self, d, path):
        cur = d
        for k in path:
            cur = cur[k]
        return cur

    def _set_by_path(self, d, path, value):
        cur = d
        for k in path[:-1]:
            cur = cur[k]
        cur[path[-1]] = value

    # ----------------------------
    # Config management
    # ----------------------------
    def load_config(self):
        p = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not p:
            return
        with open(p, "r") as f:
            self.config_data = json.load(f)
        self.config_path = p

        # set model selectors
        self.mode_var.set(self.config_data["model"]["mode"])
        self.family_var.set(self.config_data["model"]["family"])

        # populate sliders from config (and auto-expand slider range if needed)
        for name, spec in self.param_specs.items():
            path, mn, mx, res, cast = spec
            v = self._get_by_path(self.config_data, path)
            sc, lbl = self.sliders[name]

            # expand range if config value out-of-range
            try:
                fv = float(v)
                if fv < float(sc.cget("from")):
                    sc.config(from_=fv)
                if fv > float(sc.cget("to")):
                    sc.config(to=fv)
            except Exception:
                pass

            sc.set(v)
            lbl.config(text=str(v))

    def save_config(self):
        if self.config_data is None or self.config_path is None:
            messagebox.showerror("Error", "Load a config first.")
            return

        self._update_config_from_ui()

        with open(self.config_path, "w") as f:
            json.dump(self.config_data, f, indent=4)

        messagebox.showinfo("Saved", "Configuration saved.")

    def _update_config_from_ui(self):
        # selectors
        self.config_data["model"]["mode"] = self.mode_var.get()
        self.config_data["model"]["family"] = self.family_var.get()

        # sliders
        for name, spec in self.param_specs.items():
            path, mn, mx, res, cast = spec
            sc, _ = self.sliders[name]
            raw = sc.get()
            val = cast(raw)
            self._set_by_path(self.config_data, path, val)

    # ----------------------------
    # Run + plot
    # ----------------------------
    def run(self):
        if self.config_data is None:
            messagebox.showerror("Error", "Load a config first.")
            return

        self._update_config_from_ui()

        model = MultiTypeEpidemic(self.config_data)
        results = model.run()

        self._plot(results)

    def _plot(self, results):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig = MultiTypeEpidemic.plot_results(results)

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiTypeEpidemicGUI(root)
    root.mainloop()

