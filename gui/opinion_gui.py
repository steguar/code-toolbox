import tkinter as tk
from tkinter import filedialog, messagebox
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from opinion_dynamics import Simulation


class SimulatorGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Epidemic–Opinion Simulator")

        self.config_data = None
        self.config_path = None
        self.canvas = None

        self.sliders = {}

        self.build_layout()

    # =====================================================
    # LAYOUT
    # =====================================================

    def build_layout(self):

        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=5)

        tk.Button(top_frame, text="Load Config",
                  command=self.load_config).pack(side=tk.LEFT, padx=5)

        tk.Button(top_frame, text="Save Config",
                  command=self.save_config).pack(side=tk.LEFT, padx=5)

        tk.Button(top_frame, text="Run Simulation",
                  command=self.run_simulation).pack(side=tk.LEFT, padx=5)

        # Parameter panel
        self.param_frame = tk.LabelFrame(self.root, text="Model Parameters")
        self.param_frame.pack(fill="x", padx=10, pady=5)

        # Define parameter ranges here
        self.param_specs = {
            "alpha_social": ("opinion_dynamics", 0.0, 3.0, 0.01),
            "alpha_community": ("opinion_dynamics", 0.0, 3.0, 0.01),
            "alpha_R": ("external_effect", 0.0, 3.0, 0.01),
            "p": ("external_effect", 1, 5, 1),
            "mode": ("external_effect", 0, 1, 1),
            "a": ("media_bias", 0.0, 1.0, 0.01),
            "g_threshold": ("media_bias", 0.0, 0.01, 0.0001),
            "delay_steps": ("risk_perception", 0, 100, 1),
            "tau": ("risk_perception", 0.1, 20.0, 0.1),
        }

        for i, (param, (section, min_val, max_val, resolution)) in enumerate(self.param_specs.items()):

            label = tk.Label(self.param_frame, text=param)
            label.grid(row=i, column=0, sticky="w")

            slider = tk.Scale(
                self.param_frame,
                from_=min_val,
                to=max_val,
                resolution=resolution,
                orient=tk.HORIZONTAL,
                length=250
            )
            slider.grid(row=i, column=1, padx=5)

            value_label = tk.Label(self.param_frame, text="")
            value_label.grid(row=i, column=2)

            # Update numeric label live
            slider.config(command=lambda val, lbl=value_label: lbl.config(text=f"{float(val):.4f}"))

            self.sliders[param] = (slider, section, value_label)

        # Plot frame
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(fill="both", expand=True)

    # =====================================================
    # CONFIG MANAGEMENT
    # =====================================================

    def load_config(self):

        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")]
        )
        if not file_path:
            return

        with open(file_path, "r") as f:
            self.config_data = json.load(f)

        self.config_path = file_path

        self.populate_sliders()

    def save_config(self):

        if self.config_data is None:
            messagebox.showerror("Error", "No config loaded.")
            return

        self.update_config_from_sliders()

        with open(self.config_path, "w") as f:
            json.dump(self.config_data, f, indent=4)

        messagebox.showinfo("Saved", "Configuration saved.")

    def populate_sliders(self):

        for param, (slider, section, value_label) in self.sliders.items():
            if param in self.config_data[section]:
                value = self.config_data[section][param]
                slider.set(value)
                value_label.config(text=f"{float(value):.4f}")

    def update_config_from_sliders(self):

        for param, (slider, section, _) in self.sliders.items():

            value = slider.get()

            # Cast integers where needed
            if param in ["mode", "delay_steps", "p"]:
                value = int(value)
            else:
                value = float(value)

            self.config_data[section][param] = value

    # =====================================================
    # SIMULATION
    # =====================================================

    def run_simulation(self):

        if self.config_data is None:
            messagebox.showerror("Error", "Load a config first.")
            return

        self.update_config_from_sliders()

        sim = Simulation(self.config_data)
        results = sim.run()

        self.plot_results(results)

    # =====================================================
    # PLOTTING
    # =====================================================

    def plot_results(self, results):

        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, ax1 = plt.subplots(figsize=(8, 4))

        t = results["t"]

        ax1.plot(t, results["I"], label="I(t)")
        ax1.plot(t, results["R_perceived"], linestyle="--", label="R(t)")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Epidemic / Risk")

        ax2 = ax1.twinx()

        ax2.plot(t, results["o_mean"], label="Global opinion mean")

        ax2.fill_between(
            t,
            results["o_mean"] - results["o_std"],
            results["o_mean"] + results["o_std"],
            alpha=0.2
        )

        for k in range(results["o_mean_c"].shape[1]):
            ax2.plot(
                t,
                results["o_mean_c"][:, k],
                linestyle=":",
                label=f"Community {k}"
            )

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc="upper center", fontsize=8)

        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = SimulatorGUI(root)
    root.mainloop()

