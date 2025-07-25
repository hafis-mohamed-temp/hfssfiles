from pyaedt import Desktop, Hfss
from ansys.aedt.toolkits.antenna.backend.antenna_models.patch import RectangularPatchInset
import matplotlib.pyplot as plt, numpy as np
from pathlib import Path

hfss = Hfss(project="Patch_FR4_24GHz", design="InsetPatch", solution_type="Terminal")

hfss.create_open_region(frequency="2.4GHz")

hfss.modeler.model_units = "mm"

patch = RectangularPatchInset(hfss, frequency=2.4, frequency_unit="GHz",
                              material="FR4_epoxy", substrate_height=1.6, length_unit="mm")
patch.model_hfss()
patch.setup_hfss()

port_name = hfss.excitation_names[0]

if not hfss.setup_names:
    setup = hfss.create_setup("Setup1")
    setup_name = setup.name
    setup.props["Frequency"] = "2.4GHz"
    setup.props["MaximumPasses"] = 20
    setup.update()
else:
    setup = hfss.get_setup(hfss.setup_names[0])
    setup_name = setup.name

hfss.create_linear_count_sweep(setup=setup_name, units="GHz",
                               start_frequency=1, stop_frequency=3,
                               num_of_freq_points=201,
                               name="Sweep1", sweep_type="Interpolating",
                               save_fields=True)

hfss.save_project()

hfss.analyze_setup(setup_name)

s11 = hfss.post.get_solution_data(
    expressions=[f"S({port_name},{port_name})"],
    setup_sweep_name=f"{setup_name} : Sweep1")

freq = np.asarray(s11.primary_sweep_values, float)  # GHz
mag_lin = np.asarray(s11.data_magnitude())
mag_db = 20 * np.log10(mag_lin)

idx_min = np.argmin(mag_db)
f0_GHz = freq[idx_min]
below10 = np.where(mag_db < -10)[0]
fL_GHz = freq[below10[0]] if below10.size else None
fH_GHz = freq[below10[-1]] if below10.size else None
bw_MHz = None if fL_GHz is None else (fH_GHz - fL_GHz) * 1000

Path("results").mkdir(exist_ok=True)

plt.figure(dpi=120)
plt.plot(freq, mag_db)
plt.scatter(f0_GHz, mag_db[idx_min], color="red", label=f"f₀ = {f0_GHz:.3f} GHz")
plt.axhline(-10, color="gray", ls="--", lw=0.8)
plt.title("|S11| (Driven-Modal)")
plt.xlabel("Frequency (GHz)")
plt.ylabel("dB")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("results/s11_plot.png")
plt.close()

print("\n--- Numerical results -----------------------------------------")
print(f"Resonant frequency f₀ : {f0_GHz:.3f} GHz (|S11| = {mag_db[idx_min]:.1f} dB)")
if bw_MHz:
    print(f"-10 dB bandwidth : {fL_GHz:.3f} – {fH_GHz:.3f} GHz ({bw_MHz:.1f} MHz)")
else:
    print("Return-loss never crossed –10 dB in 1-3 GHz sweep.")
    print("S-parameter plot : results/s11_plot.png")

# ─── 3D Far-Field Radiation Pattern ────────────────────────────────────────────

# 3D polar report for Total Gain in dB
report_3d = hfss.post.reports_by_category.far_field(
    expressions="db(GainTotal)",             # plot GainTotal in dB
    setup_sweep_name="Setup1 : Sweep1",      # sweep name
    sphere_name="3D",                        # default infinite sphere
    Freq=[f"{f0_GHz:.3f}GHz"],               # evaluate at resonance
)
report_3d.report_type = "3D Polar Plot"      # switch to a 3D surface
report_3d.create(name="3D Radiation Pattern")

# Extract solution data and render in Python
sol_data = report_3d.get_solution_data()      # returns a SolutionData object :contentReference[oaicite:0]{index=0}
sol_data.plot_3d()                             # convenience wrapper draws and shows/saves the figure

fig = sol_data.plot_3d(
    snapshot_path="results/3d_radiation_pattern.png", 
    show=False                                     
)


fig.savefig("results/3d_radiation_pattern.png")