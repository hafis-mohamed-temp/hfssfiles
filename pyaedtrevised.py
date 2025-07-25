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
print(port_name)


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

hfss.analyze_setup(setup_name)

antenna_name = patch.name

patch_y_var      = f"patch_y_{antenna_name}"
inset_dist_var   = f"inset_distance_{antenna_name}"

# Read the current numerical value of each variable from the design
p0 = hfss.variable_manager[patch_y_var].numeric_value
i0 = hfss.variable_manager[inset_dist_var].numeric_value

print("\n--- Printing Initial Variable Values ---")
val_y = hfss.variable_manager[patch_y_var].numeric_value
val_inset = hfss.variable_manager[inset_dist_var].numeric_value
print(f"Read from HFSS -> {patch_y_var}: {val_y}mm")
print(f"Read from HFSS -> {inset_dist_var}: {val_inset}mm")


opt = hfss.optimizations.add(
    calculation=f"dB(St({port_name},{port_name}))",  
    ranges={"Freq": ["2.4GHz"]},
    # variables to be optimized
    variables=[patch_y_var, inset_dist_var],
    optimization_type="Optimization",  
    #optimization_type="DesignExplorer",
    condition="<=",
    goal_value=-16,  # Target: S11 <= –20 dB
    goal_weight=1
)

opt.add_variation(
    variable_name=patch_y_var,
    min_value=p0 * 0.98,
    max_value=p0 * 1.02
)
opt.add_variation(
    variable_name=inset_dist_var,
    min_value=i0 * 0.98,
    max_value=i0 * 1.02
)

hfss.save_project()


#opt.analyze()                   

print("\n--- Printing Final Variable Values ---")
val_y = hfss.variable_manager[patch_y_var].numeric_value
val_inset = hfss.variable_manager[inset_dist_var].numeric_value
print(f"Read from HFSS -> {patch_y_var}: {val_y}mm")
print(f"Read from HFSS -> {inset_dist_var}: {val_inset}mm")


#hfss.analyze_setup(setup_name)  


s11 = hfss.post.get_solution_data(
    expressions=[f"St({port_name},{port_name})"],
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

print("\n--- Generating 3D Radiation Pattern ---")

# Define the solution name for the far-field report
solution_name = f"{setup_name} : Sweep1"

# Create the 3D far-field plot in HFSS
ff_plot = hfss.post.create_3d_far_field_plot(
    solution=solution_name,
    quantity="RealizedGainTotal",
    primary_sweep="theta",
    secondary_sweep="phi",
    context={"Freq": "2.4GHz"} # Specify frequency for the plot
)

# Export the plot to an image file 
hfss.post.export_plot_to_file(
    output_dir="results",
    plot_name=ff_plot.name, # Use the name of the plot object
    file_format=".png"
)
print(f"3D radiation plot saved to: results/{ff_plot.name}.png")


    
hfss.release_desktop(False, False)