import math
import numpy as np
import matplotlib.pyplot as plt
from pyaedt import Desktop, Hfss
import os

class PatchAntennaDesigner:
    """
    Designs a 2.4-GHz rectangular patch antenna with an inset feed on an FR4 substrate.
    """

    def __init__(self, f0_ghz=2.4, er=4.4, h_mm=1.6, inset_gap_mm=1.0):
        # --- Constants and Inputs ---
        self.c = 3e8                   # Speed of light in m/s
        self.f0 = f0_ghz * 1e9         # Design frequency in Hz
        self.er = er                   # Substrate relative permittivity
        self.h = h_mm                  # Substrate thickness in mm
        self.inset_gap_mm = inset_gap_mm
        self.port_name = "LumpedPort"
        self.hfss = None
        self.desktop = None

        # --- Compute all of the closed-form dimensions ---
        self._closed_form()

    def _closed_form(self):
        h = self.h
        # 1) Patch width W (mm)
        W = self.c/(2*self.f0)*math.sqrt(2/(self.er+1))*1e3
        # 2) Effective permittivity
        eps_eff = (self.er+1)/2 + (self.er-1)/2*(1+12*h/W)**-0.5
        # 3) deltaL (mm)
        delta_L = 0.412*h*((eps_eff+0.3)*(W/h+0.264))/((eps_eff-0.258)*(W/h+0.8))
        # 4) Patch length L (mm)
        L = self.c/(2*self.f0*math.sqrt(eps_eff))*1e3 - 2*delta_L
        # 5) Edge resistance (ohm)
        lambda0_mm = self.c/self.f0*1e3
        G1 = (1/90)*(W/lambda0_mm)**2
        R_edge = 1/(2*G1) if G1!=0 else float("inf")
        # 6) Inset distance y0
        R_edge_clamped = max(R_edge,50)
        y0 = (L/math.pi)*math.acos(math.sqrt(50/R_edge_clamped))
        # 7) Feed width Wf
        Wf = self._solve_feed_width(self.er, h, 50)

        # store
        self.patch_width_mm  = W
        self.patch_length_mm = L
        self.er_eff          = eps_eff
        self.delta_L         = delta_L
        self.inset_dist_mm   = y0
        self.feed_width_mm   = Wf
        # quarter‐wave feed length
        lambda_g = self.c/(self.f0*math.sqrt(eps_eff))*1e3
        self.feed_length_mm  = lambda_g/4
        # substrate dims
        margin = max(lambda_g/4, 6*h)
        self.sub_width_mm    = W + 2*margin
        self.sub_length_mm   = L + 2*margin

        # print summary
        print("-- Closed-Form Design Outputs --")
        print(f"Patch Width (mm)         : {W:.4f}")
        print(f"Patch Length (mm)        : {L:.4f}")
        print(f"Effective Permittivity   : {eps_eff:.4f}")
        print(f"Length Extension ΔL (mm) : {delta_L:.4f}")
        print(f"Inset Distance y0 (mm)   : {y0:.4f}")
        print(f"Feed Width (mm)          : {Wf:.4f}")
        print(f"Feed Length (mm)         : {self.feed_length_mm:.4f}")
        print("---------------------------------")

    def _solve_feed_width(self, er, h, Z0):
        A = (Z0/60)*math.sqrt((er+1)/2) + ((er-1)/(er+1))*(0.23+0.11/er)
        W_over_h = 8*math.exp(A)/(math.exp(2*A)-2)
        if W_over_h > 2:
            B = (377*math.pi)/(2*Z0*math.sqrt(er))
            W_over_h = (2/math.pi)*(B-1-math.log(2*B-1)
                       +((er-1)/(2*er))*(math.log(B-1)+0.39-0.61/er))
        return W_over_h * h

    def build_model(self, non_graphical=False):
        # launch AEDT
        self.desktop = Desktop(new_desktop=True,
                               non_graphical=non_graphical,
                               student_version=True,
                               close_on_exit=False)
        self.hfss = Hfss(solution_type="DrivenModal")
        self.hfss.modeler.model_units = "mm"
        self.hfss["InsetGap"] = f"{self.inset_gap_mm}mm"
        print("HFSS project created. Building geometry...")

        # pull computed dims
        W, L, y0 = (self.patch_width_mm,
                    self.patch_length_mm,
                    self.inset_dist_mm)
        Wf, h    = (self.feed_width_mm, self.h)
        fL       = self.feed_length_mm
        subW,subL= (self.sub_width_mm,self.sub_length_mm)
        gap      = self.inset_gap_mm

        var_dict = {
            "patch_width"  : f"{W:.6f}mm",
            "patch_length" : f"{L:.6f}mm",
            "inset_dist"   : f"{y0:.6f}mm",
            "feed_width"   : f"{Wf:.6f}mm",
            "feed_length"  : f"{fL:.6f}mm",
            "sub_width"    : f"{subW:.6f}mm",
            "sub_length"   : f"{subL:.6f}mm",
            "substrate_h"  : f"{h:.6f}mm"
        }
        for vname, vexpr in var_dict.items():
            self.hfss[vname] = vexpr


        # 1) Substrate box
        self.hfss.modeler.create_box(
            origin=[-subW/2, -subL/2, 0],
            sizes =[subW, subL, -h],
            name  ="Substrate",
            material_name="FR4_epoxy"
        )

        # 2) Ground
        ground = self.hfss.modeler.create_rectangle(
            orientation="XY",
            origin=[-subW/2, -subL/2, -h],
            sizes =[subW, subL],
            name  ="Ground"
        )
        self.hfss.assign_perfecte_to_sheets(ground.name,"PEC_Ground")

        # 3) Patch
        patch = self.hfss.modeler.create_rectangle(
            orientation="XY",
            origin=[-W/2, -L/2, 0],
            sizes =[W, L],
            name  ="Patch_Seed"
        )

        # 4) Cutout
        cutout = self.hfss.modeler.create_rectangle(
            orientation="XY",
            origin=[- (Wf+2*gap)/2, L/2 - y0, 0],
            sizes =[Wf+2*gap, y0],
            name  ="Cutout"
        )
        self.hfss.modeler.subtract(patch, cutout, keep_originals=False)

        # 5) Feed
        feed = self.hfss.modeler.create_rectangle(
            orientation="XY",
            origin=[-Wf/2, L/2 - y0, 0],
            sizes =[Wf, fL+y0],
            name  ="Feed"
        )
        self.hfss.modeler.unite([patch.name, feed.name])
        patch.name = "Patch_Conductor"
        self.hfss.assign_perfecte_to_sheets(patch.name,"PEC_Patch")

        # 6) Port sheet on XZ at feed‐end
        port_y = L/2 + fL
        port = self.hfss.modeler.create_rectangle(
            orientation="XZ",
            origin=[-Wf/2, port_y, -h],
            sizes =[h, Wf],
            name  ="PortSheet"
        )
        # lumped port
        self.hfss.lumped_port(
            assignment    =port.name,
            reference     ="Ground",
            impedance     =50,
            name          =self.port_name,
            integration_line=[[0,port_y,-h],[0,port_y,0]],
            create_port_sheet=False,
            renormalize=True
        )

        # 7) Airbox + radiation
        lambda0 = self.c/self.f0*1e3
        mgn     = lambda0/4
        abox_o  =[-subW/2-mgn,-subL/2-mgn,-h-mgn]
        abox_s  =[subW+2*mgn,subL+2*mgn,h+2*mgn]
        abox    =self.hfss.modeler.create_box(
            origin       =abox_o,
            sizes        =abox_s,
            name         ="Airbox",
            material_name="vacuum"
        )
        self.hfss.assign_radiation_boundary_to_objects(abox.name,"RadBoundary")

        # 8) Setup + sweep
        setup = self.hfss.create_setup("Setup1")
        setup.props.update({
            "Frequency"               : f"{self.f0/1e9}GHz",
            "MaximumPasses"           : 12,
            "MinimumPasses"           : 2,
            "MinimumConvergedPasses"  : 1,
            "PercentRefinement"       : 30,
            "PercentError"            : 1
        })
        setup.update()
        self.hfss.create_linear_count_sweep(
            setup              ="Setup1",
            name               ="Sweep1",
            start_frequency    =1.0,
            stop_frequency     =3.0,
            units              ="GHz",
            num_of_freq_points =201            
        )
        print("HFSS setup created successfully.")

    def solve(self):
        print("Validating design and starting analysis…")
        self.hfss.validate_simple()
        self.hfss.analyze_setup("Setup1")
        print("Solve finished.")

    def post_process(self):
        print("Performing post-processing…")
        expr = f"dB(S({self.port_name},{self.port_name}))"
        data = self.hfss.post.get_solution_data(
            expressions       =[expr],
            setup_sweep_name  ="Setup1 : Sweep1"
        )
        f       = np.array(data.primary_sweep_values)
        s11_db  = np.array(data.data_db20())
        idx     = np.argmin(s11_db)
        f0_sim  = f[idx]
        bw_idx  = np.where(s11_db < -10)[0]
        if bw_idx.size>1:
            bw = f[bw_idx[-1]] - f[bw_idx[0]]
            print(f"Resonance={f0_sim:.3f} GHz, BW={bw:.3f} GHz")
        else:
            print(f"Resonance={f0_sim:.3f} GHz, no –10 dB match in sweep")
        # plot…
        plt.figure()
        plt.plot(f, s11_db)
        plt.axhline(-10, ls="--", color="r")
        plt.axvline(f0_sim, ls="--", color="g")
        plt.title("S11 (dB)")
        plt.xlabel("Freq (GHz)")
        plt.ylabel("dB")
        plt.grid(True)
        plt.savefig("S11_plot.png")
        print("Plot saved: S11_plot.png")

        # ─── 3D Far-Field Radiation Pattern ────────────────────────────────────────────

        # Build a 3D polar report for Total Gain in dB
        report_3d = self.hfss.post.reports_by_category.far_field(
            expressions="db(GainTotal)",             # plot GainTotal in dB
            setup_sweep_name="Setup1 : Sweep1",      # your sweep name
            sphere_name="3D",                        # the default infinite sphere
            Freq=[f"{f0_GHz:.3f}GHz"],               # evaluate at resonance
        )
        report_3d.report_type = "3D Polar Plot"      # switch to a 3D surface
        report_3d.create(name="3D Radiation Pattern")

        # Extract solution data and render in Python
        sol_data = report_3d.get_solution_data()      
        sol_data.plot_3d()                             # convenience wrapper draws and shows/saves the figure

        fig = sol_data.plot_3d(
            snapshot_path="results/3d_radiation_pattern.png",  # where to save
            show=False                                       # don’t block with plt.show()
        )

        # if you ever need to tweak or re-save it:
        fig.savefig("results/3d_radiation_pattern.png")

    def release(self):
        if self.desktop:
            self.desktop.release_desktop()
            print("AEDT session closed.")


if __name__ == "__main__":
    antenna = PatchAntennaDesigner()
    antenna.build_model(non_graphical=False)
    antenna.solve()
    antenna.post_process()
    # antenna.release()

