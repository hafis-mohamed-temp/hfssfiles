
import math
import numpy as np
from pyaedt import Desktop, Hfss


class PatchAntennaDesigner:
    
    def __init__(self, f0_ghz: float = 2.4, er: float = 4.4, h_mm: float = 1.6, inset_gap_mm: float = 1.0):
        self._c = 3e8                       
        self.f0_ghz = f0_ghz
        self.f0 = f0_ghz * 1e9            
        self.er = er
        self.h_mm = h_mm
        self.inset_gap_mm = inset_gap_mm
        self.port_name = "LumpedPort"
        self.desktop: Desktop | None = None
        self.hfss: Hfss | None = None
        self._derive_dimensions()

    def _derive_dimensions(self) -> None:
        h = self.h_mm
        W = self._c / (2 * self.f0) * math.sqrt(2 / (self.er + 1)) * 1e3

        eps_eff = (self.er + 1) / 2 + (self.er - 1) / 2 * (
            1 + 12 * h / W) ** -0.5

        delta_L = (0.412 * h *
                   ((eps_eff + 0.3) * (W / h + 0.264)) /
                   ((eps_eff - 0.258) * (W / h + 0.8)))

        L = (self._c / (2 * self.f0 * math.sqrt(eps_eff)) * 1e3
             - 2 * delta_L) * 0.96  

        λ0_mm   = self._c / self.f0 * 1e3
        G1      = (1 / 90) * (W / λ0_mm) ** 2
        R_edge  = 1 / (2 * G1) if G1 else float("inf")
        R_edge  = max(R_edge, 50)
        y0      = L / 2 - (L / math.pi) * math.asin(math.sqrt(50 / R_edge))

        Wf      = self._solve_feed_width(self.er, h, 50)
        λg_mm   = self._c / (self.f0 * math.sqrt(eps_eff)) * 1e3

        self.patch_width_mm   = W
        self.patch_length_mm  = L
        self.er_eff           = eps_eff
        self.delta_L_mm       = delta_L
        self.inset_dist_mm    = y0
        self.feed_width_mm    = Wf
        self.feed_length_mm   = λg_mm / 4
        lambda_g = self._c/(self.f0*math.sqrt(eps_eff))*1e3
        margin = max(lambda_g/4, 6*h)
        self.sub_width_mm    = W + 2*margin
        self.sub_length_mm   = L + 2*margin

    @staticmethod
    def _solve_feed_width(er: float, h: float, Z0: float) -> float:
        A = (Z0 / 60) * math.sqrt((er + 1) / 2) + \
            ((er - 1) / (er + 1)) * (0.23 + 0.11 / er)
        W_over_h = 8 * math.exp(A) / (math.exp(2 * A) - 2)

        if W_over_h > 2:
            B = (377 * math.pi) / (2 * Z0 * math.sqrt(er))
            W_over_h = (2 / math.pi) * (
                B - 1 - math.log(2 * B - 1)
                + ((er - 1) / (2 * er)) * (math.log(B - 1) + 0.39 - 0.61 / er))
        return W_over_h * h

    def build_model(self, non_graphical: bool = False) -> None:
        self._launch_hfss(non_graphical)
        self._declare_design_variables()
        self._create_geometry()
        self._create_setup_and_frequency_sweep()

    def solve(self) -> None:
        self.hfss.validate_simple()
        self.hfss.analyze_setup("Setup1")

    def parametric_sweep(self, length_offset:float = 0.0, width_offset: float = 0.0) -> None:
        self._add_parametric_study(length_offset, width_offset)

    def post_process(self) -> None:
        self._s11_report_and_radiation()

    def export_reports_per_variation(self) -> None:
        self._export_variation_reports()

    # session launch
    def _launch_hfss(self, non_graphical: bool) -> None:
        self.desktop = Desktop(new_desktop=True,
                               non_graphical=non_graphical,
                               student_version=True,
                               close_on_exit=False)
        self.hfss = Hfss(solution_type="DrivenModal")
        self.hfss.modeler.model_units = "mm"
        self.hfss["InsetGap"] = f"{self.inset_gap_mm}mm"

    def _declare_design_variables(self) -> None:
        vars_ = {
            "patch_width":  self.patch_width_mm,
            "patch_length": self.patch_length_mm,
            "inset_dist":   self.inset_dist_mm,
            "feed_width":   self.feed_width_mm,
            "feed_length":  self.feed_length_mm,
            "sub_width":    self.sub_width_mm,
            "sub_length":   self.sub_length_mm,
            "substrate_h":  self.h_mm,
        }

        for key, value in vars_.items():
            self.hfss[key] = f"{value:.6f}mm"

    def _create_geometry(self) -> None:
        h, gap = self.h_mm, self.inset_gap_mm
        W, L   = self.patch_width_mm, self.patch_length_mm
        y0     = self.inset_dist_mm
        Wf     = self.feed_width_mm
        fL     = self.feed_length_mm
        sW, sL = self.sub_width_mm, self.sub_length_mm

        self.hfss.modeler.create_box(
            [-sW/2, -sL/2, 0], [sW, sL, -h],
            name="Substrate", material_name="FR4_epoxy")

        ground = self.hfss.modeler.create_rectangle(
            "XY", [-sW/2, -sL/2, -h], [sW, sL], name="Ground")
        self.hfss.assign_perfecte_to_sheets(ground.name, "PEC_Ground")

        patch = self.hfss.modeler.create_rectangle(
            "XY", [-W/2, -L/2, 0], [W, L], name="PatchSeed")

        cutout = self.hfss.modeler.create_rectangle(
            "XY",
            [-(Wf + 2*gap)/2, L/2 - y0, 0],
            [Wf + 2*gap, y0],
            name="Cutout")
        self.hfss.modeler.subtract(patch, cutout, keep_originals=False)

        feed = self.hfss.modeler.create_rectangle(
            "XY",
            [-Wf/2,  L/2 - y0, 0],
            [Wf, fL + y0],
            name="Feed")
        self.hfss.modeler.unite([patch.name, feed.name])
        patch.name = "Patch_Conductor"
        self.hfss.assign_perfecte_to_sheets(patch.name, "PEC_Patch")

        # port
        port_y = L/2 + fL
        port_sheet = self.hfss.modeler.create_rectangle(
            "XZ", [-Wf/2, port_y, -h], [h, Wf], name="PortSheet")

        self.hfss.lumped_port(
            assignment       = port_sheet.name,
            reference        = "Ground",
            impedance        = 50,
            name             = self.port_name,
            integration_line = [[0, port_y, -h], [0, port_y, 0]],
            create_port_sheet=False,
            renormalize=True)

        # radiation boundary
        λ0_mm = self._c / self.f0 * 1e3
        mgn    = λ0_mm / 4
        box_o  = [-sW/2 - mgn, -sL/2 - mgn, -h - mgn]
        box_s  = [ sW + 2*mgn,  sL + 2*mgn,  h + 2*mgn]

        airbox = self.hfss.modeler.create_box(
            box_o, box_s, name="Airbox", material_name="vacuum")
        self.hfss.assign_radiation_boundary_to_objects(
            airbox.name, "RadBoundary")

        self.hfss.insert_infinite_sphere(
            name="3D", x_start=0, x_stop=180, x_step=2,
            y_start=0, y_stop=360, y_step=2, units="deg")

    def _create_setup_and_frequency_sweep(self) -> None:
        setup = self.hfss.create_setup("Setup1")
        setup.props.update(
            Frequency              = f"{self.f0_ghz}GHz",
            MaximumPasses          = 20,
            MinimumPasses          = 2,
            MinimumConvergedPasses = 1,
            PercentRefinement      = 30,
            PercentError           = 0.2)
        setup.update()

        sweep = self.hfss.create_linear_count_sweep(
            setup              ="Setup1",
            name               ="Sweep1",
            sweep_type="Discrete",
            start_frequency    =1.0,
            stop_frequency     =3.0,
            units              ="GHz",
            num_of_freq_points =20,
            save_fields=True 
        )
        sweep.props["SaveRadFields"] = True
        sweep.update()

    def _add_parametric_study(self, length_offset: float, width_offset: float) -> None:
        # singlevalue sweep for length and width
        new_len = self.patch_length_mm + length_offset 
        new_width = self.patch_width_mm + width_offset

        psweep = self.hfss.parametrics.add(
            variable = "patch_length",
            start_point = new_len,
            variation_type = "SingleValue",
            name="PatchLengthSweep")
        psweep.add_variation(
            sweep_variable="patch_width",
            start_point = new_width,
            variation_type="SingleValue")
        psweep.analyze(cores=4)

    def _s11_report_and_radiation(self) -> None:
        expr = f"dB(S({self.port_name},{self.port_name}))"
        s11_report = self.hfss.create_scattering(
            plot="S11_modal", sweep="Setup1 : Sweep1",
            ports=[self.port_name], ports_excited=[self.port_name])

        data = self.hfss.post.get_solution_data(
            expressions=[expr], setup_sweep_name="Setup1 : Sweep1")
        freq_GHz = np.array(data.primary_sweep_values, float)
        mag_dB   = np.array(data.data_real())

        f0_idx   = np.argmin(mag_dB)
        f0_GHz   = freq_GHz[f0_idx]
        below10  = np.where(mag_dB < -10)[0]
        if below10.size:
            fL, fH = freq_GHz[below10[[0, -1]]]
            bw_MHz = (fH - fL) * 1e3
        else:
            fL = fH = bw_MHz = None

        print("\n Numerical results ")
        print(f"Resonant f0 : {f0_GHz:.3f} GHz    "
              f"(|S11| = {mag_dB[f0_idx]:.1f} dB)")
        if bw_MHz:
            print(f"-10 dB BW  : {fL:.3f} – {fH:.3f} GHz  ({bw_MHz:.1f} MHz)")
        else:
            print("Return-loss never crossed –10 dB in 1-3 GHz sweep.")

        ff = self.hfss.post.reports_by_category.far_field(
            expressions="db(GainTotal)",
            setup_sweep_name="Setup1 : Sweep1",
            sphere_name="3D",
            Freq=[f"{self.f0_ghz:.3f}GHz"])
        ff.report_type = "3D Polar Plot"
        ff.create(name="3D Radiation Pattern")

    def _export_variation_reports(self) -> None:
        var_list = self.hfss.list_of_variations("Setup1", "Sweep1")   
        s11_expr = f"dB(S({self.port_name},{self.port_name}))"

        for idx, var in enumerate(var_list):
            self.hfss.create_scattering(
                plot=f"S11_var{idx}",
                sweep="Setup1 : Sweep1",
                ports=[self.port_name],
                ports_excited=[self.port_name],
                variations=var)

            ff = self.hfss.post.reports_by_category.far_field(
                expressions="db(GainTotal)",
                setup_sweep_name="Setup1 : Sweep1",
                sphere_name="3D",
                Freq=[f"{self.f0_ghz:.3f}GHz"],
                variations=var)
            ff.report_type = "3D Polar Plot"
            ff.create(name=f"FF_var{idx}")


if __name__ == "__main__":
    design = PatchAntennaDesigner()
    design.build_model(non_graphical=False) 
    design.solve()                    
    design.parametric_sweep(2, 2)   #vary length by +2 mm and width by +2 mm      
    design.post_process()                 
    design.export_reports_per_variation()  
