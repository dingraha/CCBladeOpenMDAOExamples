import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
from openmdao.utils.spline_distributions import cell_centered
from ccblade_openmdao_examples.ccblade_openmdao_component import BEMTRotorSDTComp
from julia.OpenMDAO import make_component


def get_problem():
    B = 3  # Number of blades.
    D = 24.0*0.0254  # Diameter in meters.
    P = 16.0*0.0254  # Pitch in meters (used in the twist distribution).
    c = 1.5*0.0254   # Constant chord in meters.
    Rtip = 0.5*D
    Rhub = 0.2*Rtip  # Just guessing on the hub diameter.

    # Not sure about the atmospheric conditions, so I'll just use the [ICAO standard
    # atmosphere at sealevel.](https://www.engineeringtoolbox.com/international-standard-atmosphere-d_985.html)
    p0 = 101325.0
    T0 = 273.15 + 15.0
    gam = 1.4
    speedofsound = np.sqrt(gam*287.058*T0)
    rho0 = gam*p0/speedofsound**2
    mu = rho0*1.461e-5

    # Operating parameters for this case.
    rpm = 7200.0
    M_infty = 0.11
    v = M_infty*speedofsound  # axial velocity in m/sec.
    omega = 2*np.pi/60.0*rpm  # propeller rotation rate in rad/sec.
    pitch = 0.0

    # Lower and upper limits on the chord design variable, in meters.
    chord_lower = 1.0*0.0254
    chord_upper = 5.0*0.0254

    # Lower and upper limits on the twist design variable, radians.
    theta_lower = 5*np.pi/180.0
    theta_upper = 85*np.pi/180.0

    # Target thrust value in Newtons.
    thrust_target = 97.246

    # Get the initial blade geometry.
    num_cp = 8
    radii_cp0 = np.linspace(Rhub, Rtip, num_cp)
    chord_cp0 = c*np.ones(num_cp)
    theta_cp0 = np.arctan(P/(np.pi*D*radii_cp0/Rtip))

    num_operating_points = 1
    num_radial = 30

    prob = om.Problem()

    comp = om.IndepVarComp()
    comp.add_output("Rhub", val=Rhub, units="m")
    comp.add_output("Rtip", val=Rtip, units="m")
    comp.add_output("radii_cp", val=radii_cp0, units="m")
    comp.add_output("chord_cp", val=chord_cp0, units="m")
    comp.add_output("theta_cp", val=theta_cp0, units="rad")
    comp.add_output("v", val=v, shape=num_operating_points, units="m/s")
    comp.add_output("omega", val=omega, shape=num_operating_points, units="rad/s")
    comp.add_output("pitch", val=pitch, shape=num_operating_points, units="rad")
    prob.model.add_subsystem("inputs_comp", comp, promotes_outputs=["*"])

    x_cp = np.linspace(0.0, 1.0, num_cp)
    x_interp = cell_centered(num_radial, 0.0, 1.0)
    interp_options = {"delta_x": 0.1}
    comp = om.SplineComp(method="akima", interp_options=interp_options, x_cp_val=x_cp, x_interp_val=x_interp)
    comp.add_spline(y_cp_name="radii_cp", y_interp_name="radii", y_units="m")
    comp.add_spline(y_cp_name="chord_cp", y_interp_name="chord", y_units="m")
    comp.add_spline(y_cp_name="theta_cp", y_interp_name="theta", y_units="rad")
    prob.model.add_subsystem("akima_comp", comp,
                             promotes_inputs=["radii_cp", "chord_cp", "theta_cp"],
                             promotes_outputs=["radii", "chord", "theta"])

    af_fname = "../data/xf-n0012-il-500000.dat"
    comp = make_component(
        BEMTRotorSDTComp(
            af_fname=af_fname, cr75=c/Rtip, Re_exp=0.6,
            num_operating_points=num_operating_points, num_blades=B,
            num_radial=num_radial, rho=rho0, mu=mu, speedofsound=speedofsound, use_hubtip_losses=False))
    prob.model.add_subsystem("bemt_rotor_comp", comp, promotes_inputs=["Rhub", "Rtip", "radii", "chord", "theta", "v", "omega", "pitch"], promotes_outputs=["thrust", "torque", "efficiency"])

    prob.model.linear_solver = om.DirectSolver()
    prob.driver = om.pyOptSparseDriver(optimizer="SNOPT")

    prob.model.add_design_var("chord_cp", lower=chord_lower, upper=chord_upper, ref=1e-2)
    prob.model.add_design_var("theta_cp", lower=theta_lower, upper=theta_upper, ref=1e0)
    prob.model.add_objective("efficiency", ref=-1e0)
    prob.model.add_constraint("thrust", lower=thrust_target, upper=thrust_target, units="N", ref=1e2)

    prob.setup(check=True)

    return prob


if __name__ == "__main__":
    p = get_problem()
    p.run_driver()

    radii_cp = p.get_val("radii_cp", units="inch")
    radii = p.get_val("radii", units="inch")
    chord_cp = p.get_val("chord_cp", units="inch")
    chord = p.get_val("chord", units="inch")
    theta_cp = p.get_val("theta_cp", units="deg")
    theta = p.get_val("theta", units="deg")

    cmap = plt.get_cmap("tab10")
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    ax0.plot(radii_cp, chord_cp, color=cmap(0), marker="o")
    ax0.plot(radii, chord, color=cmap(0))
    ax0.set_ylim(0.0, 5.0)
    ax1.plot(radii_cp, theta_cp, color=cmap(0), marker="o")
    ax1.plot(radii, theta, color=cmap(0))
    fig.savefig("chord_theta.png")
