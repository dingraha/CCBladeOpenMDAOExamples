@concrete struct BEMTRotorCAComp <: AbstractImplicitComp
    num_operating_points
    num_blades
    num_radial
    rho
    mu
    speedofsound
    airfoil_interp
    mach
    reynolds
    rotation
    tip
    apply_nonlinear_forwarddiffable!
    x
    y
    J
    forwarddiff_config
end

function BEMTRotorCAComp(; af_fname, cr75, Re_exp, num_operating_points, num_blades, num_radial, rho, mu, speedofsound, use_hubtip_losses=true)
    # Get the airfoil polar interpolator and various correction factors.
    af, mach, reynolds, rotation, tip = get_airfoil(af_fname=af_fname, cr75=cr75, Re_exp=Re_exp)

    if ! use_hubtip_losses
        tip = nothing
    end

    function apply_nonlinear_forwarddiffable!(y, x)
        T = eltype(x)

        # Unpack the inputs.
        phi = x[:phi]
        Rhub = x[:Rhub]
        Rtip = x[:Rtip]
        radii = x[:radii]
        chord = x[:chord]
        theta = x[:theta]
        v = x[:v]
        omega = x[:omega]
        pitch = x[:pitch]

        # Create the CCBlade rotor struct.
        turbine = false
        precone = zero(T)
        rotor = CCBlade.Rotor(Rhub, Rtip, num_blades, precone, turbine, mach, reynolds, rotation, tip)

        # Create the CCBlade sections.
        sections = CCBlade.Section.(radii, chord, theta, Ref(af))

        # Create the CCBlade operating points.
        Vx = v
        Vy = omega.*radii
        ops = CCBlade.OperatingPoint.(Vx, Vy, rho, pitch, mu, speedofsound)

        # Solve the BEMT equations.
        Rs_and_outs = CCBlade.residual.(phi, Ref(rotor), sections, ops)
        Rs = getindex.(Rs_and_outs, 1)
        outs = getindex.(Rs_and_outs, 2)

        # Get the thrust and torque, then the efficiency, etc.
        # coefficients.
        thrust, torque = CCBlade.thrusttorque(rotor, sections, outs)
        eff, CT, CQ = CCBlade.nondim(thrust, torque, Vx, omega, rho, rotor, "propeller")
        if thrust > zero(T)
            figure_of_merit, CT, CP = CCBlade.nondim(thrust, torque, Vx, omega, rho, rotor, "helicopter")
        else
            figure_of_merit = zero(T)
        end

        # Put the outputs in the output array.
        y[:phi] .= Rs
        y[:thrust] = thrust
        y[:torque] = torque
        y[:eff] = eff
        y[:figure_of_merit] = figure_of_merit

        return nothing
    end

    # Initialize the input and output vectors needed by ForwardDiff.jl. (The
    # ForwardDiff.jl inputs include phi, but that's an OpenMDAO output.)
    X = ComponentArray(
        phi=zeros(Float64, num_radial), Rhub=0.0, Rtip=0.0, radii=zeros(Float64, num_radial), chord=zeros(Float64, num_radial),
        theta=zeros(Float64, num_radial), v=0.0, omega=0.0, pitch=0.0)
    Y = ComponentArray(
        phi=zeros(Float64, num_radial), thrust=0.0, torque=0.0, eff=0.0, figure_of_merit=0.0)
    J = Y.*X'

    # Get the JacobianConfig object, which we'll reuse each time when calling
    # the ForwardDiff.jacobian! function (apparently good for efficiency).
    config = ForwardDiff.JacobianConfig(apply_nonlinear_forwarddiffable!, Y, X)

    return BEMTRotorCAComp(num_operating_points, num_blades, num_radial, rho, mu, speedofsound, af, mach, reynolds, rotation, tip, apply_nonlinear_forwarddiffable!, X, Y, J, config)
end

#' Need a setup function, just like a Python OpenMDAO `Component`.
function OpenMDAO.setup(self::BEMTRotorCAComp)
    num_operating_points = self.num_operating_points
    num_radial = self.num_radial

    # Declare the OpenMDAO inputs.
    input_data = Vector{VarData}()
    push!(input_data, VarData("Rhub", shape=1, val=0.1, units="m"))
    push!(input_data, VarData("Rtip", shape=1, val=2.0, units="m"))
    push!(input_data, VarData("radii", shape=num_radial, val=1., units="m"))
    push!(input_data, VarData("chord", shape=num_radial, val=1., units="m"))
    push!(input_data, VarData("theta", shape=num_radial, val=1., units="rad"))
    push!(input_data, VarData("v", shape=num_operating_points, val=1., units="m/s"))
    push!(input_data, VarData("omega", shape=num_operating_points, val=1., units="rad/s"))
    push!(input_data, VarData("pitch", shape=num_operating_points, val=0., units="rad"))

    # Declare the OpenMDAO outputs.
    output_data = Vector{VarData}()
    push!(output_data, VarData("phi", shape=(num_operating_points, num_radial), val=1.0, units="rad"))
    push!(output_data, VarData("thrust", shape=num_operating_points, val=1.0, units="N"))
    push!(output_data, VarData("torque", shape=num_operating_points, val=1.0, units="N*m"))
    push!(output_data, VarData("efficiency", shape=num_operating_points, val=1.0))
    push!(output_data, VarData("figure_of_merit", shape=num_operating_points, val=1.0))

    # Declare the OpenMDAO partial derivatives.
    ss_sizes = Dict(:i=>num_operating_points, :j=>num_radial, :k=>1)
    partials_data = Vector{PartialsData}()

    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:k])
    push!(partials_data, PartialsData("phi", "Rhub", rows=rows, cols=cols))
    push!(partials_data, PartialsData("phi", "Rtip", rows=rows, cols=cols))

    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:j])
    push!(partials_data, PartialsData("phi", "radii", rows=rows, cols=cols))
    push!(partials_data, PartialsData("phi", "chord", rows=rows, cols=cols))
    push!(partials_data, PartialsData("phi", "theta", rows=rows, cols=cols))

    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:i])
    push!(partials_data, PartialsData("phi", "v", rows=rows, cols=cols))
    push!(partials_data, PartialsData("phi", "omega", rows=rows, cols=cols))
    push!(partials_data, PartialsData("phi", "pitch", rows=rows, cols=cols))

    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:i, :j], wrt_ss=[:i, :j])
    push!(partials_data, PartialsData("phi", "phi", rows=rows, cols=cols))

    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:k])
    push!(partials_data, PartialsData("thrust", "Rhub", rows=rows, cols=cols))
    push!(partials_data, PartialsData("thrust", "Rtip", rows=rows, cols=cols))
    push!(partials_data, PartialsData("torque", "Rhub", rows=rows, cols=cols))
    push!(partials_data, PartialsData("torque", "Rtip", rows=rows, cols=cols))
    push!(partials_data, PartialsData("efficiency", "Rhub", rows=rows, cols=cols))
    push!(partials_data, PartialsData("efficiency", "Rtip", rows=rows, cols=cols))
    push!(partials_data, PartialsData("figure_of_merit", "Rhub", rows=rows, cols=cols))
    push!(partials_data, PartialsData("figure_of_merit", "Rtip", rows=rows, cols=cols))

    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:j])
    push!(partials_data, PartialsData("thrust", "radii", rows=rows, cols=cols))
    push!(partials_data, PartialsData("thrust", "chord", rows=rows, cols=cols))
    push!(partials_data, PartialsData("thrust", "theta", rows=rows, cols=cols))
    push!(partials_data, PartialsData("torque", "radii", rows=rows, cols=cols))
    push!(partials_data, PartialsData("torque", "chord", rows=rows, cols=cols))
    push!(partials_data, PartialsData("torque", "theta", rows=rows, cols=cols))
    push!(partials_data, PartialsData("efficiency", "radii", rows=rows, cols=cols))
    push!(partials_data, PartialsData("efficiency", "chord", rows=rows, cols=cols))
    push!(partials_data, PartialsData("efficiency", "theta", rows=rows, cols=cols))
    push!(partials_data, PartialsData("figure_of_merit", "radii", rows=rows, cols=cols))
    push!(partials_data, PartialsData("figure_of_merit", "chord", rows=rows, cols=cols))
    push!(partials_data, PartialsData("figure_of_merit", "theta", rows=rows, cols=cols))

    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:i])
    push!(partials_data, PartialsData("thrust", "v", rows=rows, cols=cols))
    push!(partials_data, PartialsData("thrust", "omega", rows=rows, cols=cols))
    push!(partials_data, PartialsData("thrust", "pitch", rows=rows, cols=cols))
    push!(partials_data, PartialsData("thrust", "thrust", rows=rows, cols=cols, val=-1.0))
    push!(partials_data, PartialsData("torque", "v", rows=rows, cols=cols))
    push!(partials_data, PartialsData("torque", "omega", rows=rows, cols=cols))
    push!(partials_data, PartialsData("torque", "pitch", rows=rows, cols=cols))
    push!(partials_data, PartialsData("torque", "torque", rows=rows, cols=cols, val=-1.0))
    push!(partials_data, PartialsData("efficiency", "v", rows=rows, cols=cols))
    push!(partials_data, PartialsData("efficiency", "omega", rows=rows, cols=cols))
    push!(partials_data, PartialsData("efficiency", "pitch", rows=rows, cols=cols))
    push!(partials_data, PartialsData("efficiency", "efficiency", rows=rows, cols=cols, val=-1.0))
    push!(partials_data, PartialsData("figure_of_merit", "v", rows=rows, cols=cols))
    push!(partials_data, PartialsData("figure_of_merit", "omega", rows=rows, cols=cols))
    push!(partials_data, PartialsData("figure_of_merit", "pitch", rows=rows, cols=cols))
    push!(partials_data, PartialsData("figure_of_merit", "figure_of_merit", rows=rows, cols=cols, val=-1.0))

    rows, cols = get_rows_cols(ss_sizes=ss_sizes, of_ss=[:i], wrt_ss=[:i, :j])
    push!(partials_data, PartialsData("thrust", "phi", rows=rows, cols=cols))
    push!(partials_data, PartialsData("torque", "phi", rows=rows, cols=cols))
    push!(partials_data, PartialsData("efficiency", "phi", rows=rows, cols=cols))
    push!(partials_data, PartialsData("figure_of_merit", "phi", rows=rows, cols=cols))

    return input_data, output_data, partials_data
end

# We'll define a `solve_nonlinear` function, since CCBlade.jl knows how to
# converge it's own residual.
function OpenMDAO.solve_nonlinear!(self::BEMTRotorCAComp, inputs, outputs)
    # Unpack all the options.
    num_operating_points = self.num_operating_points
    num_blades = self.num_blades
    num_radial = self.num_radial
    rho = self.rho
    mu = self.mu
    speedofsound = self.speedofsound
    af = self.airfoil_interp
    mach = self.mach
    reynolds = self.reynolds
    rotation = self.rotation
    tip = self.tip

    # Unpack the inputs.
    Rhub = inputs["Rhub"][1]
    Rtip = inputs["Rtip"][1]
    radii = inputs["radii"]
    chord = inputs["chord"]
    theta = inputs["theta"]
    v = inputs["v"]
    omega = inputs["omega"]
    pitch = inputs["pitch"]

    # Unpack the outputs.
    phi = outputs["phi"]
    thrust = outputs["thrust"]
    torque = outputs["torque"]
    efficiency = outputs["efficiency"]
    figure_of_merit = outputs["figure_of_merit"]

    # Create the CCBlade rotor struct. Same for each operating point and radial
    # element.
    T = typeof(Rhub)
    precone = zero(T)
    turbine = false
    rotor = CCBlade.Rotor(Rhub, Rtip, num_blades, precone, turbine, mach, reynolds, rotation, tip)

    # Create the CCBlade sections.
    sections = CCBlade.Section.(radii, chord, theta, Ref(af))

    for n in 1:num_operating_points
        # Create the CCBlade operating points.
        Vx = v[n]
        Vy = omega[n].*radii
        ops = CCBlade.OperatingPoint.(Vx, Vy, rho, pitch[n], mu, speedofsound)

        # Solve the BEMT equation.
        outs = CCBlade.solve.(Ref(rotor), sections, ops)

        # Get the thrust, torque, and efficiency.
        thrust[n], torque[n] = CCBlade.thrusttorque(rotor, sections, outs)
        efficiency[n], CT, CQ = CCBlade.nondim(thrust[n], torque[n], Vx, omega[n], rho, rotor, "propeller")
        if thrust[n] > zero(T)
            figure_of_merit[n], CT, CP = CCBlade.nondim(thrust[n], torque[n], Vx, omega[n], rho, rotor, "helicopter")
        else
            figure_of_merit[n] = zero(T)
        end

        # Get the local inflow angle, the BEMT implicit variable.
        phi[n, :] .= getproperty.(outs, :phi)
    end

    return nothing
end

#' Since we have a `solve_nonlinear` function, I don't think we necessarily need
#' an `apply_nonlinear` since CCBlade will converge the BEMT equation, not
#' OpenMDAO. But I think the `apply_nonlinear` will be handy for checking the
#' the partial derivatives of the `BEMTRotorCAComp` `Component`.
#+ results="hidden"
function OpenMDAO.apply_nonlinear!(self::BEMTRotorCAComp, inputs, outputs, residuals)
    # Unpack all the options.
    num_operating_points = self.num_operating_points
    num_blades = self.num_blades
    num_radial = self.num_radial
    rho = self.rho
    mu = self.mu
    speedofsound = self.speedofsound
    af = self.airfoil_interp
    mach = self.mach
    reynolds = self.reynolds
    rotation = self.rotation
    tip = self.tip

    # Unpack the inputs.
    Rhub = inputs["Rhub"][1]
    Rtip = inputs["Rtip"][1]
    radii = inputs["radii"]
    chord = inputs["chord"]
    theta = inputs["theta"]
    v = inputs["v"]
    omega = inputs["omega"]
    pitch = inputs["pitch"]

    # Create the CCBlade rotor struct. Same for each operating point and radial
    # element.
    T = typeof(Rhub)
    precone = zero(T)
    turbine = false
    rotor = CCBlade.Rotor(Rhub, Rtip, num_blades, precone, turbine, mach, reynolds, rotation, tip)

    # Create the CCBlade sections.
    sections = CCBlade.Section.(radii, chord, theta, Ref(af))

    # outs = Vector{CCBlade.Outputs{T}}(undef, num_radial)
    for n in 1:num_operating_points
        # Create the CCBlade operating points.
        Vx = v[n]
        Vy = omega[n].*radii
        ops = CCBlade.OperatingPoint.(Vx, Vy, rho, pitch[n], mu, speedofsound)

        # Get the residual of the BEMT equation. This should return a Vector of
        # length num_radial with each entry being a 2-length Tuple. First entry
        # in the Tuple is the residual (`Float64`), second is the CCBlade
        # `Output` struct`.
        Rs_and_outs = CCBlade.residual.(outputs["phi"][n, :], Ref(rotor), sections, ops)

        # Set the phi residual.
        residuals["phi"][n, :] .= getindex.(Rs_and_outs, 1)

        # Get the thrust, torque, and efficiency.
        outs = getindex.(Rs_and_outs, 2)

        thrust, torque = CCBlade.thrusttorque(rotor, sections, outs)
        efficiency, CT, CQ = CCBlade.nondim(thrust, torque, Vx, omega[n], rho, rotor, "propeller")
        if thrust > zero(T)
            figure_of_merit, CT, CP = CCBlade.nondim(thrust, torque, Vx, omega[n], rho, rotor, "helicopter")
        else
            figure_of_merit = zero(T)
        end

        # Set the residuals of the thrust, torque, and efficiency.
        residuals["thrust"][n] = thrust - outputs["thrust"][n]
        residuals["torque"][n] = torque - outputs["torque"][n]
        residuals["efficiency"][n] = efficiency - outputs["efficiency"][n]
        residuals["figure_of_merit"][n] = figure_of_merit - outputs["figure_of_merit"][n]
    end

end

# Now for the big one: the `linearize!` function will calculate the derivatives
# of the BEMT component residuals wrt the inputs and outputs. We'll use the
# Julia package ForwardDiff.jl to actually calculate the derivatives.
function OpenMDAO.linearize!(self::BEMTRotorCAComp, inputs, outputs, partials)
    # Unpack the options we'll need.
    num_operating_points = self.num_operating_points
    num_radial = self.num_radial

    # Unpack the inputs.
    Rhub = inputs["Rhub"][1]
    Rtip = inputs["Rtip"][1]
    radii = inputs["radii"]
    chord = inputs["chord"]
    theta = inputs["theta"]
    v = inputs["v"]
    omega = inputs["omega"]
    pitch = inputs["pitch"]

    x_ce = ComponentArray(Rhub=Rhub, Rtip=Rtip, radii=radii, chord=chord, theta=theta, v=v[1], omega=omega[1], pitch=pitch[1])

    # Unpack the output.
    phi = outputs["phi"]

    y_ce = ComponentArray(phi=phi[1, :])

    # Working arrays and configuration for ForwardDiff's Jacobian routine.
    x = self.x
    y = self.y
    J = self.J
    config = self.forwarddiff_config

    # These need to be transposed because of the differences in array layout
    # between NumPy and Julia. When I declare the partials above, they get set up
    # on the OpenMDAO side in a shape=(num_operating_points, num_radial), and
    # are then flattened. That gets passed to Julia. Since Julia uses column
    # major arrays, we have to reshape the array with the indices reversed, then
    # transpose them.
    dphi_dRhub = transpose(reshape(partials["phi", "Rhub"], num_radial, num_operating_points))
    dphi_dRtip = transpose(reshape(partials["phi", "Rtip"], num_radial, num_operating_points))
    dphi_dradii = transpose(reshape(partials["phi", "radii"], num_radial, num_operating_points))
    dphi_dchord = transpose(reshape(partials["phi", "chord"], num_radial, num_operating_points))
    dphi_dtheta = transpose(reshape(partials["phi", "theta"], num_radial, num_operating_points))
    dphi_dv = transpose(reshape(partials["phi", "v"], num_radial, num_operating_points))
    dphi_domega = transpose(reshape(partials["phi", "omega"], num_radial, num_operating_points))
    dphi_dpitch = transpose(reshape(partials["phi", "pitch"], num_radial, num_operating_points))
    dphi_dphi = transpose(reshape(partials["phi", "phi"], num_radial, num_operating_points))

    dthrust_dRhub = partials["thrust", "Rhub"]
    dthrust_dRtip = partials["thrust", "Rtip"]
    dtorque_dRhub = partials["torque", "Rhub"]
    dtorque_dRtip = partials["torque", "Rtip"]
    defficiency_dRhub = partials["efficiency", "Rhub"]
    defficiency_dRtip = partials["efficiency", "Rtip"]
    dfigure_of_merit_dRhub = partials["figure_of_merit", "Rhub"]
    dfigure_of_merit_dRtip = partials["figure_of_merit", "Rtip"]

    dthrust_dradii = transpose(reshape(partials["thrust", "radii"], num_radial, num_operating_points))
    dthrust_dchord = transpose(reshape(partials["thrust", "chord"], num_radial, num_operating_points))
    dthrust_dtheta = transpose(reshape(partials["thrust", "theta"], num_radial, num_operating_points))
    dtorque_dradii = transpose(reshape(partials["torque", "radii"], num_radial, num_operating_points))
    dtorque_dchord = transpose(reshape(partials["torque", "chord"], num_radial, num_operating_points))
    dtorque_dtheta = transpose(reshape(partials["torque", "theta"], num_radial, num_operating_points))
    defficiency_dradii = transpose(reshape(partials["efficiency", "radii"], num_radial, num_operating_points))
    defficiency_dchord = transpose(reshape(partials["efficiency", "chord"], num_radial, num_operating_points))
    defficiency_dtheta = transpose(reshape(partials["efficiency", "theta"], num_radial, num_operating_points))
    dfigure_of_merit_dradii = transpose(reshape(partials["figure_of_merit", "radii"], num_radial, num_operating_points))
    dfigure_of_merit_dchord = transpose(reshape(partials["figure_of_merit", "chord"], num_radial, num_operating_points))
    dfigure_of_merit_dtheta = transpose(reshape(partials["figure_of_merit", "theta"], num_radial, num_operating_points))

    dthrust_dv = partials["thrust", "v"]
    dthrust_domega = partials["thrust", "omega"]
    dthrust_dpitch = partials["thrust", "pitch"]
    dtorque_dv = partials["torque", "v"]
    dtorque_domega = partials["torque", "omega"]
    dtorque_dpitch = partials["torque", "pitch"]
    defficiency_dv = partials["efficiency", "v"]
    defficiency_domega = partials["efficiency", "omega"]
    defficiency_dpitch = partials["efficiency", "pitch"]
    dfigure_of_merit_dv = partials["figure_of_merit", "v"]
    dfigure_of_merit_domega = partials["figure_of_merit", "omega"]
    dfigure_of_merit_dpitch = partials["figure_of_merit", "pitch"]

    dthrust_dphi = transpose(reshape(partials["thrust", "phi"], num_radial, num_operating_points))
    dtorque_dphi = transpose(reshape(partials["torque", "phi"], num_radial, num_operating_points))
    defficiency_dphi = transpose(reshape(partials["efficiency", "phi"], num_radial, num_operating_points))
    dfigure_of_merit_dphi = transpose(reshape(partials["figure_of_merit", "phi"], num_radial, num_operating_points))

    for n in 1:num_operating_points
        # Put the inputs into the input array for ForwardDiff.
        x[:phi] .= phi[n, :]
        x[:Rhub] = Rhub
        x[:Rtip] = Rtip
        x[:radii] .= radii
        x[:chord] .= chord
        x[:theta] .= theta
        x[:v] = v[n]
        x[:omega] = omega[n]
        x[:pitch] = pitch[n]

        # Get the Jacobian.
        ForwardDiff.jacobian!(J, self.apply_nonlinear_forwarddiffable!, y, x, config)

        for r in 1:num_radial
            dphi_dphi[n, r] = J[:phi, :phi][r, r]
        end

        dphi_dRhub[n, :] .= J[:phi, :Rhub]
        dphi_dRtip[n, :] .= J[:phi, :Rtip]

        for r in 1:num_radial
            dphi_dradii[n, r] = J[:phi, :radii][r, r]
        end

        for r in 1:num_radial
            dphi_dchord[n, r] = J[:phi, :chord][r, r]
        end

        for r in 1:num_radial
            dphi_dtheta[n, r] = J[:phi, :theta][r, r]
        end

        dphi_dv[n, :] .= J[:phi, :v]
        dphi_domega[n, :] .= J[:phi, :omega]
        dphi_dpitch[n, :] .= J[:phi, :pitch]

        dthrust_dphi[n, :] .= J[:thrust, :phi]
        dthrust_dRhub[n] = J[:thrust, :Rhub]
        dthrust_dRtip[n] = J[:thrust, :Rtip]
        dthrust_dradii[n, :] .= J[:thrust, :radii]
        dthrust_dchord[n, :] .= J[:thrust, :chord]
        dthrust_dtheta[n, :] .= J[:thrust, :theta]
        dthrust_dv[n] = J[:thrust, :v]
        dthrust_domega[n] = J[:thrust, :omega]
        dthrust_dpitch[n] = J[:thrust, :pitch]

        dtorque_dphi[n, :] .= J[:torque, :phi]
        dtorque_dRhub[n] = J[:torque, :Rhub]
        dtorque_dRtip[n] = J[:torque, :Rtip]
        dtorque_dradii[n, :] .= J[:torque, :radii]
        dtorque_dchord[n, :] .= J[:torque, :chord]
        dtorque_dtheta[n, :] .= J[:torque, :theta]
        dtorque_dv[n] = J[:torque, :v]
        dtorque_domega[n] = J[:torque, :omega]
        dtorque_dpitch[n] = J[:torque, :pitch]

        defficiency_dphi[n, :] .= J[:eff, :phi]
        defficiency_dRhub[n] = J[:eff, :Rhub]
        defficiency_dRtip[n] = J[:eff, :Rtip]
        defficiency_dradii[n, :] .= J[:eff, :radii]
        defficiency_dchord[n, :] .= J[:eff, :chord]
        defficiency_dtheta[n, :] .= J[:eff, :theta]
        defficiency_dv[n] = J[:eff, :v]
        defficiency_domega[n] = J[:eff, :omega]
        defficiency_dpitch[n] = J[:eff, :pitch]

        dfigure_of_merit_dphi[n, :] .= J[:figure_of_merit, :phi]
        dfigure_of_merit_dRhub[n] = J[:figure_of_merit, :Rhub]
        dfigure_of_merit_dRtip[n] = J[:figure_of_merit, :Rtip]
        dfigure_of_merit_dradii[n, :] .= J[:figure_of_merit, :radii]
        dfigure_of_merit_dchord[n, :] .= J[:figure_of_merit, :chord]
        dfigure_of_merit_dtheta[n, :] .= J[:figure_of_merit, :theta]
        dfigure_of_merit_dv[n] = J[:figure_of_merit, :v]
        dfigure_of_merit_domega[n] = J[:figure_of_merit, :omega]
        dfigure_of_merit_dpitch[n] = J[:figure_of_merit, :pitch]
    end

    return nothing
end

# Disable checking for guess_nonlinear and apply_nonlinear functions.
OpenMDAO.detect_guess_nonlinear(::Type{<:BEMTRotorCAComp}) = false
OpenMDAO.detect_apply_linear(::Type{<:BEMTRotorCAComp}) = false
