using ComponentArrays
using ConcreteStructs
using SparseArrays
using SparseDiffTools
using Symbolics
using ForwardDiff

function g2nl(A::ComponentVector)

    # A ComponentVector always has one axis, but getaxes will return a length-1
    # tuple, hence the [1] indexing. This returns an Axis object, which contains
    # the information for the mapping from a name to the indices associated with
    # the name.
    ax = getaxes(A)[1]
    # And this takes the Axis object and turns it into a named tuple.
    idxmap = ComponentArrays.indexmap(ax)
    # Now keys(idxmap) returns all the names of the component array's axis.

    global2namelocal = []
    for name in keys(idxmap)
      v = idxmap[name]
      i = 1
      if typeof(v) <: ComponentArrays.ViewAxis
        for _ in ComponentArrays.viewindex(v)
          push!(global2namelocal, [name, i])
          i += 1
        end
      else
        for _ in v
          push!(global2namelocal, [name, i])
          i += 1
        end
      end
    end

    return global2namelocal
end

@concrete struct SymbolicsSimpleComp
    num_operating_points
    num_radial
    apply_nonlinear_forwarddiffable!
    x
    y
    J
    # forwarddiff_config
    om_rows_cols
end

function SymbolicsSimpleComp()

    num_operating_points = 2
    num_radial = 3

    function apply_nonlinear_forwarddiffable!(y, x)
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

        for n in 1:num_operating_points
            # Vx = v[n]
            # Vy = omega[n].*radii
            # Rs = @. Rhub + 2*Rtip + 3*radii + 4*chord + 5*theta + 6*Vx + 7*Vy + 8*pitch[n] + 9*phi[n, :]
            # y[:phi][n, :] .= Rs
            for r in 1:num_radial
                Rphi = Rhub + 2*Rtip + 3*radii[r] + 4*chord[r] + 5*theta[r] + 6*v[n] + 7*omega[n]*radii[r] + 8*pitch[n] + 9*phi[r, n]
                y[:phi][r, n] = Rphi
            end
            y[:thrust][n] = y[:phi][1, n] + y[:phi][2, n] + y[:phi][3, n]

            y[:torque][n] = 2*y[:thrust][n]
            y[:eff][n] = 3*y[:thrust][n]
            y[:figure_of_merit][n] = 4*y[:thrust][n]
        end

        return nothing
    end

    # Initialize the input and output vectors needed by ForwardDiff.jl. (The
    # ForwardDiff.jl inputs include phi, but that's an OpenMDAO output.)
    X = ComponentArray(
        Rhub=0.2, Rtip=1.0,
        radii=collect(range(0.25, 0.95, length=num_radial)), chord=0.1*ones(num_radial), theta=0.1*ones(num_radial),
        v=ones(num_operating_points), omega=ones(num_operating_points), pitch=ones(num_operating_points),
        phi=0.1.*ones(num_radial, num_operating_points))
    Y = ComponentArray(
        thrust=zeros(num_operating_points), torque=zeros(num_operating_points),
        eff=zeros(num_operating_points), figure_of_merit=zeros(num_operating_points),
        phi=zeros(num_radial, num_operating_points))

    # Create some symbolic variables.
    @variables Rhub_s Rtip_s
    @variables radii_s1 radii_s2 radii_s3
    @variables chord_s1 chord_s2 chord_s3
    @variables theta_s1 theta_s2 theta_s3
    @variables v_s1 v_s2 omega_s1 omega_s2 pitch_s1 pitch_s2
    @variables phi_s11 phi_s21 phi_s31
    @variables phi_s12 phi_s22 phi_s32
    @variables Rphi_s11 Rphi_s21 Rphi_s31
    @variables Rphi_s12 Rphi_s22 Rphi_s32
    @variables thrust_s1 thrust_s2 torque_s1 torque_s2
    @variables eff_s1 eff_s2 figure_of_merit_s1 figure_of_merit_s2

    phi_s = [phi_s11 phi_s12;
             phi_s21 phi_s22;
             phi_s31 phi_s32]
    Rphi_s = [Rphi_s11 Rphi_s12;
              Rphi_s21 Rphi_s22;
              Rphi_s31 Rphi_s32]

    # Put the symbolic variables into a ComponentArray.
    X_s = ComponentArray(
        Rhub=Rhub_s, Rtip=Rtip_s,
        radii=[radii_s1, radii_s2, radii_s3], chord=[chord_s1, chord_s2, chord_s3], theta=[theta_s1, theta_s2, theta_s3],
        v=[v_s1, v_s2], omega=[omega_s1, omega_s2], pitch=[pitch_s1, pitch_s2],
        phi=phi_s)
    Y_s = ComponentArray(
        phi=Rphi_s, thrust=[thrust_s1, thrust_s2], torque=[torque_s1, torque_s2],
        eff=[eff_s1, eff_s2], figure_of_merit=[figure_of_merit_s1, figure_of_merit_s2])

    # Call apply_nonlinear_forwarddiffable! with the symbolic variables.
    apply_nonlinear_forwarddiffable!(Y_s, X_s)

    # Now we want sparsity.
    sparsity_pattern = Symbolics.jacobian_sparsity(Y_s, X_s)
    @show sparsity_pattern

    # Now I have a sparsity pattern that I want to use to make a sparse
    # ComponentMatrix. I think I can do that with:
    J = Float64.(sparse(sparsity_pattern))

    # Get the colors.
    colors = matrix_colors(J)

    # Now I want a sparse ComponentMatrix.
    J = ComponentMatrix(J, getaxes(Y)..., getaxes(X)...)

    # This is a config that will store the colors, etc.
    # config = ForwardColorJacCache(apply_nonlinear_forwarddiffable!,X,dx=Y,colorvec=colors,sparsity=sparsity_pattern)

    # Get a mapping from the global index (index into the full Jacobian) to the
    # variable name and local index.
    input_map = g2nl(X)
    output_map = g2nl(Y)

    # Get the sparsity data for each OpenMDAO variable.
    om_rows_cols = Dict{Tuple{Symbol, Symbol}, Tuple{Vector{Int}, Vector{Int}}}()
    for (r, c, _) in zip(findnz(getdata(J))...)
        rname, ridx = output_map[r]
        cname, cidx = input_map[c]
        if ! ((rname, cname) in keys(om_rows_cols))
            om_rows_cols[(rname, cname)] = (Int[], Int[])
        end
        push!(om_rows_cols[(rname, cname)][1], ridx)
        push!(om_rows_cols[(rname, cname)][2], cidx)
   end

    return SymbolicsSimpleComp(num_operating_points, num_radial, apply_nonlinear_forwarddiffable!, X, Y, J, om_rows_cols)
end

