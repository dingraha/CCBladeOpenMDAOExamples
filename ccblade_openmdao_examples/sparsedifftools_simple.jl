using ComponentArrays
using ConcreteStructs
using SparseArrays
using SparseDiffTools
using SparsityDetection
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
@concrete struct SDTSimpleComp
    num_operating_points
    num_radial
    apply_nonlinear_forwarddiffable!
    x
    y
    J
    forwarddiff_config
    om_rows_cols
end

function SDTSimpleComp(; num_operating_points, num_radial)

    function apply_nonlinear_forwarddiffable!(y, x)
        T = eltype(x)

        # Start unpack the inputs.
        i = 1
        Rhub = x[i]
        i += 1
        Rtip = x[i]
        i += 1
        # radii = x[i:i+num_radial-1]
        radii = zeros(T, num_radial)
        # radii .= x[i:i+num_radial-1]
        for k in eachindex(radii)
            radii[k] = x[i+k-1]
        end
        i += num_radial
        # chord = x[i:i+num_radial-1]
        chord = zeros(T, num_radial)
        for k in eachindex(chord)
            chord[k] = x[i+k-1]
        end
        # chord .= x[i:i+num_radial-1]
        i += num_radial
        # theta = x[i:i+num_radial-1]
        theta = zeros(T, num_radial)
        # theta .= x[i:i+num_radial-1]
        for k in eachindex(theta)
            theta[k] = x[i+k-1]
        end
        i += num_radial

        j = 1  # output index

        ii = i
        jj = j
        for n in 1:num_operating_points
            v = x[ii]
            ii += num_operating_points  # skip over the rest of v
            omega = x[ii]
            ii += num_operating_points
            pitch = x[ii]
            ii += num_operating_points
            # Get to the begining of the phi block.
            ii -= (n - 1)
            # Skip over the operating points we've done already.
            ii += (n - 1)*num_radial
            # phi = x[ii:ii+num_radial-1]
            phi = zeros(T, num_radial)
            # phi .= x[ii:ii+num_radial-1]
            for k in eachindex(phi)
                phi[k] = x[ii+k-1]
            end

            Vx = v

            # Vy = omega.*radii
            Vy = zeros(T, num_radial)
            for k in eachindex(Vy)
                Vy[k] = omega*radii[k]
            end

            # Rs = @. Rhub + 2*Rtip + 3*radii + 4*chord + 5*theta + 6*Vx + 7*Vy + 8*pitch + 9*phi
            Rs = zeros(T, num_radial)
            for k in eachindex(Rs)
                Rs[k] = Rhub + 2*Rtip + 3*radii[k] + 4*chord[k] + 5*theta[k] + 6*Vx + 7*Vy[k] + 8*pitch + 9*phi[k]
            end

            # Put the outputs in the output array.
            # println("jj = $jj, thrust")
            y[jj] = sum(Rs)
            jj += num_operating_points
            # println("jj = $jj, torque")
            y[jj] = 2*sum(Rs)
            jj += num_operating_points
            # println("jj = $jj, eff")
            y[jj] = 3*sum(Rs)
            jj += num_operating_points
            # println("jj = $jj, figure_of_merit")
            y[jj] = 4*sum(Rs)
            jj += num_operating_points
            # Get to the beginning of the phi block.
            jj -= (n - 1)
            # Skip over the operating points we've done already.
            jj += (n - 1)*num_radial
            # println("hi")
            # println("jj = $jj:$(jj+num_radial-1), phi")
            # y[jj:jj+num_radial-1] .= Rs
            for k in eachindex(Rs)
                y[jj+k-1] = Rs[k]
            end

            ii = i + 1
            jj = j + 1
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

    apply_nonlinear_forwarddiffable!(getdata(Y), getdata(X))
    @show Y

    # Now we want sparsity. This should work, but who knows.
    sparsity_pattern = jacobian_sparsity(apply_nonlinear_forwarddiffable!, Y, X)
    @show sparsity_pattern

    # Now I have a sparsity pattern that I want to use to make a sparse
    # ComponentMatrix. I think I can do that with:
    J = Float64.(sparse(sparsity_pattern))

    # Get the colors.
    colors = matrix_colors(J)

    # Now I want a sparse ComponentMatrix.
    J = ComponentMatrix(J, getaxes(Y)..., getaxes(X)...)

    # This is a config that will store the colors, etc.
    config = ForwardColorJacCache(apply_nonlinear_forwarddiffable!,X,dx=Y,colorvec=colors,sparsity=sparsity_pattern)

    # Now I should be able to call
    #     forwarddiff_color_jacobian!(getdata(J_cm), apply_nonlinear_forwarddiffable!, config)
    # later on in linearize.
    # But the key is to somehow store the rows and cols. How do I get those? I
    # can get the "global" rows and cols with
    #
    # rows, cols, vals = findnz(getdata(J_ca))
    #
    # But I need the rows and cols for each sub-jacobian. I need to somehow
    # invert what the axes in the ComponentArrays do: I need to take a global
    # row or column and identify which name and index that refers to. So right
    # now I can do this:
    #
    #   X_axes = getaxes(X)[1]  # getaxes(X) returns a length-1 tuple of axes
    #   X_axes[:Rhub]
    #   ComponentIndex(1, NullAxis())
    #   X_axes[:Rhub].idx[1]
    #   1
    #   X_axes[:pitch]
    #   ComponentIndex(16:17, FlatAxis())
    #   X_axes[:pitch].idx
    #   16:17
    #
    # What I want to do is the opposite: I want to take a global index, and
    # return a name and a local index. How do I do that?
    #
    # julia> keys(ComponentArrays.indexmap(X_axes))
    # (:Rhub, :Rtip, :radii, :chord, :theta, :v, :omega, :pitch, :phi)
    #
    # OK, so I think the simplest thing to do would be to loop over the entire X
    # and Y to construct the mapping.
    # julia> X_axes_idxmap = ComponentArrays.indexmap(X_axes)
    # (Rhub = 1, Rtip = 2, radii = 3:5, chord = 6:8, theta = 9:11, v = 12:13, omega = 14:15, pitch = 16:17, phi = ViewAxis(18:23, ShapedAxis(
    # (3, 2), NamedTuple())))
    # julia> for k in keys(X_axes_idxmap)
    #          v = X_axes_idxmap[k]
    #          i = 1
    #          for foo in v
    #            println("$foo: $k $i")
    #            i += 1
    #          end
    #        end
    # 1: Rhub 1
    # 2: Rtip 1
    # 3: radii 1
    # 4: radii 2
    # 5: radii 3
    # 6: chord 1
    # 7: chord 2
    # 8: chord 3
    # 9: theta 1
    # 10: theta 2
    # 11: theta 3
    # 12: v 1
    # 13: v 2
    # 14: omega 1
    # 15: omega 2
    # 16: pitch 1
    # 17: pitch 2
    # ERROR: MethodError: no method matching iterate(::ViewAxis{18:23,NamedTuple(),ShapedAxis{(3, 2),NamedTuple()}})
    # Closest candidates are:
    #   iterate(::ChainRulesCore.One) at /home/dingraha/.julia/packages/ChainRulesCore/7d1hl/src/differentials/one.jl:13
    #   iterate(::ChainRulesCore.One, ::Any) at /home/dingraha/.julia/packages/ChainRulesCore/7d1hl/src/differentials/one.jl:14
    #   iterate(::Cmd) at process.jl:638
    #   ...
    # Stacktrace:
    #  [1] top-level scope at ./REPL[273]:4
    #
    # Lame, choking on the phi variable, which is 2D:
    #
    # julia> X_axes_idxmap[:phi]
    # ViewAxis(18:23, ShapedAxis((3, 2), NamedTuple()))
    # julia> ComponentArrays.viewindex(X_axes_idxmap[:phi])
    # 18:23
    #
    # julia> for k in keys(X_axes_idxmap)
    #          v = X_axes_idxmap[k]
    #          i = 1
    #          if typeof(v) <: ComponentArrays.ViewAxis
    #            for foo in ComponentArrays.viewindex(v)
    #              println("$foo: $k $i")
    #              i += 1
    #            end
    #          else
    #            for foo in v
    #              println("$foo: $k $i")
    #              i += 1
    #            end
    #          end
    #        end
    # 1: Rhub 1
    # 2: Rtip 1
    # 3: radii 1
    # 4: radii 2
    # 5: radii 3
    # 6: chord 1
    # 7: chord 2
    # 8: chord 3
    # 9: theta 1
    # 10: theta 2
    # 11: theta 3
    # 12: v 1
    # 13: v 2
    # 14: omega 1
    # 15: omega 2
    # 16: pitch 1
    # 17: pitch 2
    # 18: phi 1
    # 19: phi 2
    # 20: phi 3
    # 21: phi 4
    # 22: phi 5
    # 23: phi 6

    # julia> global2namelocal = []
    #        for k in keys(X_axes_idxmap)
    #          v = X_axes_idxmap[k]
    #          i = 1
    #          if typeof(v) <: ComponentArrays.ViewAxis
    #            for foo in ComponentArrays.viewindex(v)
    #              push!(global2namelocal, [k, i])
    #              i += 1
    #            end
    #          else
    #            for foo in v
    #              push!(global2namelocal, [k, i])
    #              i += 1
    #            end
    #          end
    #        end

    # julia> global2namelocal
    # 23-element Array{Any,1}:
    #  Any[:Rhub, 1]
    #  Any[:Rtip, 1]
    #  Any[:radii, 1]
    #  Any[:radii, 2]
    #  Any[:radii, 3]
    #  Any[:chord, 1]
    #  Any[:chord, 2]
    #  Any[:chord, 3]
    #  Any[:theta, 1]
    #  Any[:theta, 2]
    #  Any[:theta, 3]
    #  Any[:v, 1]
    #  Any[:v, 2]
    #  Any[:omega, 1]
    #  Any[:omega, 2]
    #  Any[:pitch, 1]
    #  Any[:pitch, 2]
    #  Any[:phi, 1]
    #  Any[:phi, 2]
    #  Any[:phi, 3]
    #  Any[:phi, 4]
    #  Any[:phi, 5]
    #  Any[:phi, 6]

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

    return SDTSimpleComp(num_operating_points, num_radial, apply_nonlinear_forwarddiffable!, X, Y, J, config, om_rows_cols)
end

