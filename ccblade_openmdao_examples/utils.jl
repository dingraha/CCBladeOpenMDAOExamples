function get_airfoil(; af_fname, cr75, Re_exp)
    (info, Re, Mach, alpha, cl, cd) = CCBlade.parsefile(af_fname, false)

    # Extend the angle of attack with the Viterna method.
    (alpha, cl, cd) = CCBlade.viterna(alpha, cl, cd, cr75)
    af = CCBlade.AlphaAF(alpha, cl, cd, info, Re, Mach)

    # Reynolds number correction. The 0.6 factor seems to match the NACA 0012
    # drag data from airfoiltools.com.
    reynolds = CCBlade.SkinFriction(Re, Re_exp)

    # Mach number correction.
    mach = CCBlade.PrandtlGlauert()

    # Rotational stall delay correction. Need some parameters from the CL curve.
    m, alpha0 = CCBlade.linearliftcoeff(af, 1.0, 1.0)  # dummy values for Re and Mach
    # Create the Du Selig and Eggers correction.
    rotation = CCBlade.DuSeligEggers(1.0, 1.0, 1.0, m, alpha0)

    # The usual hub and tip loss correction.
    tip = CCBlade.PrandtlTipHub()

    return af, mach, reynolds, rotation, tip
end

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
