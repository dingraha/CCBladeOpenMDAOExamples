using ComponentArrays
using ConcreteStructs
using SparseDiffTools
using SparsityDetection

@concrete struct Test1
    num_operating_points
    num_radial
    compute!
    x
    y
    J
    forwarddiff_config
end

function Test1(num_operating_points, num_radial)
    function
end

function compute!(self::Test1, inputs, outputs)
    Rhub = inputs["Rhub"]
    Rtip = inputs["Rtip"]
    radii = inputs["radii"]
    phi = inputs["phi"]
    thrust = outputs["thrust"]
    R_phi = outputs["phi"]

    for n in 1:self.num_operating_points
    end
        
end
