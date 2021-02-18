using Pkg
Pkg.add("CCBlade")
Pkg.add("ConcreteStructs")
Pkg.add("ForwardDiff")
# OpenMDAO.jl isn't a Registered Julia Package®™©, so need to specify the url.
Pkg.add(url="https://github.com/byuflowlab/OpenMDAO.jl.git")
Pkg.add("PyCall")
Pkg.build("PyCall")
using CCBlade
using ConcreteStructs
using ForwardDiff
using OpenMDAO
using PyCall
