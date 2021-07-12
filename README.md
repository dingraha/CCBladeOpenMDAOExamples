# CCBlade.jl Optimization Examples with OpenMDAO

This repository contains a couple of examples of using
[CCBlade.jl](https://github.com/byuflowlab/CCBlade.jl), a blade element momentum
theory code, within an [OpenMDAO](https://openmdao.org/) optimization.

## Installation
* Make sure you have Julia installed. I've had good luck with the [official
  instructions](https://julialang.org/downloads/platform/).
* (Optional) Maybe create a Python virtual environment to keep things tidy, and
  a new Julia "Project" by setting the envirnoment variable `JULIA_PROJECT` to a
  directory of your choice (see Julia's [Pkg](https://pkgdocs.julialang.org/v1/)
  for more information).
* Clone/download this package, then install it:

    ```bash
    $ pip install -e <./path/to/CCBladeOpenMDAOExamples>
    ```

  which should take care of the Python dependencies. Then open up your Python
  prompt and do this:

    ```python
    import ccblade_openmdao_examples
    ccblade_openmdao_examples.install()
    ```

  That should download and install all the Julia dependencies.

## Running it
There are two scripts you can run, both in `scripts`. `run.py` uses code that
shows how I used to set up OpenMDAO optimizations with Julia, and `run_ca.py`
runs a much better version that uses a Julia package called
[ComponentArrays](https://github.com/jonniedie/ComponentArrays.jl). The scripts
are currently set up to use the proprietary SNOPT optimizer via
[pyOptSparse](https://github.com/mdolab/pyoptsparse). I'm told it will also work
with [IPOPT](https://github.com/coin-or/Ipopt).
