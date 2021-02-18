import os
import subprocess
import julia

script_dir = os.path.dirname(os.path.realpath(__file__))


# This was taken from https://github.com/SciML/diffeqpy.
def install():
    """
    Install Julia packages required for ccblade_openmdao_examples.
    """
    julia.install()
    subprocess.check_call(['julia', os.path.join(script_dir, 'install.jl')])
