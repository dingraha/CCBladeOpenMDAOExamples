from setuptools import setup

setup(name='ccblade_openmdao_examples',
      version='0.0.1',
      description='Examples of using CCBlade.jl in OpenMDAO optimizations',
      install_requires=['openmdao', 'julia', 'matplotlib'],
      author='Andrew Ning and Daniel Ingraham',
      zip_safe=False)
