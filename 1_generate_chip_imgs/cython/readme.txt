Activate the conda environment with cython:

  >> conda activate cython_dev

Do NOT manually initialize the Visual Studio environment. It is already
configured correctly in the miniforge environment...

Have Cython convert the .pyx file into a C file which automatically
gets compiled using the setup.py file:

  >> python setup.py build_ext --inplace --compiler=msvc

If all goes well, this will create a .pyd module. Then, the import
the module wihtin a Python script to see if it works:

  >> python main_file.py