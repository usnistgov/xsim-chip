========== COMPILING CYTHON EXTENSION: IMG2STL ==========

Activate the virtual environment which contains cython in your terminal
(or command prompt). An example of this using the conda package manager, with
an environment called cython_dev, would look like this:

  conda activate cython_dev

Ensure you have a C compiler installed on your system. For Windows, you will
need to install Visual Studio (with the C/C++ module checked). For Linux
systems, you will need to install gcc. For MacOS, you will need to install
clang.

In my experience, you do not need to manually initialize your compiler
environment script within this same prompt. In fact, for Windows, I tried to
initialize Visual Studio's compiler environment within the miniforge prompt,
which succeeded, but led to errors come time to compile the Python extensions
using Cython. So, my recommendation is to first try compiling the Python/Cython
extension without any initialization of your compiler environment. If it works,
great! If not, only then trying initializing your compiler environment, which
simply sets a number of system path variables in your current terminal. 

The next step is to have Cython convert the .pyx file into a C file which
automatically gets compiled using the setup.py file. The following command does
achieves this for a Windows environment,

  python setup.py build_ext --inplace

If you have more than one compiler on your system, you may need to specify the
compiler. For example,

  python setup.py build_ext --inplace --compiler=msvc

If all goes well, this will create a .pyd file in this source directory. This
.pyd module can be imported in a Python script like so,

  import img2stl

There is a test script that you can run to see if your build worked,

  python test_cython_extension.py
