# Transpiler (TBD)
LLVM/Clang based (ASTMatcher, Rewriter) source-to-source transpiler for CUDA applications.

If you use or build on this tool, please cite the following papers.
- Throttling ([ICPP'19](https://dl.acm.org/doi/10.1145/3337821.3337886)), Preloading ([CCPE'20](https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.5512))

## Getting the Source Code and Building Transpiler
This is an example work-flow and configuration to get and build the Transpiler.

0. Tested with following setups
	* Ubuntu 18.04, cmake-3.10.2, gcc/g++-5
	* ``sudo apt install gcc-5 g++-5 cmake``
	* ``sudo apt install libboost-all-dev``

1. Checkout llvm, clang, and transpiler
	* llvm
		* ``git clone http://github.com/llvm-mirror/llvm``
		* ``cd llvm-project``
		* ``git checkout release_80``

	* clang
		* ``git clone http://github.com/llvm-mirror/clang``
		* ``cd clang;``
		* ``git checkout release_80``

	* transpiler
		* ``cd tools``
		* ``git clone github.com/hjunkim/transpiler-throttling``
		* ``git clone github.com/hjunkim/transpiler-preloading``

2. Build them
	* Add Transpiler repositories to CMakeLists.txt
		* ``add_clang_subdirectory(transpiler-throttling)``
		* ``add_clang_subdirectory(transpiler-preloading)
	* ``cd ../../;mkdir build;cd build``
	* ``cmake -G "Unix Makefiles" ../llvm``
	* ``make -j 16;sudo make install``

## Usage
* ``{bin} {cuda_program}.cu -- --cuda-device-only --cuda-path={path/to/cuda} --cuda-gpu-arch={sm_xx} [options]``
	* ``{bin}`` --- ``./llvm-project/build/bin/{throttling/preloading}``
	* ``{cuda_program}.cu`` --- your program
	* ``--cuda-device-only`` --- will run only analysis/translate for the device code
	* ``--cuda-path=`` --- installed CUDA path (ex: /usr/local/cuda)
	* ``--cuda-gpu-arch=sm_xx`` --- [CUDA architecture](https://en.wikipedia.org/wiki/CUDA) (ex: Titan V, V100: sm\_70)

### Run Options
* Throttling:
	* ``--csize=<int>``               - <csize> : L1 cache size of the GPU (default: 32 KB)
	* ``--nblks=<int>``               - <nblks> : # of thread blocks per SM (default: 4 blks)
	* ``--tbsize=<int>``              - <tbsize> : thread block size (default: 8 warps)
	* ``--extra-arg=<string>``        - Additional argument to append to the compiler command line
	* ``--extra-arg-before=<string>`` - Additional argument to prepend to the compiler command line
	* ``-p=<string>``                 - Build path

* Preloading	
	* ``--extra-arg=<string>``        - Additional argument to append to the compiler command line
	* ``--extra-arg-before=<string>`` - Additional argument to prepend to the compiler command line
	* ``-p=<string>``                 - Build path
	* ``--prdsize=<string>``          - <prdsize> : set preloading size (default: 1)
	* ``--tbsize=<string>``           - <tbsize> : set thread block size (default: 256)
