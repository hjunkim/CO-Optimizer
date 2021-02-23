# Transpiler (TBD)
Clang based (ASTMatcher, Rewriter) source-to-source transpiler for CUDA applications.

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

2. Build them
	* ``cd ../../;mkdir build;cd build``
	* ``cmake -G "Unix Makefiles" ../llvm``
	* ``make -j 16;sudo make install``

## Usage
* ``./llvm-project/build/bin/throttling {cuda_program}.cu -- --cuda-device-only --cuda-path={path/to/cuda} --cuda-gpu-arch={sm_xx} [options]``
Some common options:
	* ``{cuda_program}.cu`` --- your program
	* ``--cuda-device-only`` --- will run only analysis/translate device code
	* ``--cuda-path=`` --- CUDA path (ex: /usr/local/cuda)
	* ``--cuda-gpu-arch=sm_xx`` --- CUDA architecture (ex: Titan V, V100: sm\_70)

### Run Options
* Throttling Transpiler:
	* ``--csize=<int>``               - <csize> : L1 cache size of the GPU (default: 32 KB)
	* ``--nblks=<int>``               - <nblks> : # of thread blocks per SM (default: 4 blks)
	* ``--tbsize=<int>``              - <tbsize> : thread block size (default: 8 warps)
	* ``--extra-arg=<string>``        - Additional argument to append to the compiler command line
	* ``--extra-arg-before=<string>`` - Additional argument to prepend to the compiler command line
	* ``-p=<string>``                 - Build path
