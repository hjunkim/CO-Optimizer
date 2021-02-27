# CO-Optimizer: Code-level On-chip memory Optimizer
This project gives an opportunity to optimize space-limited on-chip memories (L1, shared memory) of your CUDA applications.

If you use or build on this tool, please cite the following papers.
- Throttling ([ICPP'19](https://dl.acm.org/doi/10.1145/3337821.3337886)), Preloading ([CCPE'20](https://onlinelibrary.wiley.com/doi/full/10.1002/cpe.5512))

## Getting the Source Code and Building the On-chip Memory Optimizer
This is an example work-flow and configuration to get and build the Transpiler.

0. Tested with following setups
	* Ubuntu 18.04, cmake-3.10.2, gcc/g++-5
	* ``sudo apt install gcc-5 g++-5 cmake``
	* ``sudo apt install libboost-all-dev``
	* Benchmark -- [PolyBench/GPU](http://web.cse.ohio-state.edu/~pouchet.2/software/polybench/GPU/) and [Rodinia](http://www.cs.virginia.edu/rodinia/doku.php)

1. Checkout llvm, clang, and transpiler
	* llvm
		* ``git clone http://github.com/llvm-mirror/llvm``
		* ``cd llvm``
		* ``git checkout release_80``

	* clang
		* ``cd tools;``
		* ``git clone http://github.com/llvm-mirror/clang``
		* ``cd clang;``
		* ``git checkout release_80``

	* transpiler
		* ``cd tools``
		* ``git clone github.com/hjunkim/Transpiler``

2. Build them
	* Add the Transpiler repository to ``llvm/tools/clang/tools/CMakeLists.txt``
		* ``add_clang_subdirectory(Transpiler)``
	* ``cd ../../../../;mkdir build;cd build``
	* ``cmake -G "Unix Makefiles" ../llvm``
	* ``make -j 16;sudo make install``

## Usage
### Run
* ``{bin} {cuda_program}.cu [Run Options] -- --cuda-device-only --cuda-path={path/to/cuda} --cuda-gpu-arch={sm_xx}``
	* ``{bin}`` --- ``./build/bin/{throttling/preloading}``
	* ``{cuda_program}.cu`` --- your target CUDA program
	* ``--cuda-device-only`` --- will run analysis/translate for the device code
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
	* ``--prdsize=<string>``          - <prdsize> : set preloading size (default: 1)
	* ``--tbsize=<string>``           - <tbsize> : set thread block size (default: 8 warps)
	* ``--extra-arg=<string>``        - Additional argument to append to the compiler command line
	* ``--extra-arg-before=<string>`` - Additional argument to prepend to the compiler command line
	* ``-p=<string>``                 - Build path
