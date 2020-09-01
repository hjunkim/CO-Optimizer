# Transpiler
Clang tool based (ASTMatcher, Rewriter) source-to-source transpiler for CUDA.

## Build

First, checkout llvm, clang, and transpiler

```bash
$ cd where-you-want-llvm-to-live
#... downloading out llvm repository
$ git clone http://github.com/llvm-mirror/llvm

#... switching to release 8.0
$ cd llvm
$ git checkout release_80

#... downloading clang
#... llvm/tools
$ cd tools 
$ git clone http://github.com/llvm-mirror/clang

#... switching to release 8.0
#... llvm/tools/clang
$ cd clang 
$ git checkout release_80

#... downloading transpiler
#... llvm/tools/clang/tools
$ cd .. 
$ git clone github.com/hjunkim/transpiler-throttling
```

Then, building the llvm, clang, and transpiler

```bash
$ cd ~
$ mkdir llvm-build
$ cd llvm-build
$ cmake -G "Unix Makefiles" ../llvm
#... N= 4 or 8 or 16... multi-thread
$ make -j N
$ sudo make install
```

### Tested on Machines with following settings

```bash
# Ubuntu 18.04, gcc/g++ 5.5 is tested
$ sudo apt install build-essential gcc-5 g++-5 cmake
# boost/tokenizer library is used
$ sudo apt install libboost-all-dev
```

## Run

```bash
#... transpiler는 다음 경로에 있음 llvm-project/build/bin/throttling
$ transpiler cuda_program.cu -- --cuda-device-only --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_xx
```

- '--' 이후 flag/option들을 정의
- --cuda-device-only: CUDA 프로그램의 host/device 코드들 중 device (cuda kernel function) 만을 대상으로 수행한다고 명시
- --cuda-path=/usr/local/cuda: CUDA 설치 path
- --cuda-gpu-arch=sm\_xx: CUDA architecture에 맞게 작성 (Titan V, V100: sm\_70)


### Run Options
Throttling Transpiler:

  --csize=<int>               - <csize> : set L1 cache size of a GPU (default: 32 KB)
  --extra-arg=<string>        - Additional argument to append to the compiler command line
  --extra-arg-before=<string> - Additional argument to prepend to the compiler command line
  --nblks=<int>               - <nblks> : set # of thread blocks per SM (default: 4 blks)
  -p=<string>                 - Build path
  --tbsize=<int>              - <tbsize> : set thread block size (default: 8 warps)
