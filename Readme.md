# Transpiler
Clang tool-based (ASTMatcher, Rewriter) source-to-source transpiler.

## Build

First, checkout llvm, clang, and transpiler

```bash
$ cd where-you-want-llvm-to-live
$ git clone http://github.com/llvm-mirror/llvm
#... downloading out llvm repository
$ cd llvm
$ git checkout release_80
#... switching to release 8.0
$ cd tools # llvm/tools
$ git clone http://github.com/llvm-mirror/clang
#... downloading clang
$ cd clang # llvm/tools/clang
$ git checkout release_80
#... switching to release 8.0

# THE FOLLOWING PART IS NEW
$ cd .. # llvm/tools/clang/tools
$ git clone github.com/hjunkim/transpiler
#... downloading transpiler for CUDA
```

## Run
- '--' 이후 flag/option들을 정의
- --cuda-device-only: CUDA 프로그램의 host/device 코드들 중 device (cuda kernel function) 만을 대상으로 수행한다고 명시
- --cuda-path=/usr/local/cuda: CUDA 설치 path
- --cuda-gpu-arch=sm\_xx: CUDA architecture에 맞게 작성 (Titan V, V100: sm\_70)

```bash
$ transpiler cuda_program.cu -- --cuda-device-only --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_xx
```
