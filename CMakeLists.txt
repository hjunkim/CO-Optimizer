set(LLVM_LINK_COMPONENTS support)
add_clang_executable(transpiler Transpiler.cpp)
target_link_libraries(transpiler PRIVATE clangAST
  clangASTMatchers clangBasic
  clangFrontend clangRewrite clangTooling)
