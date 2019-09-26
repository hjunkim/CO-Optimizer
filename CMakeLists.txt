set(LLVM_LINK_COMPONENTS support)
add_clang_executable(transpiler transpiler.cpp)
target_link_libraries(transpiler PRIVATE clangFrontend
  clangDriver clangTooling clangFrontendTool
  clangSerialization clangCodeGen clangParse
  clangSema clangStaticAnalyzerFrontend
  clangStaticAnalyzerCore clangAnalysis
  clangARCMigrate clangRewriteFrontend
  clangEdit clangAST clangLex clangBasic clangASTMatchers)
