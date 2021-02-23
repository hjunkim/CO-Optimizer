set(LLVM_LINK_COMPONENTS support)
add_clang_executable(preloading preloading.cpp)
target_link_libraries(preloading PRIVATE clangFrontend
  clangDriver clangTooling clangFrontendTool
  clangSerialization clangCodeGen clangParse
  clangSema clangStaticAnalyzerFrontend
  clangStaticAnalyzerCore clangAnalysis
  clangARCMigrate clangRewriteFrontend
  clangEdit clangAST clangLex clangBasic clangASTMatchers)
