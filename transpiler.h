// hjunkim@skku.edu

#include <string>
#include <iostream>

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"

// user defined parameters
int BLOCK_SIZE = 8;	// 8 warps per thread block
int N_BLOCKS_SM = 4;	// 8 blocks per SM
int WARPS_SM = 8*4;
#define CACHE_SIZE 256 	// 32KB (cache size) * 1024B / 128B (cache line size)

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;

static llvm::cl::OptionCategory MatcherSampleCategory("Matcher Sample");

bool isSecCall = false;
