// hjunkim@skku.edu
// This program requires boost library
// Ubuntu: sudo apt-get install libboost-all-dev

#include <string>
#include <iostream>
#include <boost/tokenizer.hpp>

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

#define PDBG(tstr) \
		std::cout << "// DEBUG: " << tstr << std::endl;

typedef boost::tokenizer<boost::char_separator<char> > tokenizer;

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;

static llvm::cl::OptionCategory MatcherSampleCategory("Transpiler");

// Throttling
static llvm::cl::opt<int> op_blksize("tbsize", llvm::cl::desc("<tbsize> : set thread block size (default: 8 warps)"),
                            llvm::cl::init(8), llvm::cl::cat(MatcherSampleCategory));
static llvm::cl::opt<int> op_nblks("nblks", llvm::cl::desc("<nblks> : set # of thread blocks per SM (default: 4 blks)"),
                            llvm::cl::init(4), llvm::cl::cat(MatcherSampleCategory));
static llvm::cl::opt<int> op_csize("csize", llvm::cl::desc("<csize> : set L1 cache size of a GPU (default: 32 KB)"),
                            llvm::cl::init(32), llvm::cl::cat(MatcherSampleCategory));

// Preloading
static llvm::cl::opt<int> op_prdsize("prdsize", llvm::cl::desc("<prdsize> : set preloading size (default: 1)"),
                            llvm::cl::init(1), llvm::cl::cat(MatcherSampleCategory));

bool isSecCall = false;
bool isTrdCall = false;

// Throttling -- user defined parameters
int WARPS_SM; // = op_blksize*op_nblks;
int CACHE_SIZE = op_csize * 1024 / 128; // 1024: KB --> B, 128: cache line size: 128B

// Preloading -- user defined parameters
int blksize;
int prdsize;
int allocsize;

