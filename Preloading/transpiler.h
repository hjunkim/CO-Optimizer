// hjunkim@skku.edu
// This program requires boost library
// Ubuntu: sudo apt-get install libboost-all-dev

#include <iostream>
#include <string>
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

static llvm::cl::OptionCategory MatcherSampleCategory("Transpiler Preloading");
static llvm::cl::opt<std::string> op_blksize("tbsize", llvm::cl::desc("<tbsize> : set thread block size (default: 256)"),
                            llvm::cl::init("256"), llvm::cl::cat(MatcherSampleCategory));
static llvm::cl::opt<std::string> op_prdsize("prdsize", llvm::cl::desc("<prdsize> : set preloading size (default: 1)"),
                            llvm::cl::init("1"), llvm::cl::cat(MatcherSampleCategory));


bool isSecCall = false;
bool isTrdCall = false;

// user defined parameters
std::string blksize;
std::string prdsize;
std::string allocsize;
