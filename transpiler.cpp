//------------------------------------------------------------------------------
// AST matching sample. Demonstrates:
//
// * How to write a simple source tool using libTooling.
// * How to use AST matchers to find interesting AST nodes.
// * How to use the Rewriter API to rewrite the source code.
//
// Eli Bendersky (eliben@gmail.com)
// This code is in the public domain
//------------------------------------------------------------------------------
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

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;

static llvm::cl::OptionCategory MatcherSampleCategory("Matcher Sample");

bool isSecCall = false;

class ForStmtHandler : public MatchFinder::MatchCallback {
public:
  ForStmtHandler(Rewriter &Rewrite) : Rewrite(Rewrite) {
	isGlobalVar = false;
	isGPUKernel = false;
  }

  virtual void run(const MatchFinder::MatchResult &Result) {
	if (const FunctionDecl *t_funcDecl = Result.Nodes.getNodeAs<FunctionDecl>("funcDecl")) {
		if (t_funcDecl->hasAttr<CUDAGlobalAttr>()) {
			std::cout << "CUDA: " << t_funcDecl->getNameInfo().getAsString() << std::endl;
			isGPUKernel = true;
		} else {
			isGPUKernel = false;
		}

		// init tidVar
		for (int is=0; is<tidVar.size(); is++)
			tidVar.erase(tidVar.begin() + is);
	}

	// GPU Code && first AST traverse
	if (isGPUKernel && !isSecCall) {
		// visit ForStmt, initialize parameters
		if (const ForStmt *t_ForLoop = Result.Nodes.getNodeAs<ForStmt>("forLoop")) {
			ForLoop = const_cast<ForStmt*>(t_ForLoop);
		
			// init parmVarName
			for (int is=0; is<parmVarName.size(); is++)
				parmVarName.erase(parmVarName.begin() + is);

			footprints.erase(ForLoop);
			hasIterVar.erase(ForLoop);

		}

		// visit Parameter variable lists (they are global variables we are targeting on!!!)
    	if (const ValueDecl *t_parmVar = Result.Nodes.getNodeAs<ValueDecl>("parmVarDecl")) {
			// std::cout << "parmVar: " << t_parmVar->getNameAsString() << std::endl;
			parmVarName.push_back(t_parmVar->getNameAsString());
		}


		// visit variable assignment assigned with threadIdx.x/y/z, 
		// 		pattern 1) int i = threadIdx.x; --> "varDecl"
		//  	pattern 2) int i; i = threadIdx.x; --> "declRefExpr"
		if (const ValueDecl *t_decl = Result.Nodes.getNodeAs<ValueDecl>("varDecl")) {
				// std::cout << "varDecl: " << t_decl->getNameAsString() << std::endl;
				tidVar.push_back(t_decl->getNameAsString());
		}
		if (const DeclRefExpr *t_decl = Result.Nodes.getNodeAs<DeclRefExpr>("declRefExpr")) {
				// std::cout << "declRefExpr: " << t_decl->getNameInfo().getAsString() << std::endl;
				tidVar.push_back(t_decl->getNameInfo().getAsString());
		}


		// visit Array ??? for what ???
   		if (const ArraySubscriptExpr *t_array = Result.Nodes.getNodeAs<ArraySubscriptExpr>("array")) {
    		// Rewrite.InsertText(ArrayVar->getBeginLoc(), "-", true, true);
			// t_array->getIdx()->dump();
			// t_array->dump();
			// temp footprints variable, iteratively collect footprints
			int t_ftp = 1;
		
			footprints[ForLoop] += t_ftp;
		}
		// check if array var is global
   		if (const DeclRefExpr *t_var = Result.Nodes.getNodeAs<DeclRefExpr>("checkGlobalArrayVar")) {
			// t_var->getNameInfo().getAsString()
			std::string t_str = t_var->getNameInfo().getAsString();
			for (int is=0; is<parmVarName.size(); is++) {
				if (t_str == parmVarName[is]) {
					// std::cout << t_str << " is global var" << std::endl;
					isGlobalVar = true;
				}
			}
		}
		// visit each var in array index
   		if (const DeclRefExpr *t_var = Result.Nodes.getNodeAs<DeclRefExpr>("arrayIdx")) {
			std::string t_str = t_var->getNameInfo().getAsString();
			if (t_str == iterVarName) {
				// std::cout << t_str << " is an iter. var" << std::endl;
				hasIterVar[ForLoop] = true;
			}

			for (int is=0; is<tidVar.size(); is++) {
				if (t_str == tidVar[is]) {
					// std::cout << t_str << " is ---- tid var" << std::endl;
					// check its coefficient
					;
				}
			}
		}
		// visit each var in array index
   		if (const Expr *t = Result.Nodes.getNodeAs<Expr>("pattern0")) {
			std::cout << "pattern 0: " << std::endl;
		}
		if (const Expr *t = Result.Nodes.getNodeAs<Expr>("pattern1")) {
			std::cout << "\tpattern 1: " << std::endl;
		}
		if (const Expr *t = Result.Nodes.getNodeAs<Expr>("pattern2")) {
			std::cout << "\t\tpattern 2: " << std::endl;
		}


		// visit Iterator variable
		// saves iterator variable to iterVarName
   		if (const DeclRefExpr *t_iterVar = Result.Nodes.getNodeAs<DeclRefExpr>("iterVar")) {
			// std::cout << t_iterVar->getNameInfo().getAsString() << std::endl;
			iterVarName = t_iterVar->getNameInfo().getAsString();
		}
	}


	// GPU code && second AST traverse
	if (isGPUKernel && isSecCall) {
		// check footprints and loop-variable (iterVar)
		// std::map<string, int> footprints, std::map<string, bool>
    	if (const ForStmt *t_ForLoop = Result.Nodes.getNodeAs<ForStmt>("forLoop")){
			ForLoop = const_cast<ForStmt*>(t_ForLoop);

			for (auto is=footprints.begin(); is!=footprints.end(); is++) {
				std::cout << "\t\tfootprints-Key: " << is->first << "- Value: " << is->second << std::endl;
			}
			for (auto is=hasIterVar.begin(); is!=hasIterVar.end(); is++) {
				std::cout << "\t\thasIterVar-Key: " << is->first << "- Value: " << is->second << std::endl;
			}

			// cache size --> cmdline parameter?, if hasIterVar is set
			if (footprints[ForLoop] > 256 && hasIterVar[ForLoop]) {
				// cache contention --> rewrite ForStmt
				Rewrite.InsertText(ForLoop->getBeginLoc(), "/* throttling start */", true, true);
				Rewrite.InsertText(ForLoop->getEndLoc(), "/* throttling end */", true, true);
			}
		}
	}
  }

private:
  Rewriter &Rewrite;
  // current ForStmt
  ForStmt *ForLoop;
  
  //DeclRefExpr *iterVar;
  std::string iterVarName;
  std::vector<std::string> parmVarName;
  std::vector<std::string> tidVar;
  std::map<ForStmt *, int> footprints;
  std::map<ForStmt *, bool> hasIterVar;
  bool isGlobalVar;
  bool isGPUKernel;
};

// Implementation of the ASTConsumer interface for reading an AST produced
// by the Clang parser. It registers a couple of matchers and runs them on
// the AST.
class MyASTConsumer : public ASTConsumer {
public:
  MyASTConsumer(Rewriter &R) : HandlerForTT(R) {
	// to distiguish cuda function and others
	Matcher.addMatcher(
		functionDecl().bind("funcDecl"),
	&HandlerForTT);

	// to collect global variables
	Matcher.addMatcher(
		parmVarDecl().bind("parmVarDecl"),
	&HandlerForTT);

	// find threadIdx.x
	Matcher.addMatcher(
		memberExpr(
			has(opaqueValueExpr(hasType(recordDecl(hasName("__cuda_builtin_threadIdx_t"))))),
			member(matchesName(".__fetch_builtin_x")),
			anyOf(
					hasAncestor(binaryOperator(has(declRefExpr().bind("declRefExpr")))),
					hasAncestor(varDecl().bind("varDecl"))
			)
		).bind("memberExpr"),
	&HandlerForTT);

	
	// is it global variable?
    Matcher.addMatcher(
		arraySubscriptExpr(
			hasAncestor(forStmt()),
			has(implicitCastExpr(has(declRefExpr().bind("checkGlobalArrayVar")))),
			forEachDescendant(declRefExpr().bind("arrayIdx"))
		).bind("array"),
	&HandlerForTT);


	// array index for calculating footprints size
    Matcher.addMatcher(
		arraySubscriptExpr(
			anyOf(
				hasIndex(integerLiteral().bind("pattern0")), // array[0]
				hasIndex(implicitCastExpr().bind("pattern1")), // array[var]
				hasIndex(binaryOperator().bind("pattern2")) // array[var * N + ..]
				// hasIndex(parenExpr().bind("pattern2")) 
			),
			hasAncestor(forStmt())
		),
	&HandlerForTT);


	// find a for loop that exceeds L1 footprints, then send it to the handler above
    Matcher.addMatcher(
        forStmt(
			hasDescendant(arraySubscriptExpr()),
			hasIncrement(unaryOperator(hasUnaryOperand(declRefExpr().bind("iterVar"))))
		).bind("forLoop"),
	&HandlerForTT);
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    // Run the matchers when we have the whole TU parsed.
    Matcher.matchAST(Context);
	isSecCall = true;
    Matcher.matchAST(Context);
  }

private:
  // added Matcher by hjkim
  ForStmtHandler HandlerForTT;
  MatchFinder Matcher;
};

// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
public:
  MyFrontendAction() {}
  void EndSourceFileAction() override {
    
    TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID()).write(llvm::outs());
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, 
		  				StringRef file) override {
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<MyASTConsumer>(TheRewriter);
    // c++14 feature, return std::make_unique<MyASTConsumer>(TheRewriter);
  }

private:
  Rewriter TheRewriter;
};

int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, MatcherSampleCategory);
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());

  return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
