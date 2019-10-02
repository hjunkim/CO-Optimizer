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

int footprints = 0;

static llvm::cl::OptionCategory MatcherSampleCategory("Matcher Sample");

class ForStmtHandler : public MatchFinder::MatchCallback {
public:
  ForStmtHandler(Rewriter &Rewrite) : Rewrite(Rewrite) {}

  virtual void run(const MatchFinder::MatchResult &Result) {
	// visit ForStmt
	// t_ForLoop = newly visited ForStmt, ForLoop = prev. visited ForStmt
    if (const ForStmt *t_ForLoop = Result.Nodes.getNodeAs<ForStmt>("forLoop")) {
		// cache size --> cmdline parameter?
		if (footprints > 256) {
			// cache contention --> rewrite ForStmt
			Rewrite.InsertText(ForLoop->getBeginLoc(), "/* throttling start */", true, true);
			Rewrite.InsertText(ForLoop->getEndLoc(), "/* throttling end */", true, true);
		} 

		// move to next ForStmt
		ForLoop = const_cast<ForStmt*>(t_ForLoop);
		footprints = 0;
		
	}

	// visit Iterator variable
   	if (const DeclRefExpr *t_iterVar = Result.Nodes.getNodeAs<DeclRefExpr>("iterVar")) {
			std::cout << t_iterVar->getNameInfo().getAsString() << std::endl;
	}

	// visit Array
   	if (const ArraySubscriptExpr *ArrayVar = Result.Nodes.getNodeAs<ArraySubscriptExpr>("array")) {
    	// Rewrite.InsertText(ArrayVar->getBeginLoc(), "-", true, true);
		// ArrayVar->getIdx()->dump();
		// temp footprints variable
		int t_ftp = 0;

		footprints += t_ftp;
	}
  }

private:
  Rewriter &Rewrite;
  ForStmt *ForLoop;
  
  //DeclRefExpr *iterVar;
  std::string iterVarName;
};

// Implementation of the ASTConsumer interface for reading an AST produced
// by the Clang parser. It registers a couple of matchers and runs them on
// the AST.
class MyASTConsumer : public ASTConsumer {
public:
  MyASTConsumer(Rewriter &R) : HandlerForTT(R) {

        /*forStmt(hasLoopInit(declStmt(hasSingleDecl(
                    varDecl(hasInitializer(integerLiteral(equals(0))))
                        .bind("initVarName")))),
                hasIncrement(unaryOperator(
                    hasOperatorName("++"),
                    hasUnaryOperand(declRefExpr(to(
                        varDecl(hasType(isInteger())).bind("incVarName")))))),
                hasCondition(binaryOperator(
                    hasOperatorName("<"),
                    hasLHS(ignoringParenImpCasts(declRefExpr(to(
                        varDecl(hasType(isInteger())).bind("condVarName"))))),
                    hasRHS(expr(hasType(isInteger()))))))
            .bind("forLoop"),
        &HandlerForFor);*/
		

	// find a for loop that exceeds L1 footprints, then send it to the handler above
    Matcher.addMatcher(
        forStmt(hasDescendant(arraySubscriptExpr()),
				hasIncrement(unaryOperator(hasUnaryOperand(declRefExpr().bind("iterVar"))))
		).bind("forLoop"),
	&HandlerForTT);

    Matcher.addMatcher(
		arraySubscriptExpr(hasAncestor(forStmt())).bind("array"),
	&HandlerForTT);
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    // Run the matchers when we have the whole TU parsed.
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
