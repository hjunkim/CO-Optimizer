#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"

#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

#include "clang/Rewrite/Core/Rewriter.h"

using namespace llvm;
using namespace clang;
using namespace clang::tooling;
using namespace clang::ast_matchers;

static cl::OptionCategory MyToolCategory("transpiler options");
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::extrahelp MoreHelp("\nMore help text...\n");

class LiteralArgCommenter : public MatchFinder::MatchCallback {
public:
    LiteralArgCommenter(Rewriter &MyRewriter) : MyRewriter(MyRewriter) {}

    // This callback will be executed whenever the Matcher in MyASTConsumer matches
    virtual void run(const MatchFinder::MatchResult &Result) {
        // ASTContext allows us to find the source location
        ASTContext *Context = Result.Context;
        // Record the callees params. We can access the callee via the .bind("callee")
        // from the ASTMatcher. We will match these with the callers arg. later.
        std::vector<ParmVarDecl *> Params;
        const FunctionDecl *CD =
            Result.Nodes.getNodeAs<clang::FunctionDecl>("callee");
        for (FunctionDecl::param_const_iterator PI = CD->param_begin(),
                                                PE = CD->param_end();
                                                PI != PE; ++PI) {
            Params.push_back(*PI);
        }

        const CallExpr *E = Result.Nodes.getNodeAs<clang::CallExpr>("functions");
        size_t Count = 0;

        if (E && CD && !Params.empty()) {
            auto I = E->arg_begin();
            // The first param is the object itself, skip over it
            if (isa<CXXOperatorCallExpr>(E))
                ++I;
            
            // For each arg. match it with the callee param. If it is an int. or bool. literal
            // then, insert a comment into the edit buffer.
            for (auto End = E->arg_end(); I != End; ++I, ++Count) {
                ParmVarDecl *PD = Params[Count];
                FullSourceLoc ParmLocation = Context->getFullLoc(PD->getBeginLoc());
                const Expr *AE = (*I)->IgnoreParenCasts();
                if (auto *IntArg = dyn_cast<IntegerLiteral>(AE)) {
                    FullSourceLoc ArgLoc = Context->getFullLoc(IntArg->getBeginLoc());
                    if (ParmLocation.isValid() && !PD->getDeclName().isEmpty()) // &&
                        // ToDo: Fix --> EditedLocations.insert(ArgLoc).second)
                        // Will insert text immediately before the argument
                        MyRewriter.InsertText(
                            ArgLoc,
                            (Twine(" /* ") + PD->getDeclName().getAsString() + " */ ").str());
                }
            }
        }
    }
private:
    Rewriter MyRewriter;

};

class MyASTConsumer : public ASTConsumer {
public:
// Use almost the same syntax as the ASTMatcher prototyped in clang-query.
// The changes are the .bind(string) additions so that we can access these once the match has occurred.
    MyASTConsumer(Rewriter &R) : LAC(R) {
        StatementMatcher CallSiteMatcher =
            CallExpr(allOf(callee(functionDecl(unless(isVariadic())).bind("callee")),
                    unless(cxxMemberCallExpr(
                        on(hasType(substTemplateTypeParmType())))),
                    anyOf(hasAnyArgument(ignoringParenCasts(cxxBoolLiteral())),
                        hasAnyArgument(ignoringParenCasts(integerLiteral())))
                    )
                )
                .bind("functions");
        // LAC is our callback that will run when the ASTMatcher finds the pattern above
        Matcher.addMatcher(CallSiteMatcher, &LAC);
    }
    // Implement the call back so that we can run our Matcher on the source file
    void HandleTranslationUnit(ASTContext &Context) override {
        Matcher.matchAST(Context);
    }
private:
    MatchFinder Matcher;
    LiteralArgCommenter LAC;
};

class MyFrontendAction : public ASTFrontendAction {
public:
    // Output the edit buffer for this translation unit
    void EndSourceFileAction() override {
        MyRewriter.getEditBuffer(MyRewriter.getSourceMgr().getMainFileID()).write(llvm::outs());
    }
    // Returns our ASTConsumer implemenation per translation unit
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
        MyRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
        // llvm::make_unique --> std::make_unique
        return std::make_unique<MyASTConsumer>(MyRewriter);
    }
private:
    Rewriter MyRewriter;
};

int main(int argc, const char **argv) {
    CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
    ClangTool Tool(OptionsParser.getCompilations(),
        OptionsParser.getSourcePathList());
    return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
