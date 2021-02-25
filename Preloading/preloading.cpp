#include "../transpiler.h"

class ForStmtHandler : public MatchFinder::MatchCallback {
public:
	ForStmtHandler(Rewriter &Rewrite) : Rewrite(Rewrite) {
		isGPUKernel = false;
	}

	virtual void run(const MatchFinder::MatchResult &Result) {
		if (const FunctionDecl *t_funcDecl = Result.Nodes.getNodeAs<FunctionDecl>("funcDecl")) {
			funcDecl = const_cast<FunctionDecl*>(t_funcDecl);
			if (t_funcDecl->hasAttr<CUDAGlobalAttr>() || t_funcDecl->hasAttr<CUDADeviceAttr>()) {
				/* if (!isSecCall) 
					PDBG("["+t_funcDecl->getNameInfo().getAsString()+"]") */
				isGPUKernel = true;
			} else {
				isGPUKernel = false;
			}

			// init tidVar
			for (int is=0; is<tidVar.size(); is++) {
				tidVar.erase(tidVar.begin() + is);
			}
		}

		// GPU Code && first AST traverse
		if (isGPUKernel && !isSecCall) {
			// ----------------------- param var init ------------------------------- //
			// visit a Parameter variable list (those are global variables we are targeting on!!!)
			if (const ValueDecl *t_parmVar = Result.Nodes.getNodeAs<ValueDecl>("parmVarDecl")) {
				parmVarName.push_back(t_parmVar->getNameAsString());
			}
			// ----------------------- param var init ------------------------------- //

			// ----------------------- variable init ------------------------------- //
			// push an iterator variable to iterVarName
			if (const DeclRefExpr *t_iterVar = Result.Nodes.getNodeAs<DeclRefExpr>("iterVar")) {
				iterVarName = t_iterVar->getNameInfo().getAsString();
			}

			// push a variable assignment assigned with any of threadIdx.x/y/z, 
			// 		pattern 1) int i = threadIdx.x; --> "varDecl"
			if (const ValueDecl *t_decl = Result.Nodes.getNodeAs<ValueDecl>("varDecl")) {
				tidVar.push_back(t_decl->getNameAsString());
			}
			//  	pattern 2) int i; i = threadIdx.x; --> "declRefExpr"
			if (const DeclRefExpr *t_decl = Result.Nodes.getNodeAs<DeclRefExpr>("declRefExpr")) {
				tidVar.push_back(t_decl->getNameInfo().getAsString());
			}
			// 		pattern 3) int i = threadIdx.y/z; --> "varDecl"
			if (const ValueDecl *t_decl = Result.Nodes.getNodeAs<ValueDecl>("yz_varDecl")) {
				yz_tidVar.push_back(t_decl->getNameAsString());
			}
			//  	pattern 4) int i; i = threadIdx.y/z; --> "declRefExpr"
			if (const DeclRefExpr *t_decl = Result.Nodes.getNodeAs<DeclRefExpr>("yz_declRefExpr")) {
				yz_tidVar.push_back(t_decl->getNameInfo().getAsString());
			}
			// ----------------------- variable init ------------------------------- //


			// ----------------------- first array visit ------------------------------- //
			if (const ArraySubscriptExpr *t_array = Result.Nodes.getNodeAs<ArraySubscriptExpr>("array")) {
				ArrayVar = const_cast<ArraySubscriptExpr*>(t_array);
			}
			// check if an array variable is global
			if (const DeclRefExpr *t_var = Result.Nodes.getNodeAs<DeclRefExpr>("checkGlobalArrayVar")) {
				std::string t_str = t_var->getNameInfo().getAsString();
				QualType t_type = t_var->getType();

				arrayBase[ArrayVar] = t_str;
				arrayType[ArrayVar] = t_type.getAsString();

				for (int is=0; is<parmVarName.size(); is++) {
					if (t_str == parmVarName[is]) {
						isGlobalVar[ArrayVar] = true;
					}
				}
			}
			// push an index of each array variable
			if (const auto *IndexExpr = Result.Nodes.getNodeAs<Expr>("arrayIndex")) {
				std::string ttext = Lexer::getSourceText(CharSourceRange::getTokenRange(IndexExpr->getSourceRange()), *Result.SourceManager, LangOptions(), 0);
				arrayIdx[ArrayVar] = ttext;
			}
			// visit each variable in an array index
			if (const DeclRefExpr *t = Result.Nodes.getNodeAs<DeclRefExpr>("eachIdxVar")) {
				std::string t_str = t->getNameInfo().getAsString();
				// check if there is a tid variable
				for (int is=0; is<tidVar.size(); is++) {
					if (t->getNameInfo().getAsString() == tidVar[is]) {
						hasTidVar[ArrayVar] = true;
					}
				}
				// check if there is an iterator variable
				if (t_str == iterVarName) {
					hasIterVar[ArrayVar] = true;
					iterVar[ArrayVar] = iterVarName;
				}
			}
			// ----------------------- first array visit ------------------------------- //
		}

		// GPU code && second AST traverse
		if (isGPUKernel && isSecCall) {
			if (const ForStmt *t_ForLoop = Result.Nodes.getNodeAs<ForStmt>("forLoop")) {
				ForLoop = const_cast<ForStmt*>(t_ForLoop);
			}
			// visit array for the second time	
			if (const ArraySubscriptExpr *t_array = Result.Nodes.getNodeAs<ArraySubscriptExpr>("array")) {
				ArrayVar = const_cast<ArraySubscriptExpr*>(t_array);
				std::string ttext = arrayIdx[ArrayVar];
				// check & modify candidates for preloading
				if (hasIterVar[ArrayVar] && !(hasTidVar[ArrayVar]) && isGlobalVar[ArrayVar] 
						&& !(transKernel[funcDecl])) {
					PDBG("["+funcDecl->getNameInfo().getAsString()+"] has a preloading cand.")
					transKernel[funcDecl] = true;

					boost::char_separator<char> sep(" ", "+-*/()[]"); // , boost::keep_empty_tokens);
					tokenizer tokens(ttext, sep);

					sharedStore[ArrayVar] = "";
					globalLoad[ArrayVar] = "";
					sharedLoad[ArrayVar] = "";



					for (tokenizer::iterator tok=tokens.begin(); tok != tokens.end(); ++tok) {
						// tokenizing [a+b*(c+d)] -> a, +, b, *, (,  c, +, d, )

						// 1) shared store --> store preloading data
						// 2) global load --> preloading
						// 3) shared load --> load preloading data
						
						// - if tok == tidVar
						bool f_tid = false;
						for (int is=0; is<tidVar.size(); is++) {
							if ((*tok == tidVar[is])) {
								globalLoad[ArrayVar] += *tok;
								sharedStore[ArrayVar] += "threadIdx.x";
								sharedLoad[ArrayVar] += "threadIdx.x";
								f_tid = true;
							}
						}

						if (!(f_tid)) {
							for (int is=0; is<yz_tidVar.size(); is++) {
								if ((*tok == yz_tidVar[is])) {
									PDBG("2/3-D thread block cannot be transformed automatically")
									return ;
								}
							}

							if ((*tok == iterVar[ArrayVar])) {
								sharedStore[ArrayVar] += "threadIdx.x";
								globalLoad[ArrayVar] += "threadIdx.x";
								sharedLoad[ArrayVar] += "("+*tok+"%("+std::to_string(allocsize)+"))";
							}
							// 		- if tok == threadIdx.x
							else if ((*tok == "threadIdx.x")) {
								globalLoad[ArrayVar] += *tok;
							}
							// 		- if tok == threadIdx.y or threadIdx.z
							else if ((*tok == "threadIdx.y") || (*tok == "threadIdx.z")) {
								PDBG("2/3-D thread block cannot be transformed automatically")
								// globalLoad[ArrayVar] += *tok;
								return ;
							}
							else if ((*tok) == "[") {
								PDBG("Indirect memory access cannot be analyzed at compile time")
								return ;
							}
							//		- else tok == N, 
							else {
								for (int is=0; is<parmVarName.size(); is++) {
									if ((*tok) == parmVarName[is]) {
										return ;
									}
								}
								sharedStore[ArrayVar] += *tok;
								globalLoad[ArrayVar] += *tok;
								// sharedLoad[ArrayVar] += *tok;
							}
						}
					}

					// ------ preloading
					// insert preloading
					// ToDo:
					// 		1) 1-D, [i] --> [threadIdx.x]
					// 		2) 2-D, [N*x/y+i] --> [x+threadIdx.y/x]
					// 		3) 1x, Mx, [threadIdx.x], [threadIdx.x+M*blockDim.x]

					// prdsize (preloading size, default 1), tbsize (thread block size, default: 256)
					std::string datatype = arrayType[ArrayVar];
					// remove pointer from data type string (i.e., float * a)
					if (datatype.find('*') != std::string::npos) {
						datatype.erase(datatype.find('*'));
					}
					if (datatype.find('[') != std::string::npos) {
						datatype.erase(datatype.find('['));
					}

					sharedStore[ArrayVar] += "+(ii*"+std::to_string(blksize)+")";
					globalLoad[ArrayVar] += "+(ii*"+std::to_string(blksize)+")";

					// replace x[i] --> sss[i%blockDim.x]
					Rewrite.ReplaceText(t_array->getBeginLoc(), Rewrite.getRangeSize(t_array->getSourceRange()), 
											"sss["+sharedLoad[ArrayVar]+"]");

					// shared memory allocation code
					std::string sa_str = "\n__shared__ "+datatype+" sss["+std::to_string(allocsize)+"];\n";
					Rewrite.InsertText(ForLoop->getBeginLoc().getLocWithOffset(-1), sa_str, false, true);

					bool hasCompoundBody = false;
					for (auto *CI : ForLoop->children()) {
					// for (Stmt::child_iterator ci=ForLoop->child_begin(); ci != ForLoop->child_end(); ++ci) {
						if (CI && isa<CompoundStmt>(CI)) {
							hasCompoundBody = true;
						}
					}

					std::string pr_str = " ";
					// preloading code
					if (hasCompoundBody)
						pr_str +=  "\n/* preloading code added (base["+arrayIdx[ArrayVar]+"])...*/\n";
					else
						pr_str +=  "{\n/* preloading code added (base["+arrayIdx[ArrayVar]+"])...*/\n";
					
					pr_str += "if ("+iterVar[ArrayVar]+"%"+std::to_string(allocsize)+"==0) {\n";
					pr_str += "\tfor(int ii=0; ii<"+std::to_string(prdsize)+"; ii++) {\n\t";
					pr_str += "\tsss["+sharedStore[ArrayVar]+"] = "+arrayBase[ArrayVar];
					pr_str += "["+globalLoad[ArrayVar]+"];\n}\n\t__syncthreads();\n}\n";
					pr_str += "/* preloading code ended...*/\n";

					if (hasCompoundBody)
						Rewrite.InsertText(ForLoop->getBody()->getBeginLoc().getLocWithOffset(1), pr_str, true, true);
					else
						Rewrite.InsertText(ForLoop->getBody()->getBeginLoc(), pr_str, true, true);

					// preload finishing code --> add '}' end of the new code
					if (!hasCompoundBody)
						Rewrite.InsertText(ForLoop->getBody()->getEndLoc().getLocWithOffset(2), "\n}\n", true, true);
				}
			}
		}
	}

private:
	Rewriter &Rewrite;
	FunctionDecl *funcDecl;
	ForStmt *ForLoop;

	ArraySubscriptExpr *ArrayVar;

	std::map<FunctionDecl *, bool> transKernel;	// is a kernel transformed to preloading
	//
	std::string iterVarName;	// iterator variable name, i.e., for (int i=0;..)
	//
	std::vector<std::string> parmVarName;	// function param. 
	std::vector<std::string> tidVar;		// int a = threadIdx.x;
	std::vector<std::string> yz_tidVar;		// int a = threadIdx.y;
	//
	std::map<ArraySubscriptExpr *, bool> hasIterVar;	// array index has a iter variable?
	std::map<ArraySubscriptExpr *, bool> hasTidVar;	// array index has a tid?
	std::map<ArraySubscriptExpr *, bool> isGlobalVar;	// array index is a global variable?
	//
	std::map<ArraySubscriptExpr *, std::string> iterVar;	// iterator var for modif. i for [i] --> [tid.x]
	//
	std::map<ArraySubscriptExpr *, std::string> arrayType;	// 
	std::map<ArraySubscriptExpr *, std::string> arrayBase;	// base name for array index, i.e., 'a' in a[i]
	std::map<ArraySubscriptExpr *, std::string> arrayIdx;	// idx for modif. N*i for [N*i] --> [N*tid.x]
	//
	std::map<ArraySubscriptExpr *, std::string> sharedStore;	
	std::map<ArraySubscriptExpr *, std::string> sharedLoad;
	std::map<ArraySubscriptExpr *, std::string> globalLoad;

	bool isGPUKernel;
};

// Implementation of the ASTConsumer interface for reading an AST produced
// by the Clang parser. It registers a couple of matchers and runs them on
// the AST.
class PreloadingASTConsumer : public ASTConsumer {
	public:
		PreloadingASTConsumer(Rewriter &R) : HandlerForTT(R) {
			// to distiguish cuda function from others
			Matcher.addMatcher(
					functionDecl().bind("funcDecl"),
					&HandlerForTT);

			// to collect global variables
			Matcher.addMatcher(
					parmVarDecl().bind("parmVarDecl"),
					&HandlerForTT);

			// to find threadIdx.{x, y, z}
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

			Matcher.addMatcher(
					memberExpr(
						has(opaqueValueExpr(hasType(recordDecl(hasName("__cuda_builtin_threadIdx_t"))))),
						anyOf(
							member(matchesName(".__fetch_builtin_y")),
							member(matchesName(".__fetch_builtin_z"))
							),
						anyOf(
							hasAncestor(binaryOperator(has(declRefExpr().bind("yz_declRefExpr")))),
							hasAncestor(varDecl().bind("yz_varDecl"))
							)
						),
					&HandlerForTT);


			// to check if it is global variable?, if an array index has any of tid.{x/y/z}
			Matcher.addMatcher(
					arraySubscriptExpr(
						hasParent(implicitCastExpr()), // array load only, no store
						hasAncestor(forStmt()),
						hasBase(implicitCastExpr(hasSourceExpression(declRefExpr().bind("checkGlobalArrayVar")))),
						hasIndex(expr().bind("arrayIndex")),
						forEachDescendant(declRefExpr().bind("eachIdxVar"))
						).bind("array"),
					&HandlerForTT);


			// ToDo: needs to modify -- send it for modification when it has index with tid.{xyz}
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
		ForStmtHandler HandlerForTT;
		MatchFinder Matcher;
};

// For each source file provided to the tool, a new FrontendAction is created.
class PreloadingAction : public ASTFrontendAction {
	public:
		PreloadingAction() {}
		void EndSourceFileAction() override {

			TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID()).write(llvm::outs());
		}

		std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, StringRef file) override {
			TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
			return llvm::make_unique<PreloadingASTConsumer>(TheRewriter);
			// c++14 feature, return std::make_unique<PreloadingASTConsumer>(TheRewriter);
		}

	private:
		Rewriter TheRewriter;
};

int main(int argc, const char **argv) {
	CommonOptionsParser op(argc, argv, MatcherSampleCategory);
	ClangTool Tool(op.getCompilations(), op.getSourcePathList());

	/*if () {
		llvm::errs() << "Thread block size is set to default (256)";
	}*/
	// llvm::errs() << "Thread block size is set to " << opblksize << std::endl;
	blksize = op_blksize * 32;
	prdsize = op_prdsize;

	PDBG("Thread block size : " + std::to_string(blksize))
	PDBG("Preload size : " + std::to_string(prdsize))

	// shared memory allocation size, [256*1]
	allocsize = blksize * prdsize;

	return Tool.run(newFrontendActionFactory<PreloadingAction>().get());
}
