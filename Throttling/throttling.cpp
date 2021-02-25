#include "../transpiler.h"

class ForStmtHandler : public MatchFinder::MatchCallback {
	public:
		ForStmtHandler(Rewriter &Rewrite) : Rewrite(Rewrite) {
			isGlobalVar = false;
			isGPUKernel = false;
		}

		virtual void run(const MatchFinder::MatchResult &Result) {
			if (const FunctionDecl *t_funcDecl = Result.Nodes.getNodeAs<FunctionDecl>("funcDecl")) {
				if (t_funcDecl->hasAttr<CUDAGlobalAttr>()) {
					// PDBG("CUDA: " + t_funcDecl->getNameInfo().getAsString())
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

				// visit variable assignment with integer value, 
				// int i=0;
				if (const IntegerLiteral *t_var = Result.Nodes.getNodeAs<IntegerLiteral>("integerVar0")) {
					auto Parents = Result.Context->getParents(*t_var);
					const auto *t_varDecl = Parents[0].get<VarDecl>();

					std::string varName = t_varDecl->getNameAsString();
					varAss[varName] = t_var->getValue().getLimitedValue(); // varAss(name, value);
				}
				// int j; j=0;
				if (const IntegerLiteral *t_var = Result.Nodes.getNodeAs<IntegerLiteral>("integerVar1")) {
					auto Parents = Result.Context->getParents(*t_var);
					const auto *t_binaryOperator = Parents[0].get<BinaryOperator>();
					auto *t_child = t_binaryOperator->getLHS();
					const auto *Cast = dyn_cast<DeclRefExpr>(t_child);

					std::string varName = Cast->getNameInfo().getAsString();
				}

				// visit Parameter variable lists (they are global variables we are targeting on!!!)
				if (const ValueDecl *t_parmVar = Result.Nodes.getNodeAs<ValueDecl>("parmVarDecl")) {
					parmVarName.push_back(t_parmVar->getNameAsString());
				}


				// visit variable assignment assigned with threadIdx.x/y/z, 
				// 		pattern 1) int i = threadIdx.x; --> "varDecl"
				//  	pattern 2) int i; i = threadIdx.x; --> "declRefExpr"
				if (const ValueDecl *t_decl = Result.Nodes.getNodeAs<ValueDecl>("varDecl")) {
					tidVar.push_back(t_decl->getNameAsString());
				}
				if (const DeclRefExpr *t_decl = Result.Nodes.getNodeAs<DeclRefExpr>("declRefExpr")) {
					tidVar.push_back(t_decl->getNameInfo().getAsString());
				}


				// visit Array ??? for what ???
				/*if (const ArraySubscriptExpr *t_array = Result.Nodes.getNodeAs<ArraySubscriptExpr>("array")) {
				// Rewrite.InsertText(ArrayVar->getBeginLoc(), "-", true, true);
				// t_array->getIdx()->dump();
				// t_array->dump();
				// temp footprints variable, iteratively collect footprints
				int t_ftp = 0;

				footprints[ForLoop] += t_ftp;
				}*/
				// check if array var is global
				if (const DeclRefExpr *t_var = Result.Nodes.getNodeAs<DeclRefExpr>("checkGlobalArrayVar")) {
					// t_var->getNameInfo().getAsString()
					std::string t_str = t_var->getNameInfo().getAsString();
					for (int is=0; is<parmVarName.size(); is++) {
						if (t_str == parmVarName[is]) {
							isGlobalVar = true;
						}
					}
				}
				// visit each var in array index
				if (const DeclRefExpr *t_var = Result.Nodes.getNodeAs<DeclRefExpr>("arrayIdx")) {
					std::string t_str = t_var->getNameInfo().getAsString();
					if (t_str == iterVarName) {
						hasIterVar[ForLoop] = true;
					}
				}
				// visit each var in array index
				if (const Expr *t = Result.Nodes.getNodeAs<Expr>("pattern1")) {
					footprints[ForLoop] += 1;
				}
				// [tid * N .. ]
				if (const DeclRefExpr *t = Result.Nodes.getNodeAs<DeclRefExpr>("pattern2_1")) {
					for (int is=0; is<tidVar.size(); is++) {
						if (t->getNameInfo().getAsString() == tidVar[is]) {
							// check its coefficient
							auto iParents = Result.Context->getParents(*t);
							const auto *t_implicitCastExpr = iParents[0].get<ImplicitCastExpr>();
							auto bParents = Result.Context->getParents(*t_implicitCastExpr);
							const auto *t_binaryOperator = bParents[0].get<BinaryOperator>();

							auto *t_child = t_binaryOperator->getLHS();
							if (t_child == t_implicitCastExpr) {
								t_child = t_binaryOperator->getRHS();
							}
							// parenthesis?
							if (const auto *Cast = dyn_cast<ParenExpr>(t_child)) { 				// [tid * (..)]
								;
							}
							else if (auto *Cast = dyn_cast<ImplicitCastExpr>(t_child)) { 		// [tid * i]
								Stmt::child_iterator ci=Cast->child_begin();
								auto *cd = dyn_cast<DeclRefExpr>(*ci);
								if (!cd) {
									PDBG("Unknown status.")
								}
								std::string t_def = cd->getNameInfo().getAsString();
								// if the variable is not known at compile time, footprints += 1
								if (varAss[t_def]) {
									footprints[ForLoop] += (varAss[t_def] * WARPS_SM);
								}
								else {
									footprints[ForLoop] += 1;
								}
							}
							else if (const auto *Cast = dyn_cast<IntegerLiteral>(t_child)) {	// [tid * 7]
								int t_int = Cast->getValue().getLimitedValue();
								if (t_int > 32) t_int = 32;
								footprints[ForLoop] += (t_int * WARPS_SM);
							}
							/*else if (const auto *Cast = dyn_cast<FloatingLiteral>(t_child)) {
							  float t_float = Cast->getValue().convertToFloat();
							  if (t_float > 32.0) t_float = 32.0;
							  footprints[ForLoop] += t_float;
							  }*/
							  else {
								 PDBG("Unknown child")
							  }
						}
					}

				}
				// [tid + or tid -]
				if (const DeclRefExpr *t = Result.Nodes.getNodeAs<DeclRefExpr>("pattern2_2")) {
					for (int is=0; is<tidVar.size(); is++) {
						if (t->getNameInfo().getAsString() == tidVar[is]) {
							footprints[ForLoop] += 1;
						}
					}
				}


				// visit Iterator variable
				// saves iterator variable to iterVarName
				if (const DeclRefExpr *t_iterVar = Result.Nodes.getNodeAs<DeclRefExpr>("iterVar")) {
					iterVarName = t_iterVar->getNameInfo().getAsString();
				}
			}


			// GPU code && second AST traverse
			if (isGPUKernel && isSecCall) {
				// check footprints and loop-variable (iterVar)
				// std::map<string, int> footprints, std::map<string, bool>
				if (const ForStmt *t_ForLoop = Result.Nodes.getNodeAs<ForStmt>("forLoop")){
					ForLoop = const_cast<ForStmt*>(t_ForLoop);

					// debug code - properly record footprints and iterVar?
					// for (auto is=footprints.begin(); is!=footprints.end(); is++) {
					// PDBG("footprints-Key: " + ForLoop + " -- Value: " + footprints[ForLoop])
					// for (auto is=hasIterVar.begin(); is!=hasIterVar.end(); is++) {
					// PDBG("hasIterVar-Key: " + ForLoop + " -- Value: " + hasIterVar[ForLoop])

					// cache size --> cmdline parameter?, if hasIterVar is set
					if (footprints[ForLoop] > CACHE_SIZE && hasIterVar[ForLoop]) {
						// cache contention --> rewrite ForStmt
						PDBG("Cache thrashing is detected..")

						int tfactor = 0;
						int t_fpt = footprints[ForLoop];
						std::string w_limiter;

						for (int warps_divider=2; warps_divider<=64; warps_divider*=2) {
							if ( ((t_fpt / warps_divider) < CACHE_SIZE ) && (warps_divider <= op_blksize) ) {
								tfactor = warps_divider;
								w_limiter = std::to_string(op_blksize / tfactor);
								break;
							}
						}
						// PDBG(std::to_string(op_blksize) + " -- " + w_limiter + " -- " + std::to_string(t_fpt))
						if (!tfactor) {
							PDBG("Footprints is too large to throttle -- ignored for transformation")
							return ;
						}

						std::string thr_str = "\nint TID = (threadIdx.x+threadIdx.y*blockDim.x+threadIdx.z*blockDim.y*blockDim.x);\nfor(int tt_iter=0; tt_iter<"+std::to_string(tfactor)+"; tt_iter++)\nif (TID/32 >= tt_iter*"+w_limiter+" && TID/32 < (tt_iter+1)*"+w_limiter+") {\n";
						Rewrite.InsertText(ForLoop->getBeginLoc().getLocWithOffset(-1), "/* throttling start */"+thr_str, true, true);
						Rewrite.InsertText(ForLoop->getEndLoc().getLocWithOffset(1), "\n} /* throttling end */", true, true);
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
				std::map<std::string, int> varAss;
				std::map<ForStmt *, bool> hasIterVar;
				bool isGlobalVar;
				bool isGPUKernel;
			};

			// Implementation of the ASTConsumer interface for reading an AST produced
			// by the Clang parser. It registers a couple of matchers and runs them on
			// the AST.
			class ThrottlingASTConsumer : public ASTConsumer {
				public:
					ThrottlingASTConsumer(Rewriter &R) : HandlerForTT(R) {
						// to distiguish cuda function and others
						Matcher.addMatcher(
								functionDecl().bind("funcDecl"),
								&HandlerForTT);

						// to collect global variables
						Matcher.addMatcher(
								parmVarDecl().bind("parmVarDecl"),
								&HandlerForTT);

						Matcher.addMatcher(
								varDecl(has(integerLiteral().bind("integerVar0"))),
								&HandlerForTT);

						Matcher.addMatcher(
								binaryOperator(
									hasOperatorName("="), 
									has(integerLiteral().bind("integerVar1"))
									),
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
										hasIndex(implicitCastExpr().bind("pattern1")), // array[var], array[0]
										// hasIndex(binaryOperator().bind("pattern2")) // array[var * N + ..]
										forEachDescendant(
											declRefExpr(hasParent(implicitCastExpr(
														hasParent(binaryOperator(hasOperatorName("*")))))
												).bind("pattern2_1")
											)
										),
									hasAncestor(forStmt())
									),
								&HandlerForTT);
						Matcher.addMatcher(
								arraySubscriptExpr(
									forEachDescendant(
										declRefExpr(
											hasParent(implicitCastExpr(hasParent(binaryOperator(
															anyOf(
																hasOperatorName("+"),
																hasOperatorName("-")
																)
															))))).bind("pattern2_2")
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
					ForStmtHandler HandlerForTT;
					MatchFinder Matcher;
			};

			// For each source file provided to the tool, a new FrontendAction is created.
			class ThrottlingAction : public ASTFrontendAction {
				public:
					ThrottlingAction() {}
					void EndSourceFileAction() override {

						TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID()).write(llvm::outs());
					}

					std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, 
							StringRef file) override {
						TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
						return llvm::make_unique<ThrottlingASTConsumer>(TheRewriter);
						// c++14 feature, return std::make_unique<ThrottlingASTConsumer>(TheRewriter);
					}

				private:
					Rewriter TheRewriter;
			};

			int main(int argc, const char **argv) {
				CommonOptionsParser op(argc, argv, MatcherSampleCategory);
				ClangTool Tool(op.getCompilations(), op.getSourcePathList());

				PDBG("Thread block size : " + std::to_string(op_blksize) + " warps")
				PDBG(std::to_string(op_nblks) + " thread blocks per SM is set")
				PDBG("Cache size: " + std::to_string(op_csize) + "KB")

				return Tool.run(newFrontendActionFactory<ThrottlingAction>().get());
			}
