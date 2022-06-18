//
//  NSD_QQ_solver.cpp
//  NSDv2.0
//
//  Created by Shuotao Diao on 8/5/21.
//

#include "NSD_QQ_solver.hpp"

// declare global variables (need to be defined in the source file)
double QQ_SOLVER_PRECISION_LOWER = -1e-6;
double QQ_SOLVER_PRECISION_UPPER = 1e-6;

// check two face
bool if_face_equal(const face& face1, const face& face2) {
    if (face1.axis == face2.axis) {
        return true;
    }
    else {
        return false;
    }
}

// check if the face is new
bool if_face_new(const std::vector<face>& faces, const std::vector<int>& indices, const face& face_candidate) {
    if (indices.size() < 1) {
        return true;
    }
    for (int idx = 0; idx < indices.size(); ++idx) {
        if (if_face_equal(faces[indices[idx]], face_candidate) == true) {
            return false;
        }
    }
    return true;
}

bool if_face_new(const std::vector<face>& faces, const face& face_candidate) {
    if (faces.size() < 1) { // vector "faces" have no face
        return true;
    }
    for (int idx = 0; idx < faces.size(); ++idx) {
        if (if_face_equal(faces[idx], face_candidate)) {
            return false;
        }
    }
    return true;
}

// obtain dual multiplers of the second stage, given x (first stage decision variable)
dualMultipliers_QP twoStageQP_secondStagePrimal(const std::vector<double>& x, standardTwoStageParameters_QP& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap) {
    // obtain the sizes of input parameters
    long d_size = model_parameters.d.num_entry;
    long D_rowsize = model_parameters.D.num_row;
    long D_colsize = model_parameters.D.num_col;
    long equality_size = D_rowsize; // number of equality constraints
    // set up the model
    IloEnv env;
    IloModel mod(env);
    IloNumVarArray y(env,D_colsize,-IloInfinity,IloInfinity,ILOFLOAT); // second stage decision variables
    mod.add(y);
    // objective function
    IloExpr expr_obj(env);
    // quadratic part
    for (std::map<std::pair<int,int>, double>::iterator it = model_parameters.P.mat.begin(); it != model_parameters.P.mat.end(); ++it) {
        expr_obj += y[(it -> first).first] * y[(it -> first).second] * (it -> second) * 0.5;
    }
    // linear part
    for (auto it = model_parameters.d.vec.begin(); it != model_parameters.d.vec.end(); ++it) {
        expr_obj += (it -> second) * y[it -> first];
    }
    IloObjective obj = IloMinimize(env,expr_obj);
    mod.add(obj);
    // equality constraints Dy + Cx = e
    IloRangeArray constraintsEquality(env);
    std::vector<IloExpr> exprs_eq;
    for (int index_eq = 0; index_eq < model_parameters.D.num_row; ++index_eq) {
        IloExpr expr(env);
        exprs_eq.push_back(expr);
    }
    // coefficients before y
    for (std::map<std::pair<int,int>, double>::iterator it = model_parameters.D.mat.begin(); it != model_parameters.D.mat.end(); ++it) {
        // get the location of the entry
        exprs_eq[(it -> first).first] += (it -> second) * y[(it -> first).second];
    }
    // coefficients before x (deterministic part)
    for (std::map<std::pair<int,int>, double>::iterator it = model_parameters.C.mat.begin(); it != model_parameters.C.mat.end(); ++it) {
        // get the location of the entry
        exprs_eq[(it -> first).first] += (it -> second) * x[(it -> first).second];
    }
    // right hand side (deterministic part)
    for (auto it = model_parameters.e.vec.begin(); it != model_parameters.e.vec.end(); ++it) {
        exprs_eq[it -> first] -= (it -> second);
    }
    // coefficients before x (stochastic part) equality (i.e., Cij * xj map: <i,j> )
    for (int idx_Ce = 0; idx_Ce < rhs.Ce.size(); ++idx_Ce) {
        exprs_eq[RHSmap.Ce_map[idx_Ce].first] += rhs.Ce[idx_Ce] * x[RHSmap.Ce_map[idx_Ce].second];
    }
    // right hand side (stochastic part) equality be_(i) equality
    for (int idx_be = 0; idx_be < rhs.be.size(); ++idx_be) {
        exprs_eq[RHSmap.be_map[idx_be]] -= rhs.be[idx_be];
    }
    // add the equality constraints
    for (int index_eq = 0; index_eq < model_parameters.D.num_row; ++index_eq) {
        constraintsEquality.add(exprs_eq[index_eq] == 0);
    }
    mod.add(constraintsEquality);
    // non-negativity constriants
    IloRangeArray constraintsNonnegativity(env);
    for (int index = 0; index < D_colsize; ++index) {
        IloExpr expr(env);
        expr += y[index];
        constraintsNonnegativity.add(expr >= 0);
    }
    mod.add(constraintsNonnegativity);
    // set up cplex solver
    IloCplex cplex(env);
    cplex.extract(mod);
    cplex.setOut(env.getNullStream());
    IloBool solvable_flag = cplex.solve();
    IloNumArray dual_equality(env);
    IloNumArray dual_nonnegativity(env);
    dualMultipliers_QP duals;
    if (solvable_flag == IloTrue) {
        duals.feasible_flag = true; // tell the subproblem is feasible for a given x, first stage decision variable
        if (equality_size > 0) {
            cplex.getDuals(dual_equality,constraintsEquality);
            for (int index_eq = 0; index_eq < equality_size; ++index_eq) {
                double pi_temp = dual_equality[index_eq]; // sign of dual in cplex is opposite
                duals.t.push_back(pi_temp);
            }
        }
        // s
        cplex.getDuals(dual_nonnegativity, constraintsNonnegativity);
        for (int index = 0; index < D_colsize; ++index) {
            double pi_temp = dual_nonnegativity[index];
            duals.s.push_back(pi_temp);
        }
    }
    else {
        duals.feasible_flag = false; // tell the subproblem is infeasible for given x
    }
    env.end();
    return duals;
}


// obtain dual multipliers of the second stage, given x and face
dualMultipliers_QP twoStageQP_secondStageDual(const std::vector<double>& x, standardTwoStageParameters_QP& model_parameters, const secondStageRHSpoint& rhs, const face& face_cur, const secondStageRHSmap& RHSmap) {
    std::vector<double> e(model_parameters.e.num_entry,0.0);
    // deterministic part
    for (auto it = model_parameters.e.vec.begin(); it != model_parameters.e.vec.end(); ++it) {
        e[it -> first] += (it -> second);
    }
    // stochastic part
    for (int idx_eq = 0; idx_eq < rhs.be.size(); ++idx_eq) {
        e[RHSmap.be_map[idx_eq]] += rhs.be[idx_eq];
    }
    // deterministic part
    std::vector<double> Cx = model_parameters.C * x;
    // stochastic part
    for (int idx_Ce = 0; idx_Ce < rhs.Ce.size(); ++idx_Ce) {
        Cx[RHSmap.Ce_map[idx_Ce].first] += rhs.Ce[idx_Ce] * x[RHSmap.Ce_map[idx_Ce].second];
    }
    std::vector<double> e_Cx = e - Cx;
    // obtain the sizes of input parameters
    long d_size = model_parameters.d.num_entry;
    long D_rowsize = model_parameters.D.num_row;
    long D_colsize = model_parameters.D.num_col;
    // set up the model
    IloEnv env;
    IloModel mod(env);
    // decision variable
    IloNumVarArray s(env,d_size,0,IloInfinity,ILOFLOAT);
    IloNumVarArray t(env,D_rowsize,-IloInfinity,IloInfinity,ILOFLOAT);
    IloNumVarArray r(env,d_size,-IloInfinity,IloInfinity,ILOFLOAT);
    mod.add(s);
    mod.add(t);
    mod.add(r);
    // objective
    IloExpr expr_obj(env);
    // objectve quadratic part
    for (std::map<std::pair<int,int>, double>::iterator it = model_parameters.P_inv.mat.begin(); it != model_parameters.P_inv.mat.end(); ++it) {
        expr_obj += r[(it -> first).first] * r[(it -> first).second] * (it -> second) * (-0.5);
    }
    // objective linear part
    for (int index = 0; index < e_Cx.size(); ++index) {
        expr_obj += e_Cx[index] * t[index];
    }
    IloObjective obj = IloMaximize(env, expr_obj);
    mod.add(obj);
    // add the equality constraints
    std::vector<IloExpr> exprs;
    for (int idx_eq = 0; idx_eq < d_size; ++idx_eq) {
        IloExpr expr(env);
        exprs.push_back(expr);
        exprs[idx_eq] += s[idx_eq] - r[idx_eq];
    }
    for (auto it = model_parameters.d.vec.begin(); it != model_parameters.d.vec.end(); ++it) {
        exprs[it -> first] -= (it -> second);
    }
    for (std::map<std::pair<int,int>, double>::iterator it = model_parameters.D_trans.mat.begin(); it != model_parameters.D_trans.mat.end(); ++it) {
        exprs[(it -> first).first] += (it -> second) * t[(it -> first).second];
    }
    for (int idx_eq = 0; idx_eq < d_size; ++idx_eq) {
        mod.add(exprs[idx_eq] == 0);
    }
    // constraints based on the given face
    for (int idx_face = 0; idx_face < face_cur.axis.size(); ++idx_face) {
        mod.add(s[face_cur.axis[idx_face]] == 0);
    }
    // dual multiplier QP
    dualMultipliers_QP duals;
    // set up cplex solver
    IloCplex cplex(env);
    cplex.extract(mod);
    cplex.setOut(env.getNullStream());
    IloBool solvable_flag = cplex.solve();
    for (int idx_s = 0; idx_s < d_size; ++idx_s) {
        duals.s.push_back(cplex.getValue(s[idx_s]));
    }
    for (int idx_t = 0; idx_t < D_rowsize; ++idx_t) {
        duals.t.push_back(cplex.getValue(t[idx_t]));
    }
    for (int idx_r = 0; idx_r < d_size; ++idx_r) {
        duals.r.push_back(cplex.getValue(r[idx_r]));
    }
    duals.obj_val = cplex.getObjValue();
    env.end();
    return duals;
}


// functions for generating feasibility cut
dualMultipliers_QP twoStageQP_secondStageExtremRay(const std::vector<double>& x, standardTwoStageParameters_QP& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap) {
    // obtain the sizes of input parameters
    long d_size = model_parameters.d.num_entry;
    long D_rowsize = model_parameters.D.num_row;
    long D_colsize = model_parameters.D.num_col;
    long equality_size = D_rowsize; // number of equality constraints
    // set up the model
    IloEnv env;
    IloModel mod(env);
    IloNumVarArray y(env,D_colsize,-IloInfinity,IloInfinity,ILOFLOAT); // second stage decision variables
    mod.add(y);
    // objective function
    IloExpr expr_obj(env);
    // quadratic part
    for (std::map<std::pair<int,int>, double>::iterator it = model_parameters.P.mat.begin(); it != model_parameters.P.mat.end(); ++it) {
        expr_obj += y[(it -> first).first] * y[(it -> first).second] * (it -> second) * 0.5;
    }
    // linear part
    for (auto it = model_parameters.d.vec.begin(); it != model_parameters.d.vec.end(); ++it) {
        expr_obj += (it -> second) * y[it -> first];
    }
    IloObjective obj = IloMinimize(env,expr_obj);
    mod.add(obj);
    // equality constraints Dy + Cx = e
    IloRangeArray constraintsEquality(env);
    std::vector<IloExpr> exprs_eq;
    for (int index_eq = 0; index_eq < model_parameters.D.num_row; ++index_eq) {
        IloExpr expr(env);
        exprs_eq.push_back(expr);
    }
    // coefficients before y
    for (std::map<std::pair<int,int>, double>::iterator it = model_parameters.D.mat.begin(); it != model_parameters.D.mat.end(); ++it) {
        // get the location of the entry
        exprs_eq[(it -> first).first] += (it -> second) * y[(it -> first).second];
    }
    // coefficients before x (deterministic part)
    for (std::map<std::pair<int,int>, double>::iterator it = model_parameters.C.mat.begin(); it != model_parameters.C.mat.end(); ++it) {
        // get the location of the entry
        exprs_eq[(it -> first).first] += (it -> second) * x[(it -> first).second];
    }
    // right hand side (deterministic part)
    for (auto it = model_parameters.e.vec.begin(); it != model_parameters.e.vec.end(); ++it) {
        exprs_eq[it -> first] -= (it -> second);
    }
    // coefficients before x (stochastic part) equality (i.e., Cij * xj map: <i,j> )
    for (int idx_Ce = 0; idx_Ce < rhs.Ce.size(); ++idx_Ce) {
        exprs_eq[RHSmap.Ce_map[idx_Ce].first] += rhs.Ce[idx_Ce] * x[RHSmap.Ce_map[idx_Ce].second];
    }
    // right hand side (stochastic part) equality be_(i) equality
    for (int idx_be = 0; idx_be < rhs.be.size(); ++idx_be) {
        exprs_eq[RHSmap.be_map[idx_be]] -= rhs.be[idx_be];
    }
    // add the equality constraints
    for (int index_eq = 0; index_eq < model_parameters.D.num_row; ++index_eq) {
        constraintsEquality.add(exprs_eq[index_eq] == 0);
    }
    mod.add(constraintsEquality);
    // non-negativity constriants
    IloRangeArray constraintsNonnegativity(env);
    for (int index = 0; index < D_colsize; ++index) {
        IloExpr expr(env);
        expr += y[index];
        constraintsNonnegativity.add(expr >= 0);
    }
    mod.add(constraintsNonnegativity);
    // set up cplex solver
    IloCplex cplex(env);
    cplex.extract(mod);
    cplex.setOut(env.getNullStream());
    cplex.setParam(IloCplex::PreInd, false); // need to turn off presolve in order to get dual extreme rays
    cplex.setParam(IloCplex::RootAlg, IloCplex::Dual); // use dual simplex optimizer
    IloBool solvable_flag = cplex.solve();
    IloNumArray dual_equality(env);
    IloNumArray dual_nonnegativity(env);
    dualMultipliers_QP duals;
    if (solvable_flag == IloTrue) {
        duals.feasible_flag = true; // tell the subproblem is feasible for a given x, first stage decision variable
        if (equality_size > 0) {
            cplex.getDuals(dual_equality,constraintsEquality);
            for (int index_eq = 0; index_eq < equality_size; ++index_eq) {
                double pi_temp = dual_equality[index_eq]; 
                duals.t.push_back(pi_temp);
            }
        }
        // s
        cplex.getDuals(dual_nonnegativity, constraintsNonnegativity);
        for (int index = 0; index < D_colsize; ++index) {
            double pi_temp = dual_nonnegativity[index];
            duals.s.push_back(pi_temp);
        }
    }
    else {
        duals.feasible_flag = false; // tell the subproblem is infeasible for given x
    }
    env.end();
    return duals;
}


feasibilityCut twoStageQP_feasibilityCutGeneration(const dualMultipliers_QP& extremeRay, standardTwoStageParameters_QP& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap) {
    // compute -d + D^\top t + s
    std::vector<double> Dt = model_parameters.D_trans * extremeRay.t;
    std::vector<double> DtS = Dt + extremeRay.s;
    sparseVector dDtS = negate(model_parameters.d) + DtS;
    //
    sparseVector dDtSP = dDtS * model_parameters.P_inv;
    double dDtSPdDtS = dDtSP * dDtS;
    // initialize feasibility cut
    feasibilityCut cut_scenario;
    std::vector<double> e(model_parameters.e.num_entry,0.0);
    // deterministic part
    for (auto it = model_parameters.e.vec.begin(); it != model_parameters.e.vec.end(); ++it) {
        e[it -> first] += (it -> second);
    }
    // stochastic part
    for (int idx_eq = 0; idx_eq < rhs.be.size(); ++idx_eq) {
        e[RHSmap.be_map[idx_eq]] += rhs.be[idx_eq];
    }
    cut_scenario.b_newRow = 0.5 * dDtSPdDtS - e * extremeRay.t;
    // slope
    // deterministic part
    cut_scenario.A_newRow = (-1.0) * (extremeRay.t * model_parameters.C);
    // stochastic part
    // equality
    for (int idx_Ce = 0; idx_Ce < rhs.Ce.size(); ++idx_Ce) {
        cut_scenario.A_newRow[RHSmap.Ce_map[idx_Ce].second] += -1.0 * rhs.Ce[idx_Ce] * extremeRay.t[RHSmap.Ce_map[idx_Ce].first];
    }
    return cut_scenario;
}


// projection for the first stage in the two stage linear programming
std::vector<double> twoStageQP_projection(const std::vector<double>& x, standardTwoStageParameters_QP& model_parameters) {
    std::vector<double> x_projected(x.size(),0.0);
    // solve a quadratic programming
    IloEnv env;
    IloModel mod(env);
    IloNumVarArray x_temp(env,x.size(),-IloInfinity,IloInfinity,ILOFLOAT);
    mod.add(x_temp);
    IloExpr expr_obj(env);
    for (int x_index = 0; x_index < x.size(); ++x_index) {
        expr_obj += x_temp[x_index] * x_temp[x_index] - 2.0 * x_temp[x_index] * x[x_index];
    }
    IloObjective obj = IloMinimize(env,expr_obj); // objective function
    mod.add(obj);
    // constraints
    std::vector<IloExpr> exprs;
    for (int index_cons = 0; index_cons < model_parameters.A.num_row; ++index_cons) {
        IloExpr expr(env);
        exprs.push_back(expr);
    }
    for (std::map<std::pair<int,int>, double>::iterator it = model_parameters.A.mat.begin(); it != model_parameters.A.mat.end(); ++it) {
        // get the location of the entry
        exprs[(it -> first).first] += (it -> second) * x_temp[(it -> first).second];
    }
    // right hand side
    for (auto it = model_parameters.b.vec.begin(); it != model_parameters.b.vec.end(); ++it) {
        exprs[it -> first] -= (it -> second);
    }
    // add constraints
    for (int index_cons = 0; index_cons < model_parameters.A.num_row; ++index_cons) {
        mod.add(exprs[index_cons] <= 0);
    }
    // create cplex environment
    IloCplex cplex(env);
    cplex.extract(mod);
    cplex.setOut(env.getNullStream());
    cplex.solve();
    // obtain the projected point
    for (int x_index = 0; x_index < model_parameters.A.num_row; ++x_index) {
        x_projected[x_index] = cplex.getValue(x_temp[x_index]);
        //std::cout << cplex.getValue(x_temp[x_index]) << std::endl;
    }
    env.end();
    return x_projected;
}


// presolve
std::vector<double> twoStageQP_presolve(standardTwoStageParameters_QP& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap) {
    std::vector<double> x_candidate(model_parameters.c.num_entry,0.0);
    long equality_size = model_parameters.D.num_row; // number of equality constraints
    // solve a quadratic programming
    IloEnv env;
    IloModel mod(env);
    IloNumVarArray x_temp(env,model_parameters.c.num_entry,-IloInfinity,IloInfinity,ILOFLOAT);
    IloNumVarArray y(env,model_parameters.d.num_entry,0,IloInfinity,ILOFLOAT); // y >= 0
    mod.add(x_temp);
    mod.add(y);
    IloExpr expr_obj(env);
    // first stage objective linear
    for (auto it = model_parameters.c.vec.begin(); it != model_parameters.c.vec.end(); ++it) {
        expr_obj += (it -> second) * x_temp[it -> first];
    }
    // first stage objective quadratic
    for (std::map<std::pair<int,int>, double>::iterator it = model_parameters.Q.mat.begin(); it != model_parameters.Q.mat.end(); ++it) {
        expr_obj += x_temp[(it -> first).first] * x_temp[(it -> first).second] * (it -> second) * 0.5;
    }
    // second stage objective linear
    for (auto it = model_parameters.d.vec.begin(); it != model_parameters.d.vec.end(); ++it) {
        expr_obj += (it -> second) * y[it -> first];
    }
    // second stage objective quadratic
    for (std::map<std::pair<int,int>, double>::iterator it = model_parameters.P.mat.begin(); it != model_parameters.P.mat.end(); ++it) {
        expr_obj += y[(it -> first).first] * y[(it -> first).second] * (it -> second) * 0.5;
    }
    IloObjective obj = IloMinimize(env,expr_obj); // objective function
    mod.add(obj);
    // constraints
    // first stage
    std::vector<IloExpr> exprs;
    for (int index_cons = 0; index_cons < model_parameters.A.num_row; ++index_cons) {
        IloExpr expr(env);
        exprs.push_back(expr);
    }
    for (std::map<std::pair<int,int>, double>::iterator it = model_parameters.A.mat.begin(); it != model_parameters.A.mat.end(); ++it) {
        // get the location of the entry
        exprs[(it -> first).first] += (it -> second) * x_temp[(it -> first).second];
    }
    // right hand side
    for (auto it = model_parameters.b.vec.begin(); it != model_parameters.b.vec.end(); ++it) {
        exprs[it -> first] -= (it -> second);
    }
    // add constraints
    for (int index_cons = 0; index_cons < model_parameters.A.num_row; ++index_cons) {
        mod.add(exprs[index_cons] <= 0);
    }
    // second stage
    // equality constraints
    IloRangeArray constraintsEquality(env);
    std::vector<IloExpr> exprs_eq;
    for (int index_eq = 0; index_eq < model_parameters.D.num_row; ++index_eq) {
        IloExpr expr(env);
        exprs_eq.push_back(expr);
    }
    // coefficients before y
    for (std::map<std::pair<int,int>, double>::iterator it = model_parameters.D.mat.begin(); it != model_parameters.D.mat.end(); ++it) {
        // get the location of the entry
        exprs_eq[(it -> first).first] += (it -> second) * y[(it -> first).second];
    }
    // coefficients before x (deterministic part)
    for (std::map<std::pair<int,int>, double>::iterator it = model_parameters.C.mat.begin(); it != model_parameters.C.mat.end(); ++it) {
        // get the location of the entry
        exprs_eq[(it -> first).first] += (it -> second) * x_temp[(it -> first).second];
    }
    // right hand side (deterministic part)
    for (auto it = model_parameters.e.vec.begin(); it != model_parameters.e.vec.end(); ++it) {
        exprs_eq[it -> first] -= (it -> second);
    }
    // coefficients before x (stochastic part) equality (i.e., Cij * xj map: <i,j> )
    for (int idx_Ce = 0; idx_Ce < rhs.Ce.size(); ++idx_Ce) {
        exprs_eq[RHSmap.Ce_map[idx_Ce].first] += rhs.Ce[idx_Ce] * x_temp[RHSmap.Ce_map[idx_Ce].second];
    }
    // right hand side (stochastic part) equality be_(i) equality
    for (int idx_be = 0; idx_be < rhs.be.size(); ++idx_be) {
        exprs_eq[RHSmap.be_map[idx_be]] -= rhs.be[idx_be];
    }
    // add the equality constraints
    for (int index_eq = 0; index_eq < model_parameters.D.num_row; ++index_eq) {
        constraintsEquality.add(exprs_eq[index_eq] == 0);
    }
    mod.add(constraintsEquality);
    // set up cplex solver
    IloCplex cplex(env);
    cplex.extract(mod);
    cplex.setOut(env.getNullStream());
    //cplex.exportModel("/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99small/experiment1/case1/presolve_model.qp");
    IloBool solvable_flag = cplex.solve();
    // obtain the projected point
    for (int x_index = 0; x_index < model_parameters.A.num_col; ++x_index) {
        x_candidate[x_index] = cplex.getValue(x_temp[x_index]);
        //std::cout << cplex.getValue(x_temp[x_index]) << std::endl;
    }
    env.end();
    return x_candidate;
}


// incumbent selection
bool incumbent_selection_check_QP(double q, const std::vector<double>& x_candidate, const std::vector<double>& x_incumbent, standardTwoStageParameters_QP& model_parameters, const std::vector<minorant>& minorants, const std::vector<minorant>& minorants_new, const std::vector<int>& active_minorants) {
    double max_value_candidate = -99999;
    double max_value_new_candidate = -99999;
    double max_value_incumbent = -99999;
    double max_value_new_incumbent = -99999;
    for (int index = 0; index < active_minorants.size(); ++index) {
        double temp_candidate = minorants[active_minorants[index]].alpha;
        temp_candidate += minorants[active_minorants[index]].beta * x_candidate;
        if (index == 0) {
            max_value_candidate = temp_candidate;
        }
        else if (temp_candidate > max_value_candidate) {
            max_value_candidate = temp_candidate;
        }
        double temp_incumbent = minorants[active_minorants[index]].alpha;
        temp_incumbent += minorants[active_minorants[index]].beta * x_incumbent;
        if (index == 0) {
            max_value_incumbent = temp_incumbent;
        }
        else if (temp_incumbent > max_value_incumbent) {
            max_value_incumbent = temp_incumbent;
        }
    }
    for (int index = 0; index < minorants_new.size(); ++index) {
        double temp_candidate = minorants_new[index].alpha;
        temp_candidate += minorants_new[index].beta * x_candidate;
        if (index == 0) {
            max_value_new_candidate = temp_candidate;
        }
        else if (temp_candidate > max_value_new_candidate) {
            max_value_new_candidate = temp_candidate;
        }
        double temp_incumbent = minorants_new[index].alpha;
        temp_incumbent += minorants_new[index].beta * x_incumbent;
        if (index == 0) {
            max_value_new_incumbent = temp_incumbent;
        }
        else if (temp_incumbent > max_value_new_incumbent) {
            max_value_new_incumbent = temp_incumbent;
        }
    }
    double delta_first_stage = model_parameters.c * x_candidate;
    delta_first_stage += 0.5 * (x_candidate * model_parameters.Q * x_candidate);
    delta_first_stage -= model_parameters.c * x_incumbent;
    delta_first_stage -= 0.5 * (x_incumbent * model_parameters.Q * x_incumbent);
    double delta = max_value_candidate - max_value_incumbent + delta_first_stage;
    double delta_new = max_value_new_candidate - max_value_new_incumbent + delta_first_stage;
    std::cout << "=====================\n";
    std::cout << "delta_new = " << delta_new << std::endl;
    std::cout << "q * delta = " << q * delta << std::endl;
    std::cout << "=====================\n";
    if (delta_new <= q * delta) {
        return true;
    }
    else {
        return false;
    }
    return false;
}


// NSD QP solver with presolve and unique duals (initial k will be based on the presolve)
std::vector<double> dynamic_sdknn_qq_solver_presolve_fullFace(const std::string& folder_path, int max_iterations, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug) {
    std::vector<double> x_candidate;
    std::vector<double> x_incumbent;
    // STEP 1: INITIALIZATION
    // algorithm parameters
    double sigma = 1.0;
    double q = 0.5;
    //double beta = 0.5;
    double beta = 0.5; // 0 < beta < 1
    int k = 1;
    int k_new = 1;
    int N = 0;
    std::vector<double> distanceSet;
    std::vector<int> orderSet;
    std::vector<int> kNNSet;
    bool flag_be; // tell if be is generated
    bool flag_bi; // tell if bi is generated
    bool flag_Ce; // tell if Ce stochastic is generated
    bool flag_Ci; // tell if Ci stochastic is generated 
    std::vector<secondStageRHSpoint> RHS_dataset;
    // create directory paths for database and model
    std::string be_DB_path = folder_path + "/be_DB.txt";
    std::string bi_DB_path = folder_path + "/bi_DB.txt";
    std::string Ce_DB_path = folder_path + "/Ce_DB.txt";
    std::string Ci_DB_path = folder_path + "/Ci_DB.txt";
    std::string model_path = folder_path + "/model.txt";
    std::string sto_path = folder_path + "/sto.txt";
    std::string resultsOutput_path = folder_path + "/computationalResults(NSD_QQv2.0presolve).txt";
    // convert all the paths into constant chars
    const char* be_DB_path_const = be_DB_path.c_str();
    const char* bi_DB_path_const = bi_DB_path.c_str();
    const char* Ce_DB_path_const = Ce_DB_path.c_str();
    const char* Ci_DB_path_const = Ci_DB_path.c_str();
    // create stream object
    std::ifstream readFile_be(be_DB_path_const);
    std::ifstream readFile_bi(bi_DB_path_const);
    std::ifstream readFile_Ce(Ce_DB_path_const);
    std::ifstream readFile_Ci(Ci_DB_path_const);
    // create database
    std::vector<std::vector<dataPoint>> be_DB;
    std::vector<std::vector<dataPoint>> bi_DB;
    std::vector<std::vector<dataPoint>> Ce_DB;
    std::vector<std::vector<dataPoint>> Ci_DB;
    // create model structure
    standardTwoStageParameters_QP model_parameters;
    // create sto object
    secondStageRHSmap RHSmap;
    // read  be
    if (readFile_be.is_open()) {
        std::cout << "be_DB is found." << std::endl;
        readFile_be.close(); // close the file
        // read be database
        be_DB = readNonparametricDB(be_DB_path);
        flag_be = true;
    }
    else {
        readFile_be.close(); // close the file
        flag_be = false;
        std::cout << "be_DB is not found!" << std::endl;
    }
    // read bi
    if (readFile_bi.is_open()) {
        std::cout << "bi_DB is found." << std::endl;
        readFile_be.close(); // close the file
        // read bi database
        bi_DB = readNonparametricDB(bi_DB_path);
        flag_bi = true;
    }
    else {
        readFile_bi.close(); // close the file
        flag_bi = false;
        std::cout << "bi_DB is not found!" << std::endl;
    }
    // read Ce
    if (readFile_Ce.is_open()) {
        std::cout << "Ce_DB stochastic part is found." << std::endl;
        readFile_Ce.close(); // close the file
        // Ce database
        Ce_DB = readNonparametricDB(Ce_DB_path);
        flag_Ce = true;
    }
    else {
        readFile_Ce.close(); // close the file
        flag_Ce = false;
        std::cout << "Ce_DB is not found!" << std::endl;
    }
    // read Ci
    if (readFile_Ci.is_open()) {
        std::cout << "Ci_DB stochastic part is found." << std::endl;
        readFile_Ci.close(); // close the file
        // Ci database
        Ci_DB = readNonparametricDB(Ci_DB_path);
        flag_Ci = true;
    }
    else {
        readFile_Ci.close(); // close the file
        flag_Ci = false;
        std::cout << "Ci_DB is not found!" << std::endl;
    }
    // read model
    model_parameters = readStandardTwoStageParameters_QP(model_path);
    // read sto object
    RHSmap = readStochasticMap(sto_path);
    // STEP 2: SOLVING PROCESS (SD-kNN)
    // initialize feasibility cut collection
    std::vector<feasibilityCut> feasibility_cuts;
    // timer
    std::clock_t time_start;
    time_start = std::clock();
    // current time
    std::time_t currTime = std::time(nullptr);
    // initialization of output file
    const char* writeFilePath = resultsOutput_path.c_str();
    std::fstream writeFile;
    writeFile.open(writeFilePath,std::fstream::app); // append results to the end of the file
    //
    // write initial setup
    std::cout << "*******************************************\n";
    writeFile << "*******************************************\n";
    std::cout << "SD-kNN-QQ (fast version with presolve 2.0) is initialized\n";
    writeFile << "SD-kNN-QQ (fast version with presolve 2.0) is initialized\n";
    std::cout << std::put_time(localtime(&currTime), "%c %Z") << "\n";
    writeFile << std::put_time(localtime(&currTime), "%c %Z") << "\n"; // write current time
    std::cout << "Algorithmic Parameters\n";
    writeFile << "Algorithmic Parameters\n";
    std::cout << "sigma, q, beta, k, N, N_pre, sigma_lower, sigma_upper" << std::endl;
    writeFile << "sigma, q, beta, k, N, N_pre, sigma_lower, sigma_upper" << std::endl;
    std::cout << sigma << ", " << q << ", " << beta << ", " << k << ", " << N << ", " << N_pre << ", " << sigma_lowerbound << ", " << sigma_upperbound << std::endl;
    writeFile << sigma << ", " << q << ", " << beta << ", " << k << ", " << N << ", " << N_pre << ", " << sigma_lowerbound << ", " << sigma_upperbound << std::endl;
    std::cout << "Problem Complexity" << std::endl;
    writeFile << "Problem Complexity" << std::endl;
    std::cout << "Q_num_row, Q_num_col, P_num_row, P_num_col\n";
    writeFile << "Q_num_row, Q_num_col, P_num_row, P_num_col\n";
    std::cout << model_parameters.Q.num_row << ", " << model_parameters.Q.num_col << ", " << model_parameters.P.num_row << ", " << model_parameters.P.num_col << std::endl;
    writeFile << model_parameters.Q.num_row << ", " << model_parameters.Q.num_col << ", " << model_parameters.P.num_row << ", " << model_parameters.P.num_col << std::endl;
    std::cout << "A_num_row, A_num_col\n";
    writeFile << "A_num_row, A_num_col\n";
    std::cout << model_parameters.A.num_row << ", " << model_parameters.A.num_col << std::endl;
    writeFile << model_parameters.A.num_row << ", " << model_parameters.A.num_col << std::endl;
    std::cout << "D_num_row, D_num_col\n";
    writeFile << "D_num_row, D_num_col\n";
    std::cout << model_parameters.D.num_row << ", " << model_parameters.D.num_col << std::endl;
    writeFile << model_parameters.D.num_row << ", " << model_parameters.D.num_col << std::endl;
    // set up initial incumbent solution
    long x_size = model_parameters.c.num_entry;
    // size of parameters
    //long c_size = model_parameters.c.num_entry;
    long A_rowsize = model_parameters.A.num_row;
    long A_colsize = model_parameters.A.num_col;
    std::cout << "Observed Predictor: ";
    writeFile << "Observed Predictor: ";
    for (int predictor_index = 0; predictor_index < observed_predictor.size() - 1; ++predictor_index) {
        std::cout << observed_predictor[predictor_index] << ", ";
        writeFile << observed_predictor[predictor_index] << ", ";
    }
    std::cout << observed_predictor[observed_predictor.size() - 1] << std::endl;
    writeFile << observed_predictor[observed_predictor.size() - 1] << std::endl;
    // PRESEOLVE PROCESS
    // initialize incumbent solution in the first stage
    std::cout << "===PRESOLVE PROCESS===\n";
    writeFile << "===PRESOLVE PROCESS===\n";
    // find the kNN set
    for (int idx_pre = 0; idx_pre < N_pre; ++idx_pre) {
        k_new = (int) pow(N_pre, beta);
        // obtain a new data point
        dataPoint be_datapoint;
        if (flag_be == true) {
            be_datapoint = be_DB[0][idx_pre];
        }
        dataPoint bi_datapoint;
        if (flag_bi == true) {
            bi_datapoint = bi_DB[0][idx_pre];
        }
        dataPoint Ce_datapoint;
        if (flag_Ce == true) {
            Ce_datapoint = Ce_DB[0][idx_pre];
        }
        dataPoint Ci_datapoint;
        if (flag_Ci == true) {
            Ci_datapoint = Ci_DB[0][idx_pre];
        }
        // merge all the datapoints
        secondStageRHSpoint RHS_datapoint = merge_randomVector(be_datapoint, bi_datapoint, Ce_datapoint, Ci_datapoint);
        RHS_dataset.push_back(RHS_datapoint);
        // calculate the squared distance
        double distance_squared = 0;
        for (int idx = 0; idx < RHS_datapoint.predictor.size(); ++idx) {
            distance_squared += (RHS_datapoint.predictor[idx] - observed_predictor[idx]) * (RHS_datapoint.predictor[idx] - observed_predictor[idx]);
        }
        distanceSet.push_back(distance_squared);
        // store the new squared distance
        // sorting (like insert sorting)
        if (idx_pre == 0) { // first iteration
            orderSet.push_back(1);
            kNNSet.push_back(0);
        }
        else { // from left to right in increasing order
            int left_index = 0; // the index corresponds to the largest distance that is smaller than the current one
            double left_distance = -1;
            // double indices used for tie-breaking
            int right_index = -1; // the index corresponds to the smallest distance that is larger than  the current one
            double right_distance = -1;
            for (int index = 0; index < orderSet.size(); ++index) {
                if (distanceSet[index] < distance_squared) {
                    if (left_index == 0) {
                        left_distance = distanceSet[index];
                        left_index = orderSet[index];
                    }
                    else if (distanceSet[index] > left_distance) {
                        left_distance = distanceSet[index];
                        left_index = orderSet[index];
                    }
                }
                if (distanceSet[index] > distance_squared) {
                    if (right_index == -1) {
                        right_distance = distanceSet[index];
                        right_index = orderSet[index];
                    }
                    else if (distanceSet[index] < right_distance) {
                        right_distance = distanceSet[index];
                        right_index = orderSet[index];
                    }
                }
            }
            /*
            if (flag_debug == true) {
                std::cout << "Output double indices\n";
                writeFile << "Output double indices\n";
                std::cout << "left index: " << left_index << std::endl;
                writeFile << "left index: " << left_index << std::endl;
                std::cout << "right index: " << right_index << std::endl;
                writeFile << "right index: " << right_index << std::endl;
            }
             */
            // update the orderSet
            for (int index = 0; index < orderSet.size(); ++index) {
                if (right_index != -1 && orderSet[index] >= right_index) {
                    orderSet[index] = orderSet[index] + 1;
                }
                //if (left_index == 0) { // current one is the nearest neighbor
                //    orderSet[index] = orderSet[index] + 1;
                //}
                //else if (orderSet[index] > left_index) {
                //    orderSet[index] = orderSet[index] + 1;
                //}
            }
            if (right_index == -1) {
                orderSet.push_back((int) orderSet.size() + 1);
            }
            else {
                orderSet.push_back(right_index);
            }
            /*
            if (flag_debug == true) {
                std::cout << "Updated Order in the scenario set\n";
                writeFile << "Updated Order in the scenario set\n";
                std::cout << "Index, Order, Distance (Squared)\n";
                writeFile << "Index, Order, Distance (Squared)\n";
                // update the kNN set
                for (int index = 0; index < orderSet.size(); ++index) {
                    std::cout << index << ", "<< orderSet[index] << ", " << distanceSet[index];
                    writeFile << index << ", "<< orderSet[index] << ", " << distanceSet[index];
                    if (orderSet[index] <= k_new) {
                        std::cout << "*";
                        writeFile << "*";
                        //kNNSet_new.push_back(index);
                    }
                    std::cout << std::endl;
                    writeFile << std::endl;
                }
            }
            else {
                // update the kNN set
                for (int index = 0; index < orderSet.size(); ++index) {
                    if (orderSet[index] <= k_new) {
                        //kNNSet_new.push_back(index);
                    }
                }
            }
             */
            // update the kNN set
            kNNSet.clear(); // clear the old kNN set
            for (int index = 0; index < orderSet.size(); ++index) {
                if (orderSet[index] <= k_new) {
                    kNNSet.push_back(index);
                }
            }
        }
    }
    // calculate the kNN point estimate
    secondStageRHSpoint knn_point_estimate;
    if (flag_be == true) { // be sto part exists
        // initialize point estimate
        for (int idx = 0; idx < RHS_dataset[0].be.size(); ++idx) {
            knn_point_estimate.be.push_back(0.0);
        }
        for (int idx_knn = 0; idx_knn < kNNSet.size(); ++idx_knn) {
            for (int idx_component = 0; idx_component < RHS_dataset[0].be.size(); ++idx_component) {
                knn_point_estimate.be[idx_component] += (1.0 / (double) k_new) * RHS_dataset[kNNSet[idx_knn]].be[idx_component];
            }
        }
    }
    if (flag_bi == true) { // bi sto part exists
        for (int idx = 0; idx < RHS_dataset[0].bi.size(); ++idx) {
            knn_point_estimate.bi.push_back(0.0);
        }
        for (int idx_knn = 0; idx_knn < kNNSet.size(); ++idx_knn) {
            for (int idx_component = 0; idx_component < RHS_dataset[0].bi.size(); ++idx_component) {
                knn_point_estimate.bi[idx_component] += (1.0 / (double) k_new) * RHS_dataset[kNNSet[idx_knn]].bi[idx_component];
            }
        }
    }
    if (flag_Ce == true) { // Ce sto part exists
        for (int idx = 0; idx < RHS_dataset[0].Ce.size(); ++idx) {
            knn_point_estimate.Ce.push_back(0.0);
        }
        for (int idx_knn = 0; idx_knn < kNNSet.size(); ++idx_knn) {
            for (int idx_component = 0; idx_component < RHS_dataset[0].Ce.size(); ++idx_component) {
                knn_point_estimate.Ce[idx_component] += (1.0 / (double) k_new) * RHS_dataset[kNNSet[idx_knn]].Ce[idx_component];
            }
        }
    }
    if (flag_Ci == true) { // Ci sto part exists
        for (int idx = 0; idx < RHS_dataset[0].Ci.size(); ++idx) {
            knn_point_estimate.Ci.push_back(0.0);
        }
        for (int idx_knn = 0; idx_knn < kNNSet.size(); ++idx_knn) {
            for (int idx_component = 0; idx_component < RHS_dataset[0].Ci.size(); ++idx_component) {
                knn_point_estimate.Ci[idx_component] += (1.0 / (double) k_new) * RHS_dataset[kNNSet[idx_knn]].Ci[idx_component];
            }
        }
    }
    // presolve the QP problem to get x_incumbent
    x_incumbent = twoStageQP_presolve(model_parameters, knn_point_estimate, RHSmap);
    std::cout << "Incumbent solution after presolve:\n";
    writeFile << "Incumbent solution after presolve:\n";
    for (int idx_x = 0; idx_x < x_incumbent.size() - 1; ++idx_x) {
        std::cout << x_incumbent[idx_x] << ", ";
        writeFile << x_incumbent[idx_x] << ", ";
    }
    std::cout << x_incumbent[x_incumbent.size() - 1] << std::endl;
    writeFile << x_incumbent[x_incumbent.size() - 1] << std::endl;
    // obtain faces at the presolve points
    // explored faces in the second stage
    // *******
    // *******
    std::vector<face> explored_faces;
    for (int idx_knn = 0; idx_knn < kNNSet.size(); ++idx_knn) {
        dualMultipliers_QP new_dual = twoStageQP_secondStagePrimal(x_incumbent, model_parameters, RHS_dataset[kNNSet[idx_knn]], RHSmap);
        if (new_dual.feasible_flag == false) { // if second stage subproblem is infeasible
            //
            std::cout << "Warning: An infeasible case is captured.\n";
            writeFile << "Warning: An infeasible case is captured.\n";
            std::cout << "Warning: A feasibility cut will be constructed.\n";
            writeFile << "Warning: A feasibility cut will be constructed.\n";
            dualMultipliers_QP extremeRay_scenario = twoStageQP_secondStageExtremRay(x_candidate, model_parameters, RHS_dataset[kNNSet[idx_knn]], RHSmap);
            feasibilityCut feasibilityCut_scenario = twoStageQP_feasibilityCutGeneration(extremeRay_scenario, model_parameters, RHS_dataset[kNNSet[idx_knn]], RHSmap);
            // add feasibility cut
            feasibility_cuts.push_back(feasibilityCut_scenario);
        }
        else {
            // second stage subproblem is feasible
            //std::cout << "Computation Log: Subproblem in datapoint " << idx << " is feasible.\n";
            //writeFile << "Computation Log: Subproblem in datapoint " << idx << " is feasible.\n";
            /*
            // check if the new dual is found
            bool flag_new_dual_explored = if_new_dual(explored_duals, dualsTemp);
            if (flag_new_dual_explored == true) {
                std::cout << "Computation Log: New dual is found.\n";
                writeFile << "Computation Log: New dual is found.\n";
                explored_duals.push_back(dualsTemp);
            }*/
            // add faces
            face face_scenario;
            for (int index = 0; index < new_dual.s.size(); index++) {
                if (new_dual.s[index] >= QQ_SOLVER_PRECISION_LOWER && new_dual.s[index] <= QQ_SOLVER_PRECISION_UPPER) { // find dual that is equal to 0 with certain tolerance
                    //std::cout << "Debug\n";
                    //std::cout << new_dual.s[index] << std::endl;
                    face_scenario.axis.push_back(index);
                }
            }
            if (if_face_new(explored_faces, face_scenario)) { // if unique new face is found
                explored_faces.push_back(face_scenario);
            }
        }
    }
    if (explored_faces.size() == 0) {
        std::cout << "Warning: No face is explored in the presolve process.\n";
        //return dynamic_sdknn_solver_fast(folder_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, flag_debug);
    }
    // initialize a collection of minorants
    std::vector<minorant> minorant_collection;
    // construct initial minorant
    std::cout << "Construct minorant based on the presolved incumbent.\n";
    writeFile << "Construct minorant based on the presolved incumbent.\n";
    minorant initial_minorant;
    for (int idx_x = 0; idx_x < x_size; ++idx_x) {
        initial_minorant.beta.push_back(0);
    }
    minorant_collection.push_back(initial_minorant);
    // initialize the index set of active minorants
    std::vector<int> active_minorants;
    active_minorants.push_back(0);
    std::cout << "===(END) PRESOLVE PROCESS===\n";
    writeFile << "===(END) PRESOLVE PROCESS===\n";
    std::cout << "Maximum number of iterations: " << max_iterations << std::endl;
    writeFile << "Maximum number of iterations: " << max_iterations << std::endl;
    // main loop
    std::cout << "Start Solving Process\n";
    writeFile << "Start Solving Process\n";
    //N = N_pre; // update number of data points collected
    //int idx_datapoint = N - 1; // initialize the index for the datapoint
    int idx_datapoint = N_pre - 1;
    N = N_pre;
    k = k_new;
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        std::cout << "***Iteration " << iteration << "***\n";
        writeFile << "***Iteration " << iteration << "***\n";
        std::cout << "sigma: " << sigma << std::endl;
        writeFile << "sigma: " << sigma << std::endl;
        std::vector<double> x_candidate;
        N += 1; // increase sample size
        idx_datapoint += 1; // go to the next data point
        k_new = (int) pow(N, beta); // calculate new k
        std::cout << "k (number of nearest neighbor): " << k_new << std::endl;
        writeFile << "k (number of nearest neighbor): " << k_new << std::endl;
        std::cout << "k old: " << k << std::endl;// debug
        writeFile << "k old: " << k << std::endl;
        //std::vector<int> kNNSet_new;
        // PROXIMAL MAPPING (CANDIDATE SELECTION)
        //std::cout << "===CANDIDATE SELECTION IN THE PROXIMAL MAPPING===\n";
        //writeFile << "===CANDIDATE SELECTION IN THE PROXIMAL MAPPING===\n";
        // solve master problem with a proximal term
        IloEnv env;
        IloModel mod(env);
        IloNumVarArray x_temp(env,A_colsize,-IloInfinity,IloInfinity,ILOFLOAT);
        IloNumVar eta(env,-IloInfinity,IloInfinity,ILOFLOAT);
        mod.add(x_temp);
        mod.add(eta);
        IloExpr expr_obj(env);
        // objective linear
        for (auto it = model_parameters.c.vec.begin(); it != model_parameters.c.vec.end(); ++it) {
            expr_obj += (it -> second) * x_temp[it -> first];
        }
        // objective quadratic
        for (std::map<std::pair<int,int>, double>::iterator it = model_parameters.Q.mat.begin(); it != model_parameters.Q.mat.end(); ++it) {
            expr_obj += x_temp[(it -> first).first] * x_temp[(it -> first).second] * (it -> second) * 0.5;
        }
        // objective proximal term
        for (int x_index = 0; x_index < A_colsize; ++x_index) {
            expr_obj += 0.5 * sigma * (x_temp[x_index] * x_temp[x_index] - 2.0 * x_temp[x_index] * x_incumbent[x_index] + x_incumbent[x_index] * x_incumbent[x_index]);
        }
        // piecewise linear
        expr_obj += eta;
        IloObjective obj = IloMinimize(env,expr_obj); // objective function
        mod.add(obj);
        // constraints A x <= b
        std::vector<IloExpr> exprs_regular;
        for (int index_row = 0; index_row < A_rowsize; ++index_row) {
            IloExpr expr(env);
            exprs_regular.push_back(expr);
        }
        // regular
        for (std::map<std::pair<int,int>, double>::iterator it = model_parameters.A.mat.begin() ; it != model_parameters.A.mat.end(); ++it) {
            // get the location of the entry
            exprs_regular[(it -> first).first] += (it -> second) * x_temp[(it -> first).second];
        }
        for (auto it = model_parameters.b.vec.begin(); it != model_parameters.b.vec.end(); ++it) {
            exprs_regular[it -> first] -= (it -> second);
        }
        for (int index_row = 0; index_row < A_rowsize; ++index_row) {
            mod.add(exprs_regular[index_row] <= 0);
        }
        // constrants for minorants
        IloRangeArray minorant_constraints(env);
        std::cout << "Number of minorants used in promxial mapping: " << active_minorants.size() << std::endl;
        writeFile << "Number of minorants used in promxial mapping: " << active_minorants.size() << std::endl;
        for (int index_cons = 0; index_cons < active_minorants.size(); ++index_cons) {
            IloExpr expr(env);
            expr += minorant_collection[active_minorants[index_cons]].alpha - eta;
            for (int index_x = 0; index_x < x_size; ++index_x ) {
                expr += minorant_collection[active_minorants[index_cons]].beta[index_x] * x_temp[index_x];
            }
            minorant_constraints.add(expr <= 0);
        }
        mod.add(minorant_constraints);
        // constraints for the feasibility cuts
        for (int index_feas = 0; index_feas < feasibility_cuts.size(); ++index_feas) {
            IloExpr expr(env);
            for (int index_x = 0; index_x < x_size; ++index_x) {
                expr += feasibility_cuts[index_feas].A_newRow[index_x] * x_temp[index_x];
            }
            expr -= feasibility_cuts[index_feas].b_newRow;
            mod.add(expr <= 0);
        }
        // create cplex environment
        IloCplex cplex(env);
        cplex.extract(mod);
        cplex.setOut(env.getNullStream());
        cplex.solve();
        // obtain the proximal point (condidate solution)
        for (int x_index = 0; x_index < A_colsize; ++x_index) {
            x_candidate.push_back(cplex.getValue(x_temp[x_index]));
            //std::cout << cplex.getValue(x_temp[x_index]) << std::endl;
        }
        // update the set of active minorants
        IloNumArray duals(env);
        cplex.getDuals(duals, minorant_constraints);
        std::vector<int> active_minorants_new;
        int num_active_minorants = 0;
        for (int index = 0; index < active_minorants.size(); ++index) {
            if (duals[index] < QQ_SOLVER_PRECISION_LOWER || duals[index] > QQ_SOLVER_PRECISION_UPPER) { // only store the active minorants whose duals are significantly different from 0
                //std::cout << "dual: " << duals[index] << std::endl;
                active_minorants_new.push_back(active_minorants[index]);
                num_active_minorants += 1;
            }
        }
        std::cout << "Number of active minorants (dual not equal to 0): " << num_active_minorants << std::endl;
        writeFile << "Number of active minorants (dual not equal to 0): " << num_active_minorants << std::endl;
        // end the environment
        env.end();
        // output candidate solution
        std::cout << "Candidate Solution: ";
        writeFile << "Candidate Solution: ";
        for (int x_index = 0; x_index < x_size - 1; ++x_index) {
            std::cout << x_candidate[x_index] << ", ";
            writeFile << x_candidate[x_index] << ", ";
        }
        std::cout << x_candidate[x_size - 1] << std::endl;
        writeFile << x_candidate[x_size - 1] << std::endl;
        //std::cout << "===(END) CANDIDATE SELECTION IN THE PROXIMAL MAPPING===\n";
        //writeFile << "===(END) CANDIDATE SELECTION IN THE PROXIMAL MAPPING===\n";
        // proximal mapping in the first stage
        // obtain a new data point
        dataPoint be_datapoint;
        if (flag_be == true) {
            be_datapoint = be_DB[0][idx_datapoint];
        }
        else {
            std::cout << "No random variable is in be of the equality constraint of the second stage problem.\n";
        }
        dataPoint bi_datapoint;
        if (flag_bi == true) {
            bi_datapoint = bi_DB[0][idx_datapoint];
        }
        else {
            std::cout << "No random variable is in bi of the inequality constraint of the second stage problem.\n";
        }
        dataPoint Ce_datapoint;
        if (flag_Ce == true) {
            Ce_datapoint = Ce_DB[0][idx_datapoint];
        }
        else {
            std::cout << "No random variable is in Ce of the equality constraint of the second stage problem.\n";
        }
        dataPoint Ci_datapoint;
        if (flag_Ci == true) {
            Ci_datapoint = Ci_DB[0][idx_datapoint];
        }
        else {
            std::cout << "No random variable is in Ci of the inequality constraint of the second stage problem.\n";
        }
        //*********************
        // kNN ESTIMATION
        //non-parametric estimation (kNN)
        // calculate distance squared
        //std::cout << "===kNN ESTIMATION===\n";
        //writeFile << "===kNN ESTIMATION===\n";
        secondStageRHSpoint RHS_datapoint = merge_randomVector(be_datapoint, bi_datapoint, Ce_datapoint, Ci_datapoint);
        RHS_dataset.push_back(RHS_datapoint);
        if (N > N_pre) { // only update the kNN set when the number of data points exceed N_pre
            double distance_squared = 0;
            for (int idx_component = 0; idx_component < RHS_datapoint.predictor.size(); ++idx_component) {
                distance_squared += (RHS_datapoint.predictor[idx_component] - observed_predictor[idx_component]) * (RHS_datapoint.predictor[idx_component] - observed_predictor[idx_component]);
            }
            distanceSet.push_back(distance_squared);
            // store the new squared distance
            // sorting (like insert sorting)
            int left_index = 0; // the index corresponds to the largest distance that is smaller than the current one
            double left_distance = -1;
            // double indices used for tie-breaking
            int right_index = -1; // the index corresponds to the smallest distance that is larger than  the current one
            double right_distance = -1;
            for (int index = 0; index < orderSet.size(); ++index) {
                if (distanceSet[index] < distance_squared) {
                    if (left_index == 0) {
                        left_distance = distanceSet[index];
                        left_index = orderSet[index];
                    }
                    else if (distanceSet[index] > left_distance) {
                        left_distance = distanceSet[index];
                        left_index = orderSet[index];
                    }
                }
                if (distanceSet[index] > distance_squared) {
                    if (right_index == -1) {
                        right_distance = distanceSet[index];
                        right_index = orderSet[index];
                    }
                    else if (distanceSet[index] < right_distance) {
                        right_distance = distanceSet[index];
                        right_index = orderSet[index];
                    }
                }
            }
            /*
            if (flag_debug == true) {
                std::cout << "Output double indices\n";
                writeFile << "Output double indices\n";
                std::cout << "left index: " << left_index << std::endl;
                writeFile << "left index: " << left_index << std::endl;
                std::cout << "right index: " << right_index << std::endl;
                writeFile << "right index: " << right_index << std::endl;
            }
             */
            // update the orderSet
            for (int index = 0; index < orderSet.size(); ++index) {
                if (right_index != -1 && orderSet[index] >= right_index) {
                    orderSet[index] = orderSet[index] + 1;
                }
                //if (left_index == 0) { // current one is the nearest neighbor
                //    orderSet[index] = orderSet[index] + 1;
                //}
                //else if (orderSet[index] > left_index) {
                //    orderSet[index] = orderSet[index] + 1;
                //}
            }
            if (right_index == -1) {
                orderSet.push_back(orderSet.size() + 1);
            }
            else {
                orderSet.push_back(right_index);
            }
            /*
            if (flag_debug == true) {
                std::cout << "Updated Order in the scenario set\n";
                writeFile << "Updated Order in the scenario set\n";
                std::cout << "Index, Order, Distance (Squared)\n";
                writeFile << "Index, Order, Distance (Squared)\n";
                // update the kNN set
                for (int index = 0; index < orderSet.size(); ++index) {
                    std::cout << index << ", "<< orderSet[index] << ", " << distanceSet[index];
                    writeFile << index << ", "<< orderSet[index] << ", " << distanceSet[index];
                    if (orderSet[index] <= k_new) {
                        std::cout << "*";
                        writeFile << "*";
                        //kNNSet_new.push_back(index);
                    }
                    std::cout << std::endl;
                    writeFile << std::endl;
                }
            }
            else {
                // update the kNN set
                for (int index = 0; index < orderSet.size(); ++index) {
                    if (orderSet[index] <= k_new) {
                        //kNNSet_new.push_back(index);
                    }
                }
            }
             */
            // update the kNN set (need to optimize)
            kNNSet.clear(); // clear the old kNN set
            for (int index = 0; index < orderSet.size(); ++index) {
                if (orderSet[index] <= k_new) {
                    kNNSet.push_back(index);
                }
            }
        }
        //*********************
        //end non-parametric estimation (kNN)
        //std::cout << "===(END) kNN ESTIMATION===\n";
        //writeFile << "===(END) kNN ESTIMATION===\n";
        // calculate the dual multipliers
        dualMultipliers_QP dualsTemp = twoStageQP_secondStagePrimal(x_candidate, model_parameters, RHS_datapoint, RHSmap);
        if (dualsTemp.feasible_flag == false) { // if second stage subproblem is infeasible
            //
            std::cout << "Warning: An infeasible case is captured in iteration " << iteration << ".\n";
            writeFile << "Warning: An infeasible case is captured in iteration " << iteration << ".\n";
            std::cout << "Warning: A feasibility cut will be constructed.\n";
            writeFile << "Warning: A feasibility cut will be constructed.\n";
            dualMultipliers_QP extremeRay_scenario = twoStageQP_secondStageExtremRay(x_candidate, model_parameters, RHS_datapoint, RHSmap);
            feasibilityCut feasibilityCut_scenario = twoStageQP_feasibilityCutGeneration(extremeRay_scenario, model_parameters, RHS_datapoint, RHSmap);
            // add feasibility cut
            feasibility_cuts.push_back(feasibilityCut_scenario);
        }
        else {
            // second stage subproblem is feasible
            std::cout << "Computation Log: Subproblem in iteration " << iteration << " is feasible.\n";
            writeFile << "Computation Log: Subproblem in iteration " << iteration << " is feasible.\n";
            /*
            // check if the new dual is found
            bool flag_new_dual_explored = if_new_dual(explored_duals, dualsTemp);
            if (flag_new_dual_explored == true) {
                std::cout << "Computation Log: New dual is found.\n";
                writeFile << "Computation Log: New dual is found.\n";
                explored_duals.push_back(dualsTemp);
            }*/
            face face_scenario;
            for (int idx_s = 0; idx_s < dualsTemp.s.size(); ++idx_s) {
                if (dualsTemp.s[idx_s] >= QQ_SOLVER_PRECISION_LOWER && dualsTemp.s[idx_s] <= QQ_SOLVER_PRECISION_UPPER) { // find dual that is equal to 0 with certain tolerance
                    face_scenario.axis.push_back(idx_s);
                }
            }
            if (if_face_new(explored_faces, face_scenario)) { // if new face is found
                explored_faces.push_back(face_scenario);
            }
        }
        std::cout << "Number of faces: " << explored_faces.size() << std::endl;
        writeFile << "Number of faces: " << explored_faces.size() << std::endl;
        //std::cout << "===(END) DUAL SPACE EXPLORATION===\n";
        //writeFile << "===(END) DUAL SPACE EXPLORATION===\n";
        //  MINORANT CUTS CONSTRUCTION
        //std::cout << "===MINORANT CONSTRUCTION===\n";
        //writeFile << "===MINORANT CONSTRUCTION===\n";
        // find the duals correspond to the kNN
        //std::vector<dualMultipliers> dualSet_candidate;
        //std::vector<dualMultipliers> dualSet_incumbent;
        minorant minorant_candidate;
        minorant minorant_incumbent;
        minorant_candidate.alpha = 0;
        minorant_incumbent.alpha = 0;
        for (int index_x = 0; index_x < x_size; ++index_x) {
            minorant_candidate.beta.push_back(0.0);
            minorant_incumbent.beta.push_back(0.0);
        }
        /*
        if (flag_debug == true) {
            std::cout << "Index(kNN), Index(argmax dual for candidate), Index(argmax dual for incumbent)\n";
            writeFile << "Index(kNN), Index(argmax dual for candidate), Index(argmax dual for candidate)\n";
        }
         */
        for (int index = 0; index < k_new; ++index) {
            double max_value = -99999; // NOTE: need to make it smaller
            dualMultipliers_QP max_dual_candidate;
            dualMultipliers_QP max_dual_incumbent;
            // incumbent
            double max_value_incumbent = -99999; // NOTE: need to make it smaller
            for (int dual_index = 0; dual_index < explored_faces.size(); ++dual_index) {
                // find optimal dual based on the given face
                // candidate
                dualMultipliers_QP current_dual = twoStageQP_secondStageDual(x_candidate,model_parameters,RHS_dataset[kNNSet[index]],explored_faces[dual_index], RHSmap);
                double current_value = current_dual.obj_val;
                if (dual_index == 0) {
                    max_value = current_value;
                    max_dual_candidate.r = current_dual.r;
                    max_dual_candidate.s = current_dual.s;
                    max_dual_candidate.t = current_dual.t;
                }
                else if (max_value < current_value) {
                    max_value = current_value;
                    max_dual_candidate.r = current_dual.r;
                    max_dual_candidate.s = current_dual.s;
                    max_dual_candidate.t = current_dual.t;
                }
                // incumbent
                dualMultipliers_QP current_dual_incumbent = twoStageQP_secondStageDual(x_incumbent,model_parameters,RHS_dataset[kNNSet[index]],explored_faces[dual_index], RHSmap);
                double current_value_incumbent = current_dual_incumbent.obj_val;
                if (dual_index == 0) {
                    max_value_incumbent = current_value_incumbent;
                    max_dual_incumbent.r = current_dual_incumbent.r;
                    max_dual_incumbent.s = current_dual_incumbent.s;
                    max_dual_incumbent.t = current_dual_incumbent.t;
                }
                else if (max_value_incumbent < current_value_incumbent) {
                    max_value_incumbent = current_value_incumbent;
                    max_dual_incumbent.r = current_dual_incumbent.r;
                    max_dual_incumbent.s = current_dual_incumbent.s;
                    max_dual_incumbent.t = current_dual_incumbent.t;
            }
        } // end for
            /*
            if (flag_debug == true) {
                std::cout << kNNSet[index] << ", " << max_index << ",               " << max_index_incumbent << std::endl;
                writeFile << kNNSet[index] << ", " << max_index << ",               " << max_index_incumbent << std::endl;
            }
            */
            // calculate alpha and beta
            // candidate
            // deterministic part
            minorant_candidate.alpha += (-0.5 / (double) k_new) * (max_dual_candidate.r * model_parameters.P_inv * max_dual_candidate.r);
            std::vector<double> e(model_parameters.e.num_entry,0.0);
            // deterministic part
            for (auto it = model_parameters.e.vec.begin(); it != model_parameters.e.vec.end(); ++it) {
                e[it -> first] += (it -> second);
            }
            // stochastic part
            for (int idx_eq = 0; idx_eq < RHS_dataset[kNNSet[index]].be.size(); ++idx_eq) {
                e[RHSmap.be_map[idx_eq]] += RHS_dataset[kNNSet[index]].be[idx_eq];
            }
            // stochastic part
            minorant_candidate.alpha += (1.0 / (double) k_new) * (e * max_dual_candidate.t);
            // beta
            // deterministic part
            std::vector<double> beta_candidate = (-1.0 / (double) k_new) * (max_dual_candidate.t * model_parameters.C);
            // stochastic part
            // equality
            for (int idx_Ce = 0; idx_Ce < RHS_dataset[kNNSet[index]].Ce.size(); ++idx_Ce) {
                beta_candidate[RHSmap.Ce_map[idx_Ce].second] += (-1.0 / (double) k_new) * RHS_dataset[kNNSet[index]].Ce[idx_Ce] * max_dual_candidate.t[RHSmap.Ce_map[idx_Ce].first];
            }
            for (int idx_beta = 0; idx_beta < beta_candidate.size(); ++idx_beta) {
                minorant_candidate.beta[idx_beta] += beta_candidate[idx_beta];
            }
            // incumbent
            // deterministic part
            minorant_incumbent.alpha += (-0.5 / (double) k_new) * (max_dual_incumbent.r * model_parameters.P_inv * max_dual_incumbent.r);
            // stochastic part
            minorant_incumbent.alpha += (1.0 / (double) k_new) * (e * max_dual_incumbent.t);
            // beta
            // deterministic part
            std::vector<double> beta_incumbent = (-1.0 / (double) k_new) * (max_dual_incumbent.t * model_parameters.C);
            // stochastic part
            // equality
            for (int idx_Ce = 0; idx_Ce < RHS_dataset[kNNSet[index]].Ce.size(); ++idx_Ce) {
                beta_incumbent[RHSmap.Ce_map[idx_Ce].second] += (-1.0 / (double) k_new) * RHS_dataset[kNNSet[index]].Ce[idx_Ce] * max_dual_incumbent.t[RHSmap.Ce_map[idx_Ce].first];
            }
            for (int idx_beta = 0; idx_beta < beta_incumbent.size(); ++idx_beta) {
                minorant_incumbent.beta[idx_beta] += beta_incumbent[idx_beta];
            }
        }
        // end for (loop finding arg max of duals)
        // MINORANT UPDATES
        // update old minorants
        //std::cout << "Update old active minorants.\n";
        //writeFile << "Update old active minorants.\n";
        std::vector<minorant> minorant_collection_new;
        if (k == k_new) { // will have more advanced version to store the radius of kNN set
            for (int minorant_index = 0; minorant_index < active_minorants_new.size(); ++minorant_index) {
                minorant minorant_new;
                minorant_new.alpha = minorant_collection[active_minorants_new[minorant_index]].alpha +  (f_lowerbound - f_upperbound) / ((double) k);
                minorant_new.beta = minorant_collection[active_minorants_new[minorant_index]].beta;
                //minorant_collection[active_minorants_new[minorant_index]].alpha -= f_upperbound / ((double) k);
                minorant_collection_new.push_back(minorant_new);
            }
        }
        else {
            for (int minorant_index = 0; minorant_index < active_minorants_new.size(); ++minorant_index) {
                minorant minorant_new;
                minorant_new.alpha = minorant_collection[active_minorants_new[minorant_index]].alpha * ((double) k) / ((double) k_new) + ((double) (k_new - k)) / ((double) k_new) * f_lowerbound;
                //minorant_collection[active_minorants_new[minorant_index]].alpha = minorant_collection[active_minorants_new[minorant_index]].alpha * ((double) k) / ((double) k_new);
                minorant_new.beta = ((double) k) / ((double) k_new) * minorant_collection[active_minorants_new[minorant_index]].beta;
                minorant_collection_new.push_back(minorant_new);
            }
        } // end minorant update
        minorant_collection_new.push_back(minorant_candidate);
        minorant_collection_new.push_back(minorant_incumbent);
        // output new minorants
        if (flag_debug == true) {
            std::cout << "Minorant Candidate\n";
            writeFile << "Minorant Candidate\n";
            std::cout << "alpha: " << minorant_candidate.alpha << std::endl;
            writeFile << "alpha: " << minorant_candidate.alpha << std::endl;
            std::cout << "beta: ";
            writeFile << "beta: ";
            for (int x_index = 0; x_index < x_size - 1; ++x_index) {
                std::cout << minorant_candidate.beta[x_index] << ", ";
                writeFile << minorant_candidate.beta[x_index] << ", ";
            }
            std::cout << minorant_candidate.beta[x_size - 1] << std::endl;
            writeFile << minorant_candidate.beta[x_size - 1] << std::endl;
            std::cout << "Minorant Incumbent\n";
            writeFile << "Minorant Incumbent\n";
            std::cout << "alpha: " << minorant_incumbent.alpha << std::endl;
            writeFile << "alpha: " << minorant_incumbent.alpha << std::endl;
            std::cout << "beta: ";
            writeFile << "beta: ";
            for (int x_index = 0; x_index < x_size - 1; ++x_index) {
                std::cout << minorant_incumbent.beta[x_index] << ", ";
                writeFile << minorant_incumbent.beta[x_index] << ", ";
            }
            std::cout << minorant_incumbent.beta[x_size - 1] << std::endl;
            writeFile << minorant_incumbent.beta[x_size - 1] << std::endl;
        }
        //std::cout << "===(END) MINORANT CONSTRUCTION===\n";
        //writeFile << "===(END) MINORANT CONSTRUCTION===\n";
        // Incumbent Selection
        //std::cout << "===INCUMBENT SELECTION===\n";
        //writeFile << "===INCUMBENT SELECTION===\n";
        bool flag_incumbent_selection = incumbent_selection_check_QP(q, x_candidate, x_incumbent, model_parameters, minorant_collection, minorant_collection_new, active_minorants);
        if (flag_incumbent_selection == true) {
            std::cout << "Computation Log: Incumbent selection criterion is passed.\n";
            writeFile <<"Computation Log: Incumbent selection criterion is passed.\n";
            x_incumbent = x_candidate;
            // update stepsize
            sigma = max(sigma * 0.5, sigma_lowerbound);
        }
        else {
            std::cout << "Computation Log: Incumbent selection criterion is not passed.\n";
            writeFile <<"Computation Log: Incumbent solution selection criterion is not passed.\n";
            sigma = min(sigma * 2.0, sigma_upperbound);
        }
        // print out the incumbent solution
        std::cout << "Incumbent Solution: ";
        writeFile << "Incumbent Solution: ";
        for (int index = 0; index < x_size-1; ++index) {
            std::cout << x_incumbent[index] << ", ";
            writeFile << x_incumbent[index] << ", ";
        }
        std::cout << x_incumbent[x_size - 1] << std::endl;
        writeFile << x_incumbent[x_size - 1] << std::endl;
        // updates on the objective function (need to check)
        if (k == k_new) { // will have more advanced version to store the radius of kNN set
            for (int minorant_index = 0; minorant_index < active_minorants_new.size(); ++minorant_index) {
                minorant_collection[active_minorants_new[minorant_index]].alpha += (f_lowerbound - f_upperbound) / ((double) k);
            }
        }
        else {
            for (int minorant_index = 0; minorant_index < active_minorants_new.size(); ++minorant_index) {
                /*
                std::cout << "Old alpha:" << minorant_collection[active_minorants_new[minorant_index]].alpha << std::endl; // debug
                std::cout << "Old beta:\n";
                for (int index = 0; index < x_size; ++index) {
                    std::cout << minorant_collection[active_minorants_new[minorant_index]].beta[index] << " ";
                }
                std::cout << std::endl;
                 */
                minorant_collection[active_minorants_new[minorant_index]].alpha = minorant_collection[active_minorants_new[minorant_index]].alpha * ((double) k) / ((double) k_new) + ((double) (k_new - k)) / ((double) k_new) * f_lowerbound;
                minorant_collection[active_minorants_new[minorant_index]].beta = ((double) k) / ((double) k_new) * minorant_collection[active_minorants_new[minorant_index]].beta;
                /*
                std::cout << "New alpha: "<<  minorant_collection[active_minorants_new[minorant_index]].alpha << std::endl; // debug
                for (int index = 0; index < x_size; ++index) {
                    std::cout << minorant_collection[active_minorants_new[minorant_index]].beta[index] << " ";
                }
                std::cout << std::endl;
                 */
            }
        } // end minorant update
        //std::cout << "Add new minorants to the the collection of minorants.\n";
        //writeFile << "Add new minorants to the the collection of minorants.\n";
        // add two newly generated minorants to the minorant collection
        minorant_collection.push_back(minorant_candidate);
        int index_minorant_candidate = 2 * iteration + 1;
        minorant_collection.push_back(minorant_incumbent);
        int index_minorant_incumbent = 2 * (iteration + 1);
        active_minorants_new.push_back(index_minorant_candidate);
        active_minorants_new.push_back(index_minorant_incumbent);
        //std::cout << "Minorant Index (candidate): " << index_minorant_candidate << std::endl;
        //std::cout << "Minorant Index (incumbent): " << index_minorant_incumbent << std::endl;
        // final step of one iteration
        active_minorants = active_minorants_new; // update the indices of active minorants
        // update k
        k = k_new;
        //std::cout << "===(END) INCUMBENT SELECTION===\n";
        //writeFile << "===(END) INCUMBENT SELECTION===\n";
        
    } // end main loop
    std::cout << "*******************************************\n";
    writeFile << "*******************************************\n";
    std::cout << "Output Solution: ";
    writeFile << "Output Solution: ";
    for (int index = 0; index < x_size-1; ++index) {
        std::cout << x_incumbent[index] << ", ";
        writeFile << x_incumbent[index] << ", ";
    }
    std::cout << x_incumbent[x_size - 1] << std::endl;
    writeFile << x_incumbent[x_size - 1] << std::endl;
    std::cout << "Computation Log: Finish Solving Process.\n";
    writeFile << "Computation Log: Finish Solving Process.\n";
    // write time elapsed
    double duration = (std::clock() - time_start ) / (double) CLOCKS_PER_SEC;
    writeFile << "Time elapsed(secs) : " << duration << "\n";
    std::cout << "Output explored faces:\n";
    writeFile << "Output explored faces:\n";
    std::cout << "Only output the axis that equals to 0\n";
    writeFile << "Only output the axis that equals to 0\n";
    for (int idx_face = 0; idx_face < explored_faces.size(); ++idx_face) {
        std::cout << "face #" << idx_face << ": " << std::endl;
        writeFile << "face #" << idx_face << ": " << std::endl;
        for (int idx_axis = 0; idx_axis < explored_faces[idx_face].axis.size(); ++idx_axis) {
            std::cout << explored_faces[idx_face].axis[idx_axis] << "  ";
            writeFile << explored_faces[idx_face].axis[idx_axis] << "  ";
        }
        std::cout << std::endl;
        writeFile << std::endl;
    }
    writeFile << "*******************************************\n";
    writeFile.close();
    return x_incumbent;
}
