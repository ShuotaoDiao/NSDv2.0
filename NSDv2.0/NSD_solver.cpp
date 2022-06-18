//
//  NSD_solver.cpp
//  NSDv2.0
//
//  Created by Shuotao Diao on 8/4/21.
//

#include "NSD_solver.hpp"

// declare global variables (need to be defined in the source file)
double SOLVER_PRECISION_LOWER = -1e-6;
double SOLVER_PRECISION_UPPER = 1e-6;

// compare duals
bool if_duals_equal(const dualMultipliers& dual1, const dualMultipliers& dual2){
    if (if_equal(dual1.equality, dual2.equality)) {
        if (if_equal(dual1.inequality, dual2.inequality)) {
            return true;
        }
        else {
            return false;
        }
    }
    else {
        return false;
    }
    return true;
}

bool if_new_dual(const std::vector<dualMultipliers>& duals, const std::vector<int>& indices, const dualMultipliers& candidate_dual) {
    if (indices.size() < 1) {
        return true;
    }
    for (int index = 0; index < indices.size(); ++index) {
        if (if_duals_equal(duals[indices[index]], candidate_dual)) { // if candidate_dual is equal to some dual in the duals
            //std::cout << "Two duals are equal.\n";
            //print(duals[index].inequality);
            //print(candidate_dual.inequality);
            return false;
        }
    }
    return true;
}

bool if_new_dual(const std::vector<dualMultipliers>& duals, const dualMultipliers& candidate_dual) {
    if (duals.size() < 1) {
        return true; // new dual is found
    }
    for (int index = 0; index < duals.size(); ++index) {
        if (if_duals_equal(duals[index], candidate_dual)) {
            std::cout << "Dual is not new.\n";
            return false; // candidate dual already exists
        }
    }
    return true; // new dual is found
}


// obtain dual multiplers of the second stage by solving the primal, given x (first stage decision variable)
dualMultipliers twoStageLP_secondStageDual(const std::vector<double>& x, standardTwoStageParameters& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap) {
    dualMultipliers pi; // initialize dual multipliers
    int y_size = model_parameters.D.num_col;
    // set up the model
    IloEnv env;
    IloModel mod(env);
    IloNumVarArray y(env,y_size,0,IloInfinity,ILOFLOAT); // second stage decision variables
    mod.add(y);
    // objective function
    IloExpr expr_obj(env);
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
    // coefficients before x (stochastic part) inequality (location is behind equality constraints)
    for (int idx_Ci = 0; idx_Ci < rhs.Ci.size(); ++idx_Ci) {
        exprs_eq[RHSmap.Ci_map[idx_Ci].first + model_parameters.num_eq] += rhs.Ci[idx_Ci] * x[RHSmap.Ci_map[idx_Ci].second];
    }
    // right hand side (stochastic part) equality be_(i) equality
    for (int idx_be = 0; idx_be < rhs.be.size(); ++idx_be) {
        exprs_eq[RHSmap.be_map[idx_be]] -= rhs.be[idx_be];
    }
    // right hand side (stochastic part) equality bi_(i) inequality
    for (int idx_bi = 0; idx_bi < rhs.bi.size(); ++idx_bi) {
        exprs_eq[RHSmap.bi_map[idx_bi] + model_parameters.num_eq] -= rhs.bi[idx_bi];
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
    IloBool solvable_flag = cplex.solve();
    IloNumArray dual_equality(env);
    IloNumArray dual_inequality(env);
    if (solvable_flag == IloTrue) {
        pi.feasible_flag = true; // tell the subproblem is feasible for a given x, first stage decision variable
        cplex.getDuals(dual_equality,constraintsEquality);
        for (int index_eq = 0; index_eq < model_parameters.D.num_row; ++index_eq) {
            double pi_temp = dual_equality[index_eq]; // move y to the right hand side
            pi.equality.push_back(pi_temp);
        }
    }
    else {
        pi.feasible_flag = false; // tell the subproblem is infeasible for given x
    }
    env.end();
    return pi;
}

// functions for generating feasibility cut
dualMultipliers twoStageLP_secondStageExtremRay(const std::vector<double>& x, standardTwoStageParameters& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap) {
    dualMultipliers extremeRay;
    extremeRay.feasible_flag = false;
    // obtain the sizes of input parameters
    int y_size = model_parameters.D.num_col;
    // set up the model
    IloEnv env;
    IloModel mod(env);
    IloNumVarArray y(env,y_size,0,IloInfinity,ILOFLOAT); // second stage decision variables
    mod.add(y);
    // objective function
    IloExpr expr_obj(env);
    for (auto it = model_parameters.d.vec.begin(); it != model_parameters.d.vec.end(); ++it) {
        expr_obj += (it -> second) * y[it -> first];
    }
    IloObjective obj = IloMinimize(env,expr_obj);
    mod.add(obj);
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
    // coefficients before x (stochastic part) inequality (location is behind equality constraints)
    for (int idx_Ci = 0; idx_Ci < rhs.Ci.size(); ++idx_Ci) {
        exprs_eq[RHSmap.Ci_map[idx_Ci].first + model_parameters.num_eq] += rhs.Ci[idx_Ci] * x[RHSmap.Ci_map[idx_Ci].second];
    }
    // right hand sie (stochastic part) equality be_(i) equality
    for (int idx_be = 0; idx_be < rhs.be.size(); ++idx_be) {
        exprs_eq[RHSmap.be_map[idx_be]] -= rhs.be[idx_be];
    }
    // right hand sie (stochastic part) equality bi_(i) inequality
    for (int idx_bi = 0; idx_bi < rhs.bi.size(); ++idx_bi) {
        exprs_eq[RHSmap.bi_map[idx_bi] + model_parameters.num_eq] -= rhs.bi[idx_bi];
    }
    // add the equality constraints
    for (int index_eq = 0; index_eq < model_parameters.D.num_row; ++index_eq) {
        constraintsEquality.add(exprs_eq[index_eq] == 0);
    }
    mod.add(constraintsEquality);
    // set up cplex solver
    IloCplex cplex(env);
    cplex.extract(mod);
    cplex.setParam(IloCplex::PreInd, false); // need to turn off presolve in order to get dual extreme rays
    cplex.setParam(IloCplex::RootAlg, IloCplex::Dual); // use dual simplex optimizer
    cplex.solve(); // solve the problem
    IloNumArray extremeRay_eq(env);
    cplex.dualFarkas(constraintsEquality, extremeRay_eq);
    for (int index_eq = 0; index_eq < model_parameters.D.num_row; ++index_eq) {
        double pi_temp = extremeRay_eq[index_eq]; // move y to the right hand side
        extremeRay.equality.push_back(pi_temp);
    }
    return extremeRay;
}

// construct feasibility cut
feasibilityCut twoStageLP_feasibilityCutGeneration(const dualMultipliers& extremeRay, standardTwoStageParameters& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap) {
    feasibilityCut cut_scenario;
    // intercept deterministic part  A_newRow x \leq b_newRow, so the negative is taken
    cut_scenario.b_newRow = (-1.0) * (model_parameters.e * extremeRay.equality);
    // stochastic part
    // equality part
    for (int idx_eq = 0; idx_eq < rhs.be.size(); ++idx_eq) {
        cut_scenario.b_newRow -= extremeRay.equality[RHSmap.be_map[idx_eq]] * rhs.be[idx_eq];
    }
    // inequality part (before standardizing) inequality constraint is after the equality constraints
    for (int idx_ineq = 0; idx_ineq < rhs.bi.size(); ++idx_ineq) {
        cut_scenario.b_newRow -= extremeRay.equality[RHSmap.bi_map[idx_ineq] + model_parameters.num_eq] * rhs.bi[idx_ineq];
    }
    // slope
    // deterministic part
    cut_scenario.A_newRow = (-1.0) * (extremeRay.equality * model_parameters.C);
    // stochastic part
    // equality
    for (int idx_Ce = 0; idx_Ce < rhs.Ce.size(); ++idx_Ce) {
        cut_scenario.A_newRow[RHSmap.Ce_map[idx_Ce].second] += -1.0 * rhs.Ce[idx_Ce] * extremeRay.equality[RHSmap.Ce_map[idx_Ce].first];
    }
    // inequality before standardizing
    for (int idx_Ci = 0; idx_Ci < rhs.Ci.size(); ++idx_Ci) {
        cut_scenario.A_newRow[RHSmap.Ce_map[idx_Ci].second] += -1.0 * rhs.Ci[idx_Ci] * extremeRay.equality[RHSmap.Ci_map[idx_Ci].first + model_parameters.num_eq];
    }
    return cut_scenario;
}

// projection for the first stage in the two stage linear programming
std::vector<double> twoStageLP_projection(const std::vector<double>& x, standardTwoStageParameters& model_parameters) {
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
std::vector<double> twoStageLP_presolve(standardTwoStageParameters& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap) {
    std::vector<double> x_candidate(model_parameters.c.num_entry,0.0);
    // solve a two stage LP
    IloEnv env;
    IloModel mod(env);
    IloNumVarArray x_temp(env,model_parameters.c.num_entry,-IloInfinity,IloInfinity,ILOFLOAT);
    IloNumVarArray y(env,model_parameters.D.num_col,0,IloInfinity,ILOFLOAT); // y >= 0
    mod.add(x_temp);
    mod.add(y);
    IloExpr expr_obj(env);
    // first stage objective
    for (auto it = model_parameters.c.vec.begin(); it != model_parameters.c.vec.end(); ++it) {
        expr_obj += (it -> second) * x_temp[it -> first];
    }
    // second stage objective
    for (auto it = model_parameters.d.vec.begin(); it != model_parameters.d.vec.end(); ++it) {
        expr_obj += (it -> second) * y[it -> first];
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
    // coefficients before x (stochastic part) inequality (location is behind equality constraints)
    for (int idx_Ci = 0; idx_Ci < rhs.Ci.size(); ++idx_Ci) {
        exprs_eq[RHSmap.Ci_map[idx_Ci].first + model_parameters.num_eq] += rhs.Ci[idx_Ci] * x_temp[RHSmap.Ci_map[idx_Ci].second];
    }
    // right hand side (stochastic part) equality be_(i) equality
    for (int idx_be = 0; idx_be < rhs.be.size(); ++idx_be) {
        exprs_eq[RHSmap.be_map[idx_be]] -= rhs.be[idx_be];
    }
    // right hand side (stochastic part) equality bi_(i) inequality
    for (int idx_bi = 0; idx_bi < rhs.bi.size(); ++idx_bi) {
        exprs_eq[RHSmap.bi_map[idx_bi] + model_parameters.num_eq] -= rhs.bi[idx_bi];
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
    //cplex.exportModel("/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99small/experiment1/case1/presolve_model.lp");
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
bool incumbent_selection_check(double q, const std::vector<double>& x_candidate, const std::vector<double>& x_incumbent, sparseVector& c, const std::vector<minorant>& minorants, const std::vector<minorant>& minorants_new, const std::vector<int>& active_minorants) {
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
    double f_candidate = c * x_candidate + max_value_candidate;
    double f_incumbent = c * x_incumbent + max_value_incumbent;
    double f_new_candidate = c * x_candidate + max_value_new_candidate;
    double f_new_incumbent = c * x_incumbent + max_value_new_incumbent;
    double delta = f_candidate - f_incumbent;
    double delta_new = f_new_candidate - f_new_incumbent;
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


bool incumbent_selection_check_v2(double q, const std::vector<double>& x_candidate, const std::vector<double>& x_incumbent, sparseVector& c, const std::vector<minorant>& minorants, const std::vector<minorant>& minorants_new) {
    double max_value_candidate = -99999;
    double max_value_new_candidate = -99999;
    double max_value_incumbent = -99999;
    double max_value_new_incumbent = -99999;
    for (int index = 0; index < minorants.size(); ++index) {
        double temp_candidate = minorants[index].alpha;
        temp_candidate += minorants[index].beta * x_candidate;
        if (index == 0) {
            max_value_candidate = temp_candidate;
        }
        else if (temp_candidate > max_value_candidate) {
            max_value_candidate = temp_candidate;
        }
        double temp_incumbent = minorants[index].alpha;
        temp_incumbent += minorants[index].beta * x_incumbent;
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
    double f_candidate = c * x_candidate + max_value_candidate;
    double f_incumbent = c * x_incumbent + max_value_incumbent;
    double f_new_candidate = c * x_candidate + max_value_new_candidate;
    double f_new_incumbent = c * x_incumbent + max_value_new_incumbent;
    double delta = f_candidate - f_incumbent;
    double delta_new = f_new_candidate - f_new_incumbent;
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



// NSD solver with presolve and selected track of duals (initial k will be based on the presolve)
std::vector<double> dynamic_sdknn_solver_presolve(const std::string& folder_path, int max_iterations, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug) {
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
    bool flag_be; // tell if be stochastic is generated
    bool flag_bi; // tell if bi stochastic is generated
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
    std::string resultsOutput_path = folder_path + "/computationalResults(NSDv2.0presolve).txt";
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
    standardTwoStageParameters model_parameters;
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
    // read model file
    model_parameters = readStandardTwoStageParameters(model_path);
    // read sto file
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
    std::cout << "SD-kNN (fast version with presolve v2.0) is initialized\n";
    writeFile << "SD-kNN (fast version with presolve v2.0) is initialized\n";
    std::cout << "Algorithmic Parameters\n";
    writeFile << "Algorithmic Parameters\n";
    std::cout << "sigma, q, beta, k, N, N_pre, sigma_lower, sigma_upper" << std::endl;
    writeFile << "sigma, q, beta, k, N, N_pre, sigma_lower, sigma_upper" << std::endl;
    std::cout << sigma << ", " << q << ", " << beta << ", " << k << ", " << N << ", " << N_pre << ", " << sigma_lowerbound << ", " << sigma_upperbound << std::endl;
    writeFile << sigma << ", " << q << ", " << beta << ", " << k << ", " << N << ", " << N_pre << ", " << sigma_lowerbound << ", " << sigma_upperbound << std::endl;
    std::cout << "Problem Complexity\n";
    writeFile << "Problem Complexity\n";
    std::cout << "A_num_row, A_num_col\n";
    writeFile << "A_num_row, A_num_col\n";
    std::cout << model_parameters.A.num_row << ", " << model_parameters.A.num_col << std::endl;
    writeFile << model_parameters.A.num_row << ", " << model_parameters.A.num_col << std::endl;
    std::cout << "D_num_row, D_num_col (after converting into standard form)\n";
    writeFile << "D_num_row, D_num_col (after converting into standard form)\n";
    std::cout << model_parameters.D.num_row << ", " << model_parameters.D.num_col << std::endl;
    writeFile << model_parameters.D.num_row << ", " << model_parameters.D.num_col << std::endl;
    // set up initial incumbent solution
    long x_size = model_parameters.c.num_entry;
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
                        kNNSet_new.push_back(index);
                    }
                    std::cout << std::endl;
                    writeFile << std::endl;
                }
            }
            else {
                // update the kNN set
                for (int index = 0; index < orderSet.size(); ++index) {
                    if (orderSet[index] <= k_new) {
                        kNNSet_new.push_back(index);
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
    // presolve problem to get x_incumbent
    x_incumbent = twoStageLP_presolve(model_parameters, knn_point_estimate, RHSmap);
    std::cout << "Incumbent solution after presolve:\n";
    writeFile << "Incumbent solution after presolve:\n";
    for (int idx_x = 0; idx_x < x_incumbent.size() - 1; ++idx_x) {
        std::cout << x_incumbent[idx_x] << ", ";
        writeFile << x_incumbent[idx_x] << ", ";
    }
    std::cout << x_incumbent[x_incumbent.size() - 1] << std::endl;
    writeFile << x_incumbent[x_incumbent.size() - 1] << std::endl;
    // obtain duals at the presolve points
    // explored dual multipliers in the second stage
    std::vector<dualMultipliers> explored_duals;
    for (int idx = 0; idx < N_pre; ++idx) {
        dualMultipliers new_dual = twoStageLP_secondStageDual(x_incumbent, model_parameters, RHS_dataset[idx], RHSmap);
        if (new_dual.feasible_flag == false) { // if second stage subproblem is infeasible
            //
            std::cout << "Warning: An infeasible case is captured.\n";
            writeFile << "Warning: An infeasible case is captured.\n";
            std::cout << "Warning: A feasibility cut will be constructed.\n";
            writeFile << "Warning: A feasibility cut will be constructed.\n";
            dualMultipliers extremeRay_scenario = twoStageLP_secondStageExtremRay(x_candidate, model_parameters, RHS_dataset[idx], RHSmap);
            feasibilityCut feasibilityCut_scenario = twoStageLP_feasibilityCutGeneration(extremeRay_scenario, model_parameters, RHS_dataset[idx], RHSmap);
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
            explored_duals.push_back(new_dual);
        }
    }
    if (explored_duals.size() == 0) {
        std::cout << "Warning: No dual is explored in the presolve process.\n";
        std::cout << "Warning: The solver is terminated.\n";
        return x_incumbent;
        //std::cout << "The algorithm is transfered to the one with light presolve.\n";
        //return dynamic_sdknn_solver_fast(folder_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, flag_debug);
    }
    else if (explored_duals.size() < N_pre) {
        int dual_size = explored_duals.size();
        for (int idx = 0; idx < N_pre - explored_duals.size(); ++idx) {
            int idx_dual = rand() % dual_size + 1;
            // randomly pick a dual
            explored_duals.push_back(explored_duals[idx_dual]); // make the size of explored duals equal to N_pre
        }
    }
    // initialize a collection of minorants
    std::vector<minorant> minorant_collection;
    // construct initial minorant
    std::cout << "Construct initial minorant.\n";
    writeFile << "Construct initial minorant.\n";
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
    // initialize the index for the datapoint
    int idx_datapoint = N_pre - 1;
    N = N_pre; // update number of data points collected
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
        for (auto it = model_parameters.c.vec.begin(); it != model_parameters.c.vec.end(); ++it) {
            expr_obj += (it -> second) * x_temp[it -> first];
        }
        for (int x_index = 0; x_index < A_colsize; ++x_index) {
            expr_obj += 0.5 * sigma * (x_temp[x_index] * x_temp[x_index] - 2.0 * x_temp[x_index] * x_incumbent[x_index] + x_incumbent[x_index] * x_incumbent[x_index]);
        }
        expr_obj += eta;
        IloObjective obj = IloMinimize(env,expr_obj); // objective function
        mod.add(obj);
        // constraints
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
            if (duals[index] < SOLVER_PRECISION_LOWER || duals[index] > SOLVER_PRECISION_UPPER) { // only store the active minorants whose duals are significantly different from 0
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
                        kNNSet_new.push_back(index);
                    }
                    std::cout << std::endl;
                    writeFile << std::endl;
                }
            }
            else {
                // update the kNN set
                for (int index = 0; index < orderSet.size(); ++index) {
                    if (orderSet[index] <= k_new) {
                        kNNSet_new.push_back(index);
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
        // DUAL SPACE EXPLORATION
        //std::cout << "===DUAL SPACE EXPLORATION===\n";
        //writeFile << "===DUAL SPACE EXPLORATION===\n";
        
        // calculate the dual multipliers
        dualMultipliers dualsTemp = twoStageLP_secondStageDual(x_candidate, model_parameters, RHS_datapoint, RHSmap);
        if (dualsTemp.feasible_flag == false) { // if second stage subproblem is infeasible
            //
            std::cout << "Warning: An infeasible case is captured in iteration " << iteration << ".\n";
            writeFile << "Warning: An infeasible case is captured in iteration " << iteration << ".\n";
            std::cout << "Warning: A feasibility cut will be constructed.\n";
            writeFile << "Warning: A feasibility cut will be constructed.\n";
            dualMultipliers extremeRay_scenario = twoStageLP_secondStageExtremRay(x_candidate, model_parameters, RHS_datapoint, RHSmap);
            feasibilityCut feasibilityCut_scenario = twoStageLP_feasibilityCutGeneration(extremeRay_scenario, model_parameters, RHS_datapoint, RHSmap);
            // add feasibility cut
            feasibility_cuts.push_back(feasibilityCut_scenario);
            int idx_dual = rand() % (k_new) + 1;
            explored_duals.push_back(explored_duals[kNNSet[idx_dual]]);// still randomly add an existed dual
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
            explored_duals.push_back(dualsTemp);
        }
        std::cout << "Number of duals: " << explored_duals.size() << std::endl;
        writeFile << "Number of duals: " << explored_duals.size() << std::endl;
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
        // only use the duals that correspond to the kNN set
        // find unique duals in the knn dual set
        std::vector<int> kNNset_uniqueDuals;
        for (int index = 0; index < k_new; ++index) {
            if (if_new_dual(explored_duals, kNNset_uniqueDuals, explored_duals[kNNSet[index]])) {
                kNNset_uniqueDuals.push_back(kNNSet[index]);
            }
        }
        std::cout << "Number of unique duals (kNN set): " << kNNset_uniqueDuals.size() << std::endl;
        for (int index = 0; index < k_new; ++index) {
            double max_value = -99999; // NOTE: need to make it smaller
            int max_index = -1;
            int max_index_incumbent = -1;
            double alpha_candidate = 0;
            double alpha_incumbent = 0;
            std::vector<double> beta_candidate;
            std::vector<double> beta_incumbent;
            // incumbent
            double max_value_incumbent = -99999; // NOTE: need to make it smaller
            for (int dual_index = 0; dual_index < kNNset_uniqueDuals.size(); ++dual_index) {
                // find optimal dual based on the given set of unique duals
                double current_value = 0;
                // deterministic e
                double pi_e = model_parameters.e * explored_duals[kNNset_uniqueDuals[dual_index]].equality;
                // stochastic e
                // equality part
                for (int idx_eq = 0; idx_eq < RHS_dataset[kNNSet[index]].be.size(); ++idx_eq) {
                    pi_e += explored_duals[kNNset_uniqueDuals[dual_index]].equality[idx_eq] * RHS_dataset[kNNSet[index]].be[idx_eq];
                }
                // inequality part (before standardizing) inequality constraint is after the equality constraints
                for (int idx_ineq = 0; idx_ineq < RHS_dataset[kNNSet[index]].bi.size(); ++idx_ineq) {
                    pi_e += explored_duals[kNNset_uniqueDuals[dual_index]].equality[RHSmap.bi_map[idx_ineq] + model_parameters.num_eq] * RHS_dataset[kNNSet[index]].bi[idx_ineq];
                }
                current_value += pi_e;
                // determinitsic C
                std::vector<double> pi_C = explored_duals[kNNset_uniqueDuals[dual_index]].equality * model_parameters.C;
                // stochastic C
                // equality
                for (int idx_Ce = 0; idx_Ce < RHS_dataset[kNNSet[index]].Ce.size(); ++idx_Ce) {
                    pi_C[RHSmap.Ce_map[idx_Ce].second] += -1.0 * RHS_dataset[kNNSet[index]].Ce[idx_Ce] * explored_duals[kNNset_uniqueDuals[dual_index]].equality[RHSmap.Ce_map[idx_Ce].first];
                }
                // inequality before standardizing
                for (int idx_Ci = 0; idx_Ci < RHS_dataset[kNNSet[index]].Ci.size(); ++idx_Ci) {
                    pi_C[RHSmap.Ce_map[idx_Ci].second] += -1.0 * RHS_dataset[kNNSet[index]].Ci[idx_Ci] * explored_duals[kNNset_uniqueDuals[dual_index]].equality[RHSmap.Ci_map[idx_Ci].first + model_parameters.num_eq];
                }
                current_value += (-1.0) * (pi_C * x_candidate);
                double current_value_incumbent = 0;
                current_value_incumbent += pi_e;
                current_value_incumbent += (-1.0) * (pi_C * x_incumbent);
                if (dual_index < 1) {
                    max_index = dual_index;
                    max_value = current_value;
                    max_index_incumbent = dual_index;
                    max_value_incumbent = current_value_incumbent;
                    // store the intercept and slope
                    alpha_candidate = pi_e; // \pi^\top e
                    alpha_incumbent = pi_e;
                    beta_candidate = (-1.0) * pi_C; // -\pi^\top C
                    beta_incumbent = (-1.0) * pi_C;
                }
                else {
                    if (max_value < current_value) { // find the better dual for given candidate
                        max_index = dual_index;
                        max_value = current_value;
                        alpha_candidate = pi_e;
                        beta_candidate = (-1.0) * pi_C;
                    }
                    if (max_value_incumbent < current_value_incumbent) { // find the better dual for given incumbent
                        max_index_incumbent = dual_index;
                        max_value_incumbent = current_value_incumbent;
                        alpha_incumbent = pi_e;
                        beta_incumbent = (-1.0) * pi_C;
                    }
                }
            }
            // minorant on the candidate
            minorant_candidate.alpha += (1.0 / (double) k_new) * alpha_candidate;
            minorant_candidate.beta = minorant_candidate.beta + (1.0 / (double) k_new) * beta_candidate;
            // minorant on the incumbent
            minorant_incumbent.alpha += (1.0 / (double) k_new) * alpha_incumbent;
            minorant_incumbent.beta = minorant_incumbent.beta + (1.0 / (double) k_new) * beta_incumbent;
        }
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
        bool flag_incumbent_selection = incumbent_selection_check(q, x_candidate, x_incumbent, model_parameters.c, minorant_collection, minorant_collection_new, active_minorants);
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
                /*
                std::cout << "New alpha: "<<  minorant_collection[active_minorants_new[minorant_index]].alpha << std::endl; // debug
                minorant_collection[active_minorants_new[minorant_index]].beta = ((double) k) / ((double) k_new) * minorant_collection[active_minorants_new[minorant_index]].beta;
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
    }
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
    writeFile << "*******************************************\n";
    
    writeFile.close();
    return x_incumbent;
}


// NSD solver with presolve all the explored duals will be used
std::vector<double> dynamic_sdknn_solver_presolve_fullDual(const std::string& folder_path, int max_iterations, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug) {
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
    bool flag_be; // tell if be stochastic is generated
    bool flag_bi; // tell if bi stochastic is generated
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
    std::string resultsOutput_path = folder_path + "/computationalResults(NSDv2.0presolve).txt";
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
    standardTwoStageParameters model_parameters;
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
    // read model file
    model_parameters = readStandardTwoStageParameters(model_path);
    // read sto file
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
    std::cout << "SD-kNN (fast version with presolve v2.0) is initialized\n";
    writeFile << "SD-kNN (fast version with presolve v2.0) is initialized\n";
    std::cout << "Algorithmic Parameters\n";
    writeFile << "Algorithmic Parameters\n";
    std::cout << "sigma, q, beta, k, N, N_pre, sigma_lower, sigma_upper" << std::endl;
    writeFile << "sigma, q, beta, k, N, N_pre, sigma_lower, sigma_upper" << std::endl;
    std::cout << sigma << ", " << q << ", " << beta << ", " << k << ", " << N << ", " << N_pre << ", " << sigma_lowerbound << ", " << sigma_upperbound << std::endl;
    writeFile << sigma << ", " << q << ", " << beta << ", " << k << ", " << N << ", " << N_pre << ", " << sigma_lowerbound << ", " << sigma_upperbound << std::endl;
    std::cout << "Problem Complexity\n";
    writeFile << "Problem Complexity\n";
    std::cout << "A_num_row, A_num_col\n";
    writeFile << "A_num_row, A_num_col\n";
    std::cout << model_parameters.A.num_row << ", " << model_parameters.A.num_col << std::endl;
    writeFile << model_parameters.A.num_row << ", " << model_parameters.A.num_col << std::endl;
    std::cout << "D_num_row, D_num_col (after converting into standard form)\n";
    writeFile << "D_num_row, D_num_col (after converting into standard form)\n";
    std::cout << model_parameters.D.num_row << ", " << model_parameters.D.num_col << std::endl;
    writeFile << model_parameters.D.num_row << ", " << model_parameters.D.num_col << std::endl;
    // set up initial incumbent solution
    long x_size = model_parameters.c.num_entry;
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
                        kNNSet_new.push_back(index);
                    }
                    std::cout << std::endl;
                    writeFile << std::endl;
                }
            }
            else {
                // update the kNN set
                for (int index = 0; index < orderSet.size(); ++index) {
                    if (orderSet[index] <= k_new) {
                        kNNSet_new.push_back(index);
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
    // presolve problem to get x_incumbent
    x_incumbent = twoStageLP_presolve(model_parameters, knn_point_estimate, RHSmap);
    std::cout << "Incumbent solution after presolve:\n";
    writeFile << "Incumbent solution after presolve:\n";
    for (int idx_x = 0; idx_x < x_incumbent.size() - 1; ++idx_x) {
        std::cout << x_incumbent[idx_x] << ", ";
        writeFile << x_incumbent[idx_x] << ", ";
    }
    std::cout << x_incumbent[x_incumbent.size() - 1] << std::endl;
    writeFile << x_incumbent[x_incumbent.size() - 1] << std::endl;
    // initialize explored dual multipliers in the second stage
    std::vector<dualMultipliers> explored_duals;
    // obtain duals at the presolve points
    for (int idx_knn = 0; idx_knn < kNNSet.size(); ++idx_knn) {
        dualMultipliers new_dual = twoStageLP_secondStageDual(x_incumbent, model_parameters, RHS_dataset[kNNSet[idx_knn]], RHSmap);
        if (new_dual.feasible_flag == false) { // if second stage subproblem is infeasible
            //
            std::cout << "Warning: An infeasible case is captured.\n";
            writeFile << "Warning: An infeasible case is captured.\n";
            std::cout << "Warning: A feasibility cut will be constructed.\n";
            writeFile << "Warning: A feasibility cut will be constructed.\n";
            dualMultipliers extremeRay_scenario = twoStageLP_secondStageExtremRay(x_incumbent, model_parameters, RHS_dataset[kNNSet[idx_knn]], RHSmap);
            feasibilityCut feasibilityCut_scenario = twoStageLP_feasibilityCutGeneration(extremeRay_scenario, model_parameters, RHS_dataset[kNNSet[idx_knn]], RHSmap);
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
            if (if_new_dual(explored_duals, new_dual)) { // if new dual is found
                explored_duals.push_back(new_dual);
            }
        }
    }
    //
    std::cout << "Number of unique duals explored in presolve process: " << explored_duals.size() << std::endl;
    writeFile << "Number of unique duals explored in presolve process: " << explored_duals.size() << std::endl;
    // initialize a collection of minorants
    std::vector<minorant> minorant_collection;
    // construct initial minorant
    std::cout << "Construct initial minorant.\n";
    writeFile << "Construct initial minorant.\n";
    minorant initial_minorant;
    initial_minorant.alpha = f_lowerbound; // should use lower bound for the intercept of the initial minorant
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
    // initialize the index for the datapoint
    int idx_datapoint = N_pre - 1;
    N = N_pre; // update number of data points collected
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
        for (auto it = model_parameters.c.vec.begin(); it != model_parameters.c.vec.end(); ++it) {
            expr_obj += (it -> second) * x_temp[it -> first];
        }
        for (int x_index = 0; x_index < A_colsize; ++x_index) {
            expr_obj += 0.5 * sigma * (x_temp[x_index] * x_temp[x_index] - 2.0 * x_temp[x_index] * x_incumbent[x_index] + x_incumbent[x_index] * x_incumbent[x_index]);
        }
        expr_obj += eta;
        IloObjective obj = IloMinimize(env,expr_obj); // objective function
        mod.add(obj);
        // constraints
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
            if (duals[index] < SOLVER_PRECISION_LOWER || duals[index] > SOLVER_PRECISION_UPPER) { // only store the active minorants whose duals are significantly different from 0
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
                        kNNSet_new.push_back(index);
                    }
                    std::cout << std::endl;
                    writeFile << std::endl;
                }
            }
            else {
                // update the kNN set
                for (int index = 0; index < orderSet.size(); ++index) {
                    if (orderSet[index] <= k_new) {
                        kNNSet_new.push_back(index);
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
        // DUAL SPACE EXPLORATION
        //std::cout << "===DUAL SPACE EXPLORATION===\n";
        //writeFile << "===DUAL SPACE EXPLORATION===\n";
        // calculate the dual multipliers
        dualMultipliers dualsTemp = twoStageLP_secondStageDual(x_candidate, model_parameters, RHS_datapoint, RHSmap);
        if (dualsTemp.feasible_flag == false) { // if second stage subproblem is infeasible
            //
            std::cout << "Warning: An infeasible case is captured in iteration " << iteration << ".\n";
            writeFile << "Warning: An infeasible case is captured in iteration " << iteration << ".\n";
            std::cout << "Warning: A feasibility cut will be constructed.\n";
            writeFile << "Warning: A feasibility cut will be constructed.\n";
            dualMultipliers extremeRay_scenario = twoStageLP_secondStageExtremRay(x_candidate, model_parameters, RHS_datapoint, RHSmap);
            feasibilityCut feasibilityCut_scenario = twoStageLP_feasibilityCutGeneration(extremeRay_scenario, model_parameters, RHS_datapoint, RHSmap);
            // add feasibility cut
            feasibility_cuts.push_back(feasibilityCut_scenario);
            int idx_dual = rand() % (k_new) + 1;
            explored_duals.push_back(explored_duals[kNNSet[idx_dual]]);// still randomly add an existed dual
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
            if (if_new_dual(explored_duals, dualsTemp)) { // if new dual is found
                explored_duals.push_back(dualsTemp); // also store pi_C and pi_e
            }
        }
        std::cout << "Number of unique duals: " << explored_duals.size() << std::endl;
        writeFile << "Number of unique duals: " << explored_duals.size() << std::endl;
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
        for (int index = 0; index < k_new; ++index) {
            double max_value = -99999; // NOTE: need to make it smaller
            int max_index = -1;
            int max_index_incumbent = -1;
            double alpha_candidate = 0;
            double alpha_incumbent = 0;
            std::vector<double> beta_candidate;
            std::vector<double> beta_incumbent;
            // incumbent
            double max_value_incumbent = -99999; // NOTE: need to make it smaller
            for (int dual_index = 0; dual_index < explored_duals.size(); ++dual_index) {
                // find optimal dual based on the given set of unique duals
                double current_value = 0;
                // deterministic e
                double pi_e = model_parameters.e * explored_duals[dual_index].equality; // (IMPORTANT) need to allocate space to store it
                // stochastic e
                // equality part
                for (int idx_eq = 0; idx_eq < RHS_dataset[kNNSet[index]].be.size(); ++idx_eq) {
                    pi_e += explored_duals[dual_index].equality[idx_eq] * RHS_dataset[kNNSet[index]].be[idx_eq];
                }
                // inequality part (before standardizing) inequality constraint is after the equality constraints
                for (int idx_ineq = 0; idx_ineq < RHS_dataset[kNNSet[index]].bi.size(); ++idx_ineq) {
                    pi_e += explored_duals[dual_index].equality[RHSmap.bi_map[idx_ineq] + model_parameters.num_eq] * RHS_dataset[kNNSet[index]].bi[idx_ineq];
                }
                current_value += pi_e;
                // determinitsic C
                std::vector<double> pi_C = explored_duals[dual_index].equality * model_parameters.C; // (IMPORTANT) need to allocate space to store it
                // stochastic C
                // equality
                for (int idx_Ce = 0; idx_Ce < RHS_dataset[kNNSet[index]].Ce.size(); ++idx_Ce) {
                    pi_C[RHSmap.Ce_map[idx_Ce].second] += -1.0 * RHS_dataset[kNNSet[index]].Ce[idx_Ce] * explored_duals[dual_index].equality[RHSmap.Ce_map[idx_Ce].first];
                }
                // inequality before standardizing
                for (int idx_Ci = 0; idx_Ci < RHS_dataset[kNNSet[index]].Ci.size(); ++idx_Ci) {
                    pi_C[RHSmap.Ce_map[idx_Ci].second] += -1.0 * RHS_dataset[kNNSet[index]].Ci[idx_Ci] * explored_duals[dual_index].equality[RHSmap.Ci_map[idx_Ci].first + model_parameters.num_eq];
                }
                current_value += (-1.0) * (pi_C * x_candidate);
                double current_value_incumbent = 0;
                current_value_incumbent += pi_e;
                current_value_incumbent += (-1.0) * (pi_C * x_incumbent);
                if (dual_index < 1) {
                    max_index = dual_index;
                    max_value = current_value;
                    max_index_incumbent = dual_index;
                    max_value_incumbent = current_value_incumbent;
                    // store the intercept and slope
                    alpha_candidate = pi_e; // \pi^\top e
                    alpha_incumbent = pi_e;
                    beta_candidate = (-1.0) * pi_C; // -\pi^\top C
                    beta_incumbent = (-1.0) * pi_C;
                }
                else {
                    if (max_value < current_value) { // find the better dual for given candidate
                        max_index = dual_index;
                        max_value = current_value;
                        alpha_candidate = pi_e;
                        beta_candidate = (-1.0) * pi_C;
                    }
                    if (max_value_incumbent < current_value_incumbent) { // find the better dual for given incumbent
                        max_index_incumbent = dual_index;
                        max_value_incumbent = current_value_incumbent;
                        alpha_incumbent = pi_e;
                        beta_incumbent = (-1.0) * pi_C;
                    }
                }
            }
            // minorant on the candidate
            minorant_candidate.alpha += (1.0 / (double) k_new) * alpha_candidate;
            minorant_candidate.beta = minorant_candidate.beta + (1.0 / (double) k_new) * beta_candidate;
            // minorant on the incumbent
            minorant_incumbent.alpha += (1.0 / (double) k_new) * alpha_incumbent;
            minorant_incumbent.beta = minorant_incumbent.beta + (1.0 / (double) k_new) * beta_incumbent;
        }
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
        bool flag_incumbent_selection = incumbent_selection_check(q, x_candidate, x_incumbent, model_parameters.c, minorant_collection, minorant_collection_new, active_minorants);
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
    } // end of main loop
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
    writeFile << "*******************************************\n";
    
    writeFile.close();
    return x_incumbent;
}


// NSD basic full dual v3
// NSD solver with presolve, all the explored duals will be used, extra memory to store pi*C_det and pi*e_det
std::vector<double> dynamic_sdknn_solver_presolve_fullDual_v3(const std::string& folder_path, int max_iterations, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug) {
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
    bool flag_be; // tell if be stochastic is generated
    bool flag_bi; // tell if bi stochastic is generated
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
    std::string resultsOutput_path = folder_path + "/computationalResults(NSDv3.0basic).txt";
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
    standardTwoStageParameters model_parameters;
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
    // read model file
    model_parameters = readStandardTwoStageParameters(model_path);
    // read sto file
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
    std::cout << "SD-kNN (fast version with presolve v3.0) is initialized\n";
    writeFile << "SD-kNN (fast version with presolve v3.0) is initialized\n";
    std::cout << "Algorithmic Parameters\n";
    writeFile << "Algorithmic Parameters\n";
    std::cout << "sigma, q, beta, k, N, N_pre, sigma_lower, sigma_upper" << std::endl;
    writeFile << "sigma, q, beta, k, N, N_pre, sigma_lower, sigma_upper" << std::endl;
    std::cout << sigma << ", " << q << ", " << beta << ", " << k << ", " << N << ", " << N_pre << ", " << sigma_lowerbound << ", " << sigma_upperbound << std::endl;
    writeFile << sigma << ", " << q << ", " << beta << ", " << k << ", " << N << ", " << N_pre << ", " << sigma_lowerbound << ", " << sigma_upperbound << std::endl;
    std::cout << "Problem Complexity\n";
    writeFile << "Problem Complexity\n";
    std::cout << "A_num_row, A_num_col\n";
    writeFile << "A_num_row, A_num_col\n";
    std::cout << model_parameters.A.num_row << ", " << model_parameters.A.num_col << std::endl;
    writeFile << model_parameters.A.num_row << ", " << model_parameters.A.num_col << std::endl;
    std::cout << "D_num_row, D_num_col (after converting into standard form)\n";
    writeFile << "D_num_row, D_num_col (after converting into standard form)\n";
    std::cout << model_parameters.D.num_row << ", " << model_parameters.D.num_col << std::endl;
    writeFile << model_parameters.D.num_row << ", " << model_parameters.D.num_col << std::endl;
    // set up initial incumbent solution
    long x_size = model_parameters.c.num_entry;
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
                        kNNSet_new.push_back(index);
                    }
                    std::cout << std::endl;
                    writeFile << std::endl;
                }
            }
            else {
                // update the kNN set
                for (int index = 0; index < orderSet.size(); ++index) {
                    if (orderSet[index] <= k_new) {
                        kNNSet_new.push_back(index);
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
    // presolve problem to get x_incumbent
    x_incumbent = twoStageLP_presolve(model_parameters, knn_point_estimate, RHSmap);
    std::cout << "Incumbent solution after presolve:\n";
    writeFile << "Incumbent solution after presolve:\n";
    for (int idx_x = 0; idx_x < x_incumbent.size() - 1; ++idx_x) {
        std::cout << x_incumbent[idx_x] << ", ";
        writeFile << x_incumbent[idx_x] << ", ";
    }
    std::cout << x_incumbent[x_incumbent.size() - 1] << std::endl;
    writeFile << x_incumbent[x_incumbent.size() - 1] << std::endl;
    // initialize explored dual multipliers in the second stage
    std::vector<dualMultipliers> explored_duals;
    std::vector<double> pi_e_collection;
    std::vector<std::vector<double>> pi_C_collection;
    // obtain duals at the presolve points
    for (int idx_knn = 0; idx_knn < kNNSet.size(); ++idx_knn) {
        dualMultipliers new_dual = twoStageLP_secondStageDual(x_incumbent, model_parameters, RHS_dataset[kNNSet[idx_knn]], RHSmap);
        if (new_dual.feasible_flag == false) { // if second stage subproblem is infeasible
            //
            std::cout << "Warning: An infeasible case is captured.\n";
            writeFile << "Warning: An infeasible case is captured.\n";
            std::cout << "Warning: A feasibility cut will be constructed.\n";
            writeFile << "Warning: A feasibility cut will be constructed.\n";
            dualMultipliers extremeRay_scenario = twoStageLP_secondStageExtremRay(x_incumbent, model_parameters, RHS_dataset[kNNSet[idx_knn]], RHSmap);
            feasibilityCut feasibilityCut_scenario = twoStageLP_feasibilityCutGeneration(extremeRay_scenario, model_parameters, RHS_dataset[kNNSet[idx_knn]], RHSmap);
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
            if (if_new_dual(explored_duals, new_dual)) { // if new dual is found
                explored_duals.push_back(new_dual);
                // deterministic e
                double pi_e = model_parameters.e * new_dual.equality;
                pi_e_collection.push_back(pi_e);
                // determinictic C
                std::vector<double> pi_C = new_dual.equality * model_parameters.C;
                pi_C_collection.push_back(pi_C);
            }
        }
    }
    //
    std::cout << "Number of unique duals explored in presolve process: " << explored_duals.size() << std::endl;
    writeFile << "Number of unique duals explored in presolve process: " << explored_duals.size() << std::endl;
    // initialize a collection of minorants
    std::vector<minorant> minorant_collection;
    // construct initial minorant
    std::cout << "Construct initial minorant.\n";
    writeFile << "Construct initial minorant.\n";
    minorant initial_minorant;
    initial_minorant.alpha = f_lowerbound; // should use lower bound for the intercept of the initial minorant
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
    // initialize the index for the datapoint
    int idx_datapoint = N_pre - 1;
    N = N_pre; // update number of data points collected
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
        //IloNumVar eta(env,f_lowerbound,IloInfinity,ILOFLOAT);
        mod.add(x_temp);
        mod.add(eta);
        IloExpr expr_obj(env);
        for (auto it = model_parameters.c.vec.begin(); it != model_parameters.c.vec.end(); ++it) {
            expr_obj += (it -> second) * x_temp[it -> first];
        }
        for (int x_index = 0; x_index < A_colsize; ++x_index) {
            expr_obj += 0.5 * sigma * (x_temp[x_index] * x_temp[x_index] - 2.0 * x_temp[x_index] * x_incumbent[x_index] + x_incumbent[x_index] * x_incumbent[x_index]);
        }
        expr_obj += eta;
        IloObjective obj = IloMinimize(env,expr_obj); // objective function
        mod.add(obj);
        // constraints
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
            if (duals[index] < SOLVER_PRECISION_LOWER || duals[index] > SOLVER_PRECISION_UPPER) { // only store the active minorants whose duals are significantly different from 0
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
                        kNNSet_new.push_back(index);
                    }
                    std::cout << std::endl;
                    writeFile << std::endl;
                }
            }
            else {
                // update the kNN set
                for (int index = 0; index < orderSet.size(); ++index) {
                    if (orderSet[index] <= k_new) {
                        kNNSet_new.push_back(index);
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
        // DUAL SPACE EXPLORATION
        //std::cout << "===DUAL SPACE EXPLORATION===\n";
        //writeFile << "===DUAL SPACE EXPLORATION===\n";
        // calculate the dual multipliers
        dualMultipliers dualsTemp = twoStageLP_secondStageDual(x_candidate, model_parameters, RHS_datapoint, RHSmap);
        if (dualsTemp.feasible_flag == false) { // if second stage subproblem is infeasible
            //
            std::cout << "Warning: An infeasible case is captured in iteration " << iteration << ".\n";
            writeFile << "Warning: An infeasible case is captured in iteration " << iteration << ".\n";
            std::cout << "Warning: A feasibility cut will be constructed.\n";
            writeFile << "Warning: A feasibility cut will be constructed.\n";
            dualMultipliers extremeRay_scenario = twoStageLP_secondStageExtremRay(x_candidate, model_parameters, RHS_datapoint, RHSmap);
            feasibilityCut feasibilityCut_scenario = twoStageLP_feasibilityCutGeneration(extremeRay_scenario, model_parameters, RHS_datapoint, RHSmap);
            // add feasibility cut
            feasibility_cuts.push_back(feasibilityCut_scenario);
            int idx_dual = rand() % (k_new) + 1;
            explored_duals.push_back(explored_duals[kNNSet[idx_dual]]);// still randomly add an existed dual
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
            if (if_new_dual(explored_duals, dualsTemp)) { // if new dual is found
                explored_duals.push_back(dualsTemp); // also store pi_C and pi_e
                // deterministic e
                double pi_e = model_parameters.e * dualsTemp.equality;
                pi_e_collection.push_back(pi_e);
                // determinictic C
                std::vector<double> pi_C = dualsTemp.equality * model_parameters.C;
                pi_C_collection.push_back(pi_C);
            }
        }
        std::cout << "Number of unique duals: " << explored_duals.size() << std::endl;
        writeFile << "Number of unique duals: " << explored_duals.size() << std::endl;
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
        for (int index = 0; index < k_new; ++index) {
            double max_value = -99999; // NOTE: need to make it smaller
            int max_index = -1;
            int max_index_incumbent = -1;
            double alpha_candidate = 0;
            double alpha_incumbent = 0;
            std::vector<double> beta_candidate;
            std::vector<double> beta_incumbent;
            // incumbent
            double max_value_incumbent = -99999; // NOTE: need to make it smaller
            for (int dual_index = 0; dual_index < explored_duals.size(); ++dual_index) {
                // find optimal dual based on the given set of unique duals
                double current_value = 0;
                // deterministic e
                //double pi_e = model_parameters.e * explored_duals[dual_index].equality; // (IMPORTANT) need to allocate space to store it
                double pi_e = pi_e_collection[dual_index];
                // stochastic e
                // equality part
                for (int idx_eq = 0; idx_eq < RHS_dataset[kNNSet[index]].be.size(); ++idx_eq) {
                    pi_e += explored_duals[dual_index].equality[idx_eq] * RHS_dataset[kNNSet[index]].be[idx_eq];
                }
                // inequality part (before standardizing) inequality constraint is after the equality constraints
                for (int idx_ineq = 0; idx_ineq < RHS_dataset[kNNSet[index]].bi.size(); ++idx_ineq) {
                    pi_e += explored_duals[dual_index].equality[RHSmap.bi_map[idx_ineq] + model_parameters.num_eq] * RHS_dataset[kNNSet[index]].bi[idx_ineq];
                }
                current_value += pi_e;
                // determinitsic C
                //std::vector<double> pi_C = explored_duals[dual_index].equality * model_parameters.C; // (IMPORTANT) need to allocate space to store it
                std::vector<double> pi_C = pi_C_collection[dual_index];
                // stochastic C
                // equality (Feb 17, 2022 may remove -1.0)
                for (int idx_Ce = 0; idx_Ce < RHS_dataset[kNNSet[index]].Ce.size(); ++idx_Ce) {
                    pi_C[RHSmap.Ce_map[idx_Ce].second] += -1.0 * RHS_dataset[kNNSet[index]].Ce[idx_Ce] * explored_duals[dual_index].equality[RHSmap.Ce_map[idx_Ce].first];
                }
                // inequality before standardizing
                for (int idx_Ci = 0; idx_Ci < RHS_dataset[kNNSet[index]].Ci.size(); ++idx_Ci) {
                    pi_C[RHSmap.Ce_map[idx_Ci].second] += -1.0 * RHS_dataset[kNNSet[index]].Ci[idx_Ci] * explored_duals[dual_index].equality[RHSmap.Ci_map[idx_Ci].first + model_parameters.num_eq];
                }
                current_value += (-1.0) * (pi_C * x_candidate);
                double current_value_incumbent = 0;
                current_value_incumbent += pi_e;
                current_value_incumbent += (-1.0) * (pi_C * x_incumbent);
                if (dual_index < 1) {
                    max_index = dual_index;
                    max_value = current_value;
                    max_index_incumbent = dual_index;
                    max_value_incumbent = current_value_incumbent;
                    // store the intercept and slope
                    alpha_candidate = pi_e; // \pi^\top e
                    alpha_incumbent = pi_e;
                    beta_candidate = (-1.0) * pi_C; // -\pi^\top C
                    beta_incumbent = (-1.0) * pi_C;
                }
                else {
                    if (max_value < current_value) { // find the better dual for given candidate
                        max_index = dual_index;
                        max_value = current_value;
                        alpha_candidate = pi_e;
                        beta_candidate = (-1.0) * pi_C;
                    }
                    if (max_value_incumbent < current_value_incumbent) { // find the better dual for given incumbent
                        max_index_incumbent = dual_index;
                        max_value_incumbent = current_value_incumbent;
                        alpha_incumbent = pi_e;
                        beta_incumbent = (-1.0) * pi_C;
                    }
                }
            }
            // minorant on the candidate
            minorant_candidate.alpha += (1.0 / (double) k_new) * alpha_candidate;
            minorant_candidate.beta = minorant_candidate.beta + (1.0 / (double) k_new) * beta_candidate;
            // minorant on the incumbent
            minorant_incumbent.alpha += (1.0 / (double) k_new) * alpha_incumbent;
            minorant_incumbent.beta = minorant_incumbent.beta + (1.0 / (double) k_new) * beta_incumbent;
        }
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
        bool flag_incumbent_selection = incumbent_selection_check(q, x_candidate, x_incumbent, model_parameters.c, minorant_collection, minorant_collection_new, active_minorants);
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
    } // end of main loop
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
    writeFile << "*******************************************\n";
    
    writeFile.close();
    return x_incumbent;
}



// NSD solver with presolve and batch size (skip the incumbent selection and explored duals in the batch), all the explored duals will be used
std::vector<double> dynamic_sdknn_solver(const std::string& folder_path, int max_iterations, int batch_size,double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug) {
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
    bool flag_be; // tell if be stochastic is generated
    bool flag_bi; // tell if bi stochastic is generated
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
    std::string resultsOutput_path = folder_path + "/computationalResults(NSDv2.0presolve_batch).txt";
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
    standardTwoStageParameters model_parameters;
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
    // read model file
    model_parameters = readStandardTwoStageParameters(model_path);
    // read sto file
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
    std::cout << "SD-kNN (fast version with presolve v2.0) is initialized\n";
    writeFile << "SD-kNN (fast version with presolve v2.0) is initialized\n";
    std::cout << "Algorithmic Parameters\n";
    writeFile << "Algorithmic Parameters\n";
    std::cout << "sigma, q, beta, k, N, N_pre, sigma_lower, sigma_upper" << std::endl;
    writeFile << "sigma, q, beta, k, N, N_pre, sigma_lower, sigma_upper" << std::endl;
    std::cout << sigma << ", " << q << ", " << beta << ", " << k << ", " << N << ", " << N_pre << ", " << sigma_lowerbound << ", " << sigma_upperbound << std::endl;
    writeFile << sigma << ", " << q << ", " << beta << ", " << k << ", " << N << ", " << N_pre << ", " << sigma_lowerbound << ", " << sigma_upperbound << std::endl;
    std::cout << "Problem Complexity\n";
    writeFile << "Problem Complexity\n";
    std::cout << "A_num_row, A_num_col\n";
    writeFile << "A_num_row, A_num_col\n";
    std::cout << model_parameters.A.num_row << ", " << model_parameters.A.num_col << std::endl;
    writeFile << model_parameters.A.num_row << ", " << model_parameters.A.num_col << std::endl;
    std::cout << "D_num_row, D_num_col (after converting into standard form)\n";
    writeFile << "D_num_row, D_num_col (after converting into standard form)\n";
    std::cout << model_parameters.D.num_row << ", " << model_parameters.D.num_col << std::endl;
    writeFile << model_parameters.D.num_row << ", " << model_parameters.D.num_col << std::endl;
    // set up initial incumbent solution
    long x_size = model_parameters.c.num_entry;
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
                        kNNSet_new.push_back(index);
                    }
                    std::cout << std::endl;
                    writeFile << std::endl;
                }
            }
            else {
                // update the kNN set
                for (int index = 0; index < orderSet.size(); ++index) {
                    if (orderSet[index] <= k_new) {
                        kNNSet_new.push_back(index);
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
    // presolve problem to get x_incumbent
    x_incumbent = twoStageLP_presolve(model_parameters, knn_point_estimate, RHSmap);
    std::cout << "Incumbent solution after presolve:\n";
    writeFile << "Incumbent solution after presolve:\n";
    for (int idx_x = 0; idx_x < x_incumbent.size() - 1; ++idx_x) {
        std::cout << x_incumbent[idx_x] << ", ";
        writeFile << x_incumbent[idx_x] << ", ";
    }
    std::cout << x_incumbent[x_incumbent.size() - 1] << std::endl;
    writeFile << x_incumbent[x_incumbent.size() - 1] << std::endl;
    // initialize explored dual multipliers in the second stage
    std::vector<dualMultipliers> explored_duals;
    // obtain duals at the presolve points
    for (int idx_knn = 0; idx_knn < kNNSet.size(); ++idx_knn) {
        dualMultipliers new_dual = twoStageLP_secondStageDual(x_incumbent, model_parameters, RHS_dataset[kNNSet[idx_knn]], RHSmap);
        if (new_dual.feasible_flag == false) { // if second stage subproblem is infeasible
            //
            std::cout << "Warning: An infeasible case is captured.\n";
            writeFile << "Warning: An infeasible case is captured.\n";
            std::cout << "Warning: A feasibility cut will be constructed.\n";
            writeFile << "Warning: A feasibility cut will be constructed.\n";
            dualMultipliers extremeRay_scenario = twoStageLP_secondStageExtremRay(x_incumbent, model_parameters, RHS_dataset[kNNSet[idx_knn]], RHSmap);
            feasibilityCut feasibilityCut_scenario = twoStageLP_feasibilityCutGeneration(extremeRay_scenario, model_parameters, RHS_dataset[kNNSet[idx_knn]], RHSmap);
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
            if (if_new_dual(explored_duals, new_dual)) { // if new dual is found
                explored_duals.push_back(new_dual);
            }
        }
    }
    //
    std::cout << "Number of unique duals explored in presolve process: " << explored_duals.size() << std::endl;
    writeFile << "Number of unique duals explored in presolve process: " << explored_duals.size() << std::endl;
    // initialize a collection of minorants
    std::vector<minorant> minorant_collection;
    //std::vector<minorant> active_minorant_collection;
    // construct initial minorant
    std::cout << "Construct initial minorant.\n";
    writeFile << "Construct initial minorant.\n";
    minorant initial_minorant;
    initial_minorant.alpha = f_lowerbound; // should use lower bound for the intercept of the initial minorant 
    for (int idx_x = 0; idx_x < x_size; ++idx_x) {
        initial_minorant.beta.push_back(0);
    }
    minorant_collection.push_back(initial_minorant);
    //active_minorant_collection.push_back(initial_minorant);
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
    // initialize the index for the datapoint
    int idx_datapoint = N_pre - 1;
    N = N_pre; // update number of data points collected
    k = k_new;
    std::vector<double> x_candidate(model_parameters.c.num_entry,0.0);
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        std::cout << "***Iteration " << iteration << "***\n";
        writeFile << "***Iteration " << iteration << "***\n";
        std::cout << "sigma: " << sigma << std::endl;
        writeFile << "sigma: " << sigma << std::endl;
        N += 1; // increase sample size
        idx_datapoint += 1; // go to the next data point
        k_new = (int) pow(N, beta); // calculate new k
        std::cout << "k (number of nearest neighbor): " << k_new << std::endl;
        writeFile << "k (number of nearest neighbor): " << k_new << std::endl;
        std::cout << "k old: " << k << std::endl;// debug
        writeFile << "k old: " << k << std::endl;
        // skip the incumbent selection and dual selection when the iteration is not a multiple of batch size
        if (iteration % batch_size == 0 || k_new != k){ // iteration is the multiple of batch size or k increases
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
            for (auto it = model_parameters.c.vec.begin(); it != model_parameters.c.vec.end(); ++it) {
                expr_obj += (it -> second) * x_temp[it -> first];
            }
            for (int x_index = 0; x_index < A_colsize; ++x_index) {
                expr_obj += 0.5 * sigma * (x_temp[x_index] * x_temp[x_index] - 2.0 * x_temp[x_index] * x_incumbent[x_index] + x_incumbent[x_index] * x_incumbent[x_index]);
            }
            expr_obj += eta;
            IloObjective obj = IloMinimize(env,expr_obj); // objective function
            mod.add(obj);
            // constraints
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
                x_candidate[x_index] = cplex.getValue(x_temp[x_index]);
                //std::cout << cplex.getValue(x_temp[x_index]) << std::endl;
            }
            // update the set of active minorants
            IloNumArray duals(env);
            cplex.getDuals(duals, minorant_constraints);
            std::vector<int> active_minorants_new;
            int num_active_minorants = 0;
            for (int index = 0; index < active_minorants.size(); ++index) {
                if (duals[index] < SOLVER_PRECISION_LOWER || duals[index] > SOLVER_PRECISION_UPPER) { // only store the active minorants whose duals are significantly different from 0
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
                            kNNSet_new.push_back(index);
                        }
                        std::cout << std::endl;
                        writeFile << std::endl;
                 }
                 }
                 else {
                 // update the kNN set
                 for (int index = 0; index < orderSet.size(); ++index) {
                        if (orderSet[index] <= k_new) {
                            kNNSet_new.push_back(index);
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
            // DUAL SPACE EXPLORATION
            //std::cout << "===DUAL SPACE EXPLORATION===\n";
            //writeFile << "===DUAL SPACE EXPLORATION===\n";
            // calculate the dual multipliers
            dualMultipliers dualsTemp = twoStageLP_secondStageDual(x_candidate, model_parameters, RHS_datapoint, RHSmap);
            if (dualsTemp.feasible_flag == false) { // if second stage subproblem is infeasible
                //
                std::cout << "Warning: An infeasible case is captured in iteration " << iteration << ".\n";
                writeFile << "Warning: An infeasible case is captured in iteration " << iteration << ".\n";
                std::cout << "Warning: A feasibility cut will be constructed.\n";
                writeFile << "Warning: A feasibility cut will be constructed.\n";
                dualMultipliers extremeRay_scenario = twoStageLP_secondStageExtremRay(x_candidate, model_parameters, RHS_datapoint, RHSmap);
                feasibilityCut feasibilityCut_scenario = twoStageLP_feasibilityCutGeneration(extremeRay_scenario, model_parameters, RHS_datapoint, RHSmap);
                // add feasibility cut
                feasibility_cuts.push_back(feasibilityCut_scenario);
                int idx_dual = rand() % (k_new) + 1;
                explored_duals.push_back(explored_duals[kNNSet[idx_dual]]);// still randomly add an existed dual
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
                if (if_new_dual(explored_duals, dualsTemp)) { // if new dual is found
                    explored_duals.push_back(dualsTemp);
                }
            }
            std::cout << "Number of unique duals: " << explored_duals.size() << std::endl;
            writeFile << "Number of unique duals: " << explored_duals.size() << std::endl;
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
            for (int index = 0; index < k_new; ++index) {
                double max_value = -99999; // NOTE: need to make it smaller
                int max_index = -1;
                int max_index_incumbent = -1;
                double alpha_candidate = 0;
                double alpha_incumbent = 0;
                std::vector<double> beta_candidate;
                std::vector<double> beta_incumbent;
                // incumbent
                double max_value_incumbent = -99999; // NOTE: need to make it smaller
                for (int dual_index = 0; dual_index < explored_duals.size(); ++dual_index) {
                    // find optimal dual based on the given set of unique duals
                    double current_value = 0;
                    // deterministic e
                    double pi_e = model_parameters.e * explored_duals[dual_index].equality;
                    // stochastic e
                    // equality part
                    for (int idx_eq = 0; idx_eq < RHS_dataset[kNNSet[index]].be.size(); ++idx_eq) {
                        pi_e += explored_duals[dual_index].equality[idx_eq] * RHS_dataset[kNNSet[index]].be[idx_eq];
                    }
                    // inequality part (before standardizing) inequality constraint is after the equality constraints
                    for (int idx_ineq = 0; idx_ineq < RHS_dataset[kNNSet[index]].bi.size(); ++idx_ineq) {
                        pi_e += explored_duals[dual_index].equality[RHSmap.bi_map[idx_ineq] + model_parameters.num_eq] * RHS_dataset[kNNSet[index]].bi[idx_ineq];
                    }
                    current_value += pi_e;
                    // determinitsic C
                    std::vector<double> pi_C = explored_duals[dual_index].equality * model_parameters.C;
                    // stochastic C
                    // equality
                    for (int idx_Ce = 0; idx_Ce < RHS_dataset[kNNSet[index]].Ce.size(); ++idx_Ce) {
                        pi_C[RHSmap.Ce_map[idx_Ce].second] += -1.0 * RHS_dataset[kNNSet[index]].Ce[idx_Ce] * explored_duals[dual_index].equality[RHSmap.Ce_map[idx_Ce].first];
                    }
                    // inequality before standardizing
                    for (int idx_Ci = 0; idx_Ci < RHS_dataset[kNNSet[index]].Ci.size(); ++idx_Ci) {
                        pi_C[RHSmap.Ce_map[idx_Ci].second] += -1.0 * RHS_dataset[kNNSet[index]].Ci[idx_Ci] * explored_duals[dual_index].equality[RHSmap.Ci_map[idx_Ci].first + model_parameters.num_eq];
                    }
                    current_value += (-1.0) * (pi_C * x_candidate);
                    double current_value_incumbent = 0;
                    current_value_incumbent += pi_e;
                    current_value_incumbent += (-1.0) * (pi_C * x_incumbent);
                    if (dual_index < 1) {
                        max_index = dual_index;
                        max_value = current_value;
                        max_index_incumbent = dual_index;
                        max_value_incumbent = current_value_incumbent;
                        // store the intercept and slope
                        alpha_candidate = pi_e; // \pi^\top e
                        alpha_incumbent = pi_e;
                        beta_candidate = (-1.0) * pi_C; // -\pi^\top C
                        beta_incumbent = (-1.0) * pi_C;
                    }
                    else {
                        if (max_value < current_value) { // find the better dual for given candidate
                            max_index = dual_index;
                            max_value = current_value;
                            alpha_candidate = pi_e;
                            beta_candidate = (-1.0) * pi_C;
                        }
                        if (max_value_incumbent < current_value_incumbent) { // find the better dual for given incumbent
                            max_index_incumbent = dual_index;
                            max_value_incumbent = current_value_incumbent;
                            alpha_incumbent = pi_e;
                            beta_incumbent = (-1.0) * pi_C;
                        }
                    }
                }
                // minorant on the candidate
                minorant_candidate.alpha += (1.0 / (double) k_new) * alpha_candidate;
                minorant_candidate.beta = minorant_candidate.beta + (1.0 / (double) k_new) * beta_candidate;
                // minorant on the incumbent
                minorant_incumbent.alpha += (1.0 / (double) k_new) * alpha_incumbent;
                minorant_incumbent.beta = minorant_incumbent.beta + (1.0 / (double) k_new) * beta_incumbent;
            }
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
            bool flag_incumbent_selection = incumbent_selection_check(q, x_candidate, x_incumbent, model_parameters.c, minorant_collection, minorant_collection_new, active_minorants);
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
                    /*
                    std::cout << "New alpha: "<<  minorant_collection[active_minorants_new[minorant_index]].alpha << std::endl; // debug
                     minorant_collection[active_minorants_new[minorant_index]].beta = ((double) k) / ((double) k_new) * minorant_collection[active_minorants_new[minorant_index]].beta;
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
            //int index_minorant_candidate = 2 * iteration + 1;
            int index_minorant_candidate = (int) minorant_collection.size() - 1;
            minorant_collection.push_back(minorant_incumbent);
            //int index_minorant_incumbent = 2 * (iteration + 1);
            int index_minorant_incumbent = (int) minorant_collection.size() - 1;
            active_minorants_new.push_back(index_minorant_candidate);
            active_minorants_new.push_back(index_minorant_incumbent);
            std::cout << "Minorant Index (candidate): " << index_minorant_candidate << std::endl;
            std::cout << "Minorant Index (incumbent): " << index_minorant_incumbent << std::endl;
            // final step of one iteration
            active_minorants = active_minorants_new; // update the indices of active minorants
            // update k
            k = k_new;
            //std::cout << "===(END) INCUMBENT SELECTION===\n";
            //writeFile << "===(END) INCUMBENT SELECTION===\n";
        }
        else { // skip the incumbent selection new cut calculation
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
                            kNNSet_new.push_back(index);
                        }
                        std::cout << std::endl;
                        writeFile << std::endl;
                 }
                 }
                 else {
                 // update the kNN set
                 for (int index = 0; index < orderSet.size(); ++index) {
                        if (orderSet[index] <= k_new) {
                            kNNSet_new.push_back(index);
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
            // DUAL SPACE EXPLORATION
            //std::cout << "===DUAL SPACE EXPLORATION===\n";
            //writeFile << "===DUAL SPACE EXPLORATION===\n";
            // calculate the dual multipliers
            dualMultipliers dualsTemp = twoStageLP_secondStageDual(x_candidate, model_parameters, RHS_datapoint, RHSmap);
            if (dualsTemp.feasible_flag == false) { // if second stage subproblem is infeasible
                //
                std::cout << "Warning: An infeasible case is captured in iteration " << iteration << ".\n";
                writeFile << "Warning: An infeasible case is captured in iteration " << iteration << ".\n";
                std::cout << "Warning: A feasibility cut will be constructed.\n";
                writeFile << "Warning: A feasibility cut will be constructed.\n";
                dualMultipliers extremeRay_scenario = twoStageLP_secondStageExtremRay(x_candidate, model_parameters, RHS_datapoint, RHSmap);
                feasibilityCut feasibilityCut_scenario = twoStageLP_feasibilityCutGeneration(extremeRay_scenario, model_parameters, RHS_datapoint, RHSmap);
                // add feasibility cut
                feasibility_cuts.push_back(feasibilityCut_scenario);
                int idx_dual = rand() % (k_new) + 1;
                explored_duals.push_back(explored_duals[kNNSet[idx_dual]]);// still randomly add an existed dual
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
                if (if_new_dual(explored_duals, dualsTemp)) { // if new dual is found
                    explored_duals.push_back(dualsTemp);
                }
            }
            std::cout << "Number of unique duals: " << explored_duals.size() << std::endl;
            writeFile << "Number of unique duals: " << explored_duals.size() << std::endl;
            //std::cout << "===(END) DUAL SPACE EXPLORATION===\n";
            //writeFile << "===(END) DUAL SPACE EXPLORATION===\n";
            // updates on the active minorants
            if (k == k_new) { // will have more advanced version to store the radius of kNN set
                for (int minorant_index = 0; minorant_index < active_minorants.size(); ++minorant_index) {
                    minorant_collection[active_minorants[minorant_index]].alpha += (f_lowerbound - f_upperbound) / ((double) k);
                }
            }
            else {
                for (int minorant_index = 0; minorant_index < active_minorants.size(); ++minorant_index) {
                    /*
                     std::cout << "Old alpha:" << minorant_collection[active_minorants_new[minorant_index]].alpha << std::endl; // debug
                     std::cout << "Old beta:\n";
                     for (int index = 0; index < x_size; ++index) {
                     std::cout << minorant_collection[active_minorants_new[minorant_index]].beta[index] << " ";
                     }
                     std::cout << std::endl;
                     */
                    minorant_collection[active_minorants[minorant_index]].alpha = minorant_collection[active_minorants[minorant_index]].alpha * ((double) k) / ((double) k_new) + ((double) (k_new - k)) / ((double) k_new) * f_lowerbound;
                    /*
                    std::cout << "New alpha: "<<  minorant_collection[active_minorants_new[minorant_index]].alpha << std::endl; // debug
                     minorant_collection[active_minorants_new[minorant_index]].beta = ((double) k) / ((double) k_new) * minorant_collection[active_minorants_new[minorant_index]].beta;
                     for (int index = 0; index < x_size; ++index) {
                     std::cout << minorant_collection[active_minorants_new[minorant_index]].beta[index] << " ";
                     }
                     std::cout << std::endl;
                     */
                }
            } // end minorant update
        }
    } // end of main loop
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
    writeFile << "*******************************************\n";
    
    writeFile.close();
    return x_incumbent;
}


std::vector<double> dynamic_sdknn_solver_v2(const std::string& folder_path, int max_iterations, int batch_size,double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug) {
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
    bool flag_be; // tell if be stochastic is generated
    bool flag_bi; // tell if bi stochastic is generated
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
    std::string resultsOutput_path = folder_path + "/computationalResults(NSDv2.0).txt";
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
    standardTwoStageParameters model_parameters;
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
    // read model file
    model_parameters = readStandardTwoStageParameters(model_path);
    // read sto file
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
    std::cout << "SD-kNN (fast version with presolve and batch sampling v2.0) is initialized\n";
    writeFile << "SD-kNN (fast version with presolve and batch sampling v2.0) is initialized\n";
    std::cout << "Algorithmic Parameters\n";
    writeFile << "Algorithmic Parameters\n";
    std::cout << "sigma, q, beta, k, N, N_pre, sigma_lower, sigma_upper" << std::endl;
    writeFile << "sigma, q, beta, k, N, N_pre, sigma_lower, sigma_upper" << std::endl;
    std::cout << sigma << ", " << q << ", " << beta << ", " << k << ", " << N << ", " << N_pre << ", " << sigma_lowerbound << ", " << sigma_upperbound << std::endl;
    writeFile << sigma << ", " << q << ", " << beta << ", " << k << ", " << N << ", " << N_pre << ", " << sigma_lowerbound << ", " << sigma_upperbound << std::endl;
    std::cout << "batch size" << std::endl;
    writeFile << "batch size" << std::endl;
    std::cout << batch_size << std::endl;
    writeFile << batch_size << std::endl;
    std::cout << "Problem Complexity\n";
    writeFile << "Problem Complexity\n";
    std::cout << "A_num_row, A_num_col\n";
    writeFile << "A_num_row, A_num_col\n";
    std::cout << model_parameters.A.num_row << ", " << model_parameters.A.num_col << std::endl;
    writeFile << model_parameters.A.num_row << ", " << model_parameters.A.num_col << std::endl;
    std::cout << "D_num_row, D_num_col (after converting into standard form)\n";
    writeFile << "D_num_row, D_num_col (after converting into standard form)\n";
    std::cout << model_parameters.D.num_row << ", " << model_parameters.D.num_col << std::endl;
    writeFile << model_parameters.D.num_row << ", " << model_parameters.D.num_col << std::endl;
    // set up initial incumbent solution
    long x_size = model_parameters.c.num_entry;
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
        }
    }
    k_new = (int) pow(N_pre, beta);
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
            }
            std::cout << std::endl;
            writeFile << std::endl;
        }
    }
    // update the kNN set
    kNNSet.clear(); // clear the old kNN set
    for (int index = 0; index < orderSet.size(); ++index) {
        if (orderSet[index] <= k_new) {
            kNNSet.push_back(index);
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
    // presolve problem to get x_incumbent
    x_incumbent = twoStageLP_presolve(model_parameters, knn_point_estimate, RHSmap);
    std::cout << "Incumbent solution after presolve:\n";
    writeFile << "Incumbent solution after presolve:\n";
    for (int idx_x = 0; idx_x < x_incumbent.size() - 1; ++idx_x) {
        std::cout << x_incumbent[idx_x] << ", ";
        writeFile << x_incumbent[idx_x] << ", ";
    }
    std::cout << x_incumbent[x_incumbent.size() - 1] << std::endl;
    writeFile << x_incumbent[x_incumbent.size() - 1] << std::endl;
    // initialize explored dual multipliers in the second stage
    std::vector<dualMultipliers> explored_duals;
    // obtain duals at the presolve points
    for (int idx_knn = 0; idx_knn < kNNSet.size(); ++idx_knn) {
        dualMultipliers new_dual = twoStageLP_secondStageDual(x_incumbent, model_parameters, RHS_dataset[kNNSet[idx_knn]], RHSmap);
        if (new_dual.feasible_flag == false) { // if second stage subproblem is infeasible
            //
            std::cout << "Warning: An infeasible case is captured.\n";
            writeFile << "Warning: An infeasible case is captured.\n";
            std::cout << "Warning: A feasibility cut will be constructed.\n";
            writeFile << "Warning: A feasibility cut will be constructed.\n";
            dualMultipliers extremeRay_scenario = twoStageLP_secondStageExtremRay(x_incumbent, model_parameters, RHS_dataset[kNNSet[idx_knn]], RHSmap);
            feasibilityCut feasibilityCut_scenario = twoStageLP_feasibilityCutGeneration(extremeRay_scenario, model_parameters, RHS_dataset[kNNSet[idx_knn]], RHSmap);
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
            if (if_new_dual(explored_duals, new_dual)) { // if new dual is found
                explored_duals.push_back(new_dual);
            }
        }
    }
    //
    std::cout << "Number of unique duals explored in presolve process: " << explored_duals.size() << std::endl;
    writeFile << "Number of unique duals explored in presolve process: " << explored_duals.size() << std::endl;
    // initialize a collection of minorants
    std::vector<minorant> minorant_collection;
    std::vector<minorant> active_minorant_collection;
    // construct initial minorant
    std::cout << "Construct initial minorant.\n";
    writeFile << "Construct initial minorant.\n";
    minorant initial_minorant;
    initial_minorant.alpha = f_lowerbound; // should use lower bound for the intercept of the initial minorant
    for (int idx_x = 0; idx_x < x_size; ++idx_x) {
        initial_minorant.beta.push_back(0);
    }
    minorant_collection.push_back(initial_minorant);
    active_minorant_collection.push_back(initial_minorant);
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
    // initialize the index for the datapoint
    int idx_datapoint = N_pre - 1;
    N = N_pre; // update number of data points collected
    k = k_new;
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        std::cout << "***Iteration " << iteration << "***\n";
        writeFile << "***Iteration " << iteration << "***\n";
        std::cout << "sigma: " << sigma << std::endl;
        writeFile << "sigma: " << sigma << std::endl;
        std::vector<double> x_candidate;
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
        for (auto it = model_parameters.c.vec.begin(); it != model_parameters.c.vec.end(); ++it) {
            expr_obj += (it -> second) * x_temp[it -> first];
        }
        for (int x_index = 0; x_index < A_colsize; ++x_index) {
            expr_obj += 0.5 * sigma * (x_temp[x_index] * x_temp[x_index] - 2.0 * x_temp[x_index] * x_incumbent[x_index] + x_incumbent[x_index] * x_incumbent[x_index]);
        }
        expr_obj += eta;
        IloObjective obj = IloMinimize(env,expr_obj); // objective function
        mod.add(obj);
        // constraints
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
        std::cout << "Number of minorants used in promxial mapping: " << active_minorant_collection.size() << std::endl;
        writeFile << "Number of minorants used in promxial mapping: " << active_minorant_collection.size() << std::endl;
        for (int index_cons = 0; index_cons < active_minorant_collection.size(); ++index_cons) {
            IloExpr expr(env);
            expr += active_minorant_collection[index_cons].alpha - eta;
            for (int index_x = 0; index_x < x_size; ++index_x ) {
                expr += active_minorant_collection[index_cons].beta[index_x] * x_temp[index_x];
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
        std::vector<minorant> active_minorant_collection_new;
        //std::vector<minorant> active_minorant_collection_temp;
        int num_active_minorants = 0;
        for (int index = 0; index < active_minorant_collection.size(); ++index) {
            if (duals[index] < SOLVER_PRECISION_LOWER || duals[index] > SOLVER_PRECISION_UPPER) { // only store the active minorants whose duals are significantly different from 0
                //std::cout << "dual: " << duals[index] << std::endl;
                active_minorant_collection_new.push_back(active_minorant_collection[index]);
                //active_minorant_collection_temp.push_back(active_minorant_collection[index]);
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
        // kNN in after adding a batch of samples
        std::cout << "Start batch sampling\n";
        writeFile << "Start batch sampling\n";
        for (int idx_bacth_datapoint = 0; idx_bacth_datapoint < batch_size; ++idx_bacth_datapoint) {
            N += 1; // increase sample size
            idx_datapoint += 1; // go to the next data point
            k_new = (int) pow(N, beta); // calculate new k
            std::cout << "N: " << N << std::endl;
            writeFile << "N: " << N << std::endl;
            std::cout << "k (number of nearest neighbor): " << k_new << std::endl;
            writeFile << "k (number of nearest neighbor): " << k_new << std::endl;
            std::cout << "k old: " << k << std::endl;// debug
            writeFile << "k old: " << k << std::endl;
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
                            kNNSet_new.push_back(index);
                        }
                        std::cout << std::endl;
                        writeFile << std::endl;
                    }
                }
                else {
                    // update the kNN set
                    for (int index = 0; index < orderSet.size(); ++index) {
                        if (orderSet[index] <= k_new) {
                            kNNSet_new.push_back(index);
                        }
                    }
                }
                 */
            }
            //*********************
            //end non-parametric estimation (kNN)
            //std::cout << "===(END) kNN ESTIMATION===\n";
            //writeFile << "===(END) kNN ESTIMATION===\n";
            // DUAL SPACE EXPLORATION
            //std::cout << "===DUAL SPACE EXPLORATION===\n";
            //writeFile << "===DUAL SPACE EXPLORATION===\n";
            // calculate the dual multipliers
            dualMultipliers dualsTemp = twoStageLP_secondStageDual(x_candidate, model_parameters, RHS_datapoint, RHSmap);
            if (dualsTemp.feasible_flag == false) { // if second stage subproblem is infeasible
                //
                std::cout << "Warning: An infeasible case is captured in iteration " << iteration << ".\n";
                writeFile << "Warning: An infeasible case is captured in iteration " << iteration << ".\n";
                std::cout << "Warning: A feasibility cut will be constructed.\n";
                writeFile << "Warning: A feasibility cut will be constructed.\n";
                dualMultipliers extremeRay_scenario = twoStageLP_secondStageExtremRay(x_candidate, model_parameters, RHS_datapoint, RHSmap);
                feasibilityCut feasibilityCut_scenario = twoStageLP_feasibilityCutGeneration(extremeRay_scenario, model_parameters, RHS_datapoint, RHSmap);
                // add feasibility cut
                feasibility_cuts.push_back(feasibilityCut_scenario);
                int idx_dual = rand() % (k_new) + 1;
                explored_duals.push_back(explored_duals[kNNSet[idx_dual]]);// still randomly add an existed dual
            }
            else {
                // second stage subproblem is feasible
                std::cout << "Computation Log: Subproblem " << idx_bacth_datapoint << " in iteration " << iteration << " is feasible.\n";
                writeFile << "Computation Log: Subproblem " << idx_bacth_datapoint << " in iteration " << iteration << " is feasible.\n";
                /*
                // check if the new dual is found
                bool flag_new_dual_explored = if_new_dual(explored_duals, dualsTemp);
                if (flag_new_dual_explored == true) {
                    std::cout << "Computation Log: New dual is found.\n";
                    writeFile << "Computation Log: New dual is found.\n";
                    explored_duals.push_back(dualsTemp);
                }*/
                if (if_new_dual(explored_duals, dualsTemp)) { // if new dual is found
                    explored_duals.push_back(dualsTemp);
                }
            }
            std::cout << "Number of unique duals: " << explored_duals.size() << std::endl;
            writeFile << "Number of unique duals: " << explored_duals.size() << std::endl;
            //std::cout << "===(END) DUAL SPACE EXPLORATION===\n";
            //writeFile << "===(END) DUAL SPACE EXPLORATION===\n";
            // MINORANT UPDATES
            // update old minorants
            //std::cout << "Update old active minorants.\n";
            //writeFile << "Update old active minorants.\n";
            if (k == k_new) { // will have more advanced version to store the radius of kNN set
                for (int minorant_index = 0; minorant_index < active_minorant_collection_new.size(); ++minorant_index) {
                    minorant minorant_new;
                    active_minorant_collection_new[minorant_index].alpha = active_minorant_collection_new[minorant_index].alpha +  (f_lowerbound - f_upperbound) / ((double) k);
                    //minorant_new.beta = minorant_collection[active_minorants_new[minorant_index]].beta;
                    //minorant_collection[active_minorants_new[minorant_index]].alpha -= f_upperbound / ((double) k);
                    //minorant_collection_new.push_back(minorant_new);
                }
            }
            else {
                for (int minorant_index = 0; minorant_index < active_minorant_collection_new.size(); ++minorant_index) {
                    minorant minorant_new;
                    active_minorant_collection_new[minorant_index].alpha = active_minorant_collection_new[minorant_index].alpha * ((double) k) / ((double) k_new) + ((double) (k_new - k)) / ((double) k_new) * f_lowerbound;
                    //minorant_collection[active_minorants_new[minorant_index]].alpha = minorant_collection[active_minorants_new[minorant_index]].alpha * ((double) k) / ((double) k_new);
                    active_minorant_collection_new[minorant_index].beta = ((double) k) / ((double) k_new) * active_minorant_collection_new[minorant_index].beta;
                }
            } // end minorant update
            // update k
            k = k_new;
        }
        // update the kNN set after batch sampling
        kNNSet.clear(); // clear the old kNN set
        for (int index = 0; index < orderSet.size(); ++index) {
            if (orderSet[index] <= k_new) {
                kNNSet.push_back(index);
            }
        }
        std::cout << "End batch sampling\n";
        writeFile << "End batch sampling\n";
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
        for (int index = 0; index < k_new; ++index) {
            double max_value = -99999; // NOTE: need to make it smaller
            int max_index = -1;
            int max_index_incumbent = -1;
            double alpha_candidate = 0;
            double alpha_incumbent = 0;
            std::vector<double> beta_candidate;
            std::vector<double> beta_incumbent;
            // incumbent
            double max_value_incumbent = -99999; // NOTE: need to make it smaller
            for (int dual_index = 0; dual_index < explored_duals.size(); ++dual_index) {
                // find optimal dual based on the given set of unique duals
                double current_value = 0;
                // deterministic e
                double pi_e = model_parameters.e * explored_duals[dual_index].equality; // store somewhere for improevement (IMPORTANT)
                // stochastic e
                // equality part
                for (int idx_eq = 0; idx_eq < RHS_dataset[kNNSet[index]].be.size(); ++idx_eq) {
                    pi_e += explored_duals[dual_index].equality[idx_eq] * RHS_dataset[kNNSet[index]].be[idx_eq];
                }
                // inequality part (before standardizing) inequality constraint is after the equality constraints
                for (int idx_ineq = 0; idx_ineq < RHS_dataset[kNNSet[index]].bi.size(); ++idx_ineq) {
                    pi_e += explored_duals[dual_index].equality[RHSmap.bi_map[idx_ineq] + model_parameters.num_eq] * RHS_dataset[kNNSet[index]].bi[idx_ineq];
                }
                current_value += pi_e;
                // determinitsic C
                std::vector<double> pi_C = explored_duals[dual_index].equality * model_parameters.C; // store somewhere else for improvement (IMPORTANT)
                // stochastic C
                // equality
                for (int idx_Ce = 0; idx_Ce < RHS_dataset[kNNSet[index]].Ce.size(); ++idx_Ce) {
                    pi_C[RHSmap.Ce_map[idx_Ce].second] += -1.0 * RHS_dataset[kNNSet[index]].Ce[idx_Ce] * explored_duals[dual_index].equality[RHSmap.Ce_map[idx_Ce].first];
                }
                // inequality before standardizing
                for (int idx_Ci = 0; idx_Ci < RHS_dataset[kNNSet[index]].Ci.size(); ++idx_Ci) {
                    pi_C[RHSmap.Ce_map[idx_Ci].second] += -1.0 * RHS_dataset[kNNSet[index]].Ci[idx_Ci] * explored_duals[dual_index].equality[RHSmap.Ci_map[idx_Ci].first + model_parameters.num_eq];
                }
                current_value += (-1.0) * (pi_C * x_candidate);
                double current_value_incumbent = 0;
                current_value_incumbent += pi_e;
                current_value_incumbent += (-1.0) * (pi_C * x_incumbent);
                if (dual_index < 1) {
                    max_index = dual_index;
                    max_value = current_value;
                    max_index_incumbent = dual_index;
                    max_value_incumbent = current_value_incumbent;
                    // store the intercept and slope
                    alpha_candidate = pi_e; // \pi^\top e
                    alpha_incumbent = pi_e;
                    beta_candidate = (-1.0) * pi_C; // -\pi^\top C
                    beta_incumbent = (-1.0) * pi_C;
                }
                else {
                    if (max_value < current_value) { // find the better dual for given candidate
                        max_index = dual_index;
                        max_value = current_value;
                        alpha_candidate = pi_e;
                        beta_candidate = (-1.0) * pi_C;
                    }
                    if (max_value_incumbent < current_value_incumbent) { // find the better dual for given incumbent
                        max_index_incumbent = dual_index;
                        max_value_incumbent = current_value_incumbent;
                        alpha_incumbent = pi_e;
                        beta_incumbent = (-1.0) * pi_C;
                    }
                }
            }
            // minorant on the candidate
            minorant_candidate.alpha += (1.0 / (double) k_new) * alpha_candidate;
            minorant_candidate.beta = minorant_candidate.beta + (1.0 / (double) k_new) * beta_candidate;
            // minorant on the incumbent
            minorant_incumbent.alpha += (1.0 / (double) k_new) * alpha_incumbent;
            minorant_incumbent.beta = minorant_incumbent.beta + (1.0 / (double) k_new) * beta_incumbent;
        }
        active_minorant_collection_new.push_back(minorant_candidate);
        active_minorant_collection_new.push_back(minorant_incumbent);
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
        bool flag_incumbent_selection = incumbent_selection_check_v2(q, x_candidate, x_incumbent, model_parameters.c, active_minorant_collection, active_minorant_collection_new);
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
        active_minorant_collection.clear();
        for (int idx_minorant = 0; idx_minorant < active_minorant_collection_new.size(); ++idx_minorant) {
            active_minorant_collection.push_back(active_minorant_collection_new[idx_minorant]);
        }
        //std::cout << "===(END) INCUMBENT SELECTION===\n";
        //writeFile << "===(END) INCUMBENT SELECTION===\n";
    } // end of main loop
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
    writeFile << "*******************************************\n";
    
    writeFile.close();
    return x_incumbent;
}


// built-in batch, further refine the minorants, extra memory to store pi*C_det and pi*e_det
std::vector<double> dynamic_sdknn_solver_v3(const std::string& folder_path, int max_iterations, int batch_size,double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug) {
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
    bool flag_be; // tell if be stochastic is generated
    bool flag_bi; // tell if bi stochastic is generated
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
    std::string resultsOutput_path = folder_path + "/computationalResults(NSDv3.0).txt";
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
    standardTwoStageParameters model_parameters;
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
    // read model file
    model_parameters = readStandardTwoStageParameters(model_path);
    // read sto file
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
    std::cout << "SD-kNN (fast version with presolve and batch sampling v3.0) is initialized\n";
    writeFile << "SD-kNN (fast version with presolve and batch sampling v3.0) is initialized\n";
    std::cout << "Algorithmic Parameters\n";
    writeFile << "Algorithmic Parameters\n";
    std::cout << "sigma, q, beta, k, N, N_pre, sigma_lower, sigma_upper" << std::endl;
    writeFile << "sigma, q, beta, k, N, N_pre, sigma_lower, sigma_upper" << std::endl;
    std::cout << sigma << ", " << q << ", " << beta << ", " << k << ", " << N << ", " << N_pre << ", " << sigma_lowerbound << ", " << sigma_upperbound << std::endl;
    writeFile << sigma << ", " << q << ", " << beta << ", " << k << ", " << N << ", " << N_pre << ", " << sigma_lowerbound << ", " << sigma_upperbound << std::endl;
    std::cout << "batch size" << std::endl;
    writeFile << "batch size" << std::endl;
    std::cout << batch_size << std::endl;
    writeFile << batch_size << std::endl;
    std::cout << "Problem Complexity\n";
    writeFile << "Problem Complexity\n";
    std::cout << "A_num_row, A_num_col\n";
    writeFile << "A_num_row, A_num_col\n";
    std::cout << model_parameters.A.num_row << ", " << model_parameters.A.num_col << std::endl;
    writeFile << model_parameters.A.num_row << ", " << model_parameters.A.num_col << std::endl;
    std::cout << "D_num_row, D_num_col (after converting into standard form)\n";
    writeFile << "D_num_row, D_num_col (after converting into standard form)\n";
    std::cout << model_parameters.D.num_row << ", " << model_parameters.D.num_col << std::endl;
    writeFile << model_parameters.D.num_row << ", " << model_parameters.D.num_col << std::endl;
    // set up initial incumbent solution
    long x_size = model_parameters.c.num_entry;
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
        }
    }
    k_new = (int) pow(N_pre, beta);
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
            }
            std::cout << std::endl;
            writeFile << std::endl;
        }
    }
    // update the kNN set
    kNNSet.clear(); // clear the old kNN set
    for (int index = 0; index < orderSet.size(); ++index) {
        if (orderSet[index] <= k_new) {
            kNNSet.push_back(index);
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
    // presolve problem to get x_incumbent
    x_incumbent = twoStageLP_presolve(model_parameters, knn_point_estimate, RHSmap);
    std::cout << "Incumbent solution after presolve:\n";
    writeFile << "Incumbent solution after presolve:\n";
    for (int idx_x = 0; idx_x < x_incumbent.size() - 1; ++idx_x) {
        std::cout << x_incumbent[idx_x] << ", ";
        writeFile << x_incumbent[idx_x] << ", ";
    }
    std::cout << x_incumbent[x_incumbent.size() - 1] << std::endl;
    writeFile << x_incumbent[x_incumbent.size() - 1] << std::endl;
    // initialize explored dual multipliers in the second stage
    std::vector<dualMultipliers> explored_duals;
    std::vector<double> pi_e_collection; // pi times e(deterministic part)
    std::vector<std::vector<double>> pi_C_collection; // pi times C(deterministic part)
    // obtain duals at the presolve points
    for (int idx_knn = 0; idx_knn < kNNSet.size(); ++idx_knn) {
        dualMultipliers new_dual = twoStageLP_secondStageDual(x_incumbent, model_parameters, RHS_dataset[kNNSet[idx_knn]], RHSmap);
        if (new_dual.feasible_flag == false) { // if second stage subproblem is infeasible
            //
            std::cout << "Warning: An infeasible case is captured.\n";
            writeFile << "Warning: An infeasible case is captured.\n";
            std::cout << "Warning: A feasibility cut will be constructed.\n";
            writeFile << "Warning: A feasibility cut will be constructed.\n";
            dualMultipliers extremeRay_scenario = twoStageLP_secondStageExtremRay(x_incumbent, model_parameters, RHS_dataset[kNNSet[idx_knn]], RHSmap);
            feasibilityCut feasibilityCut_scenario = twoStageLP_feasibilityCutGeneration(extremeRay_scenario, model_parameters, RHS_dataset[kNNSet[idx_knn]], RHSmap);
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
            if (if_new_dual(explored_duals, new_dual)) { // if new dual is found
                explored_duals.push_back(new_dual); // store the new dual
                double pi_e = model_parameters.e * new_dual.equality; // store new dual times e
                pi_e_collection.push_back(pi_e);
                // determinitsic C
                std::vector<double> pi_C = new_dual.equality * model_parameters.C;
                pi_C_collection.push_back(pi_C);
            }
        }
    }
    //
    std::cout << "Number of unique duals explored in presolve process: " << explored_duals.size() << std::endl;
    writeFile << "Number of unique duals explored in presolve process: " << explored_duals.size() << std::endl;
    // initialize a collection of minorants
    std::vector<minorant> minorant_collection;
    std::vector<minorant> active_minorant_collection;
    // construct initial minorant
    std::cout << "Construct initial minorant.\n";
    writeFile << "Construct initial minorant.\n";
    minorant initial_minorant;
    initial_minorant.alpha = f_lowerbound; // should use lower bound for the intercept of the initial minorant
    for (int idx_x = 0; idx_x < x_size; ++idx_x) {
        initial_minorant.beta.push_back(0);
    }
    minorant_collection.push_back(initial_minorant);
    active_minorant_collection.push_back(initial_minorant);
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
    // initialize the index for the datapoint
    int idx_datapoint = N_pre - 1;
    N = N_pre; // update number of data points collected
    k = k_new;
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        std::cout << "***Iteration " << iteration << "***\n";
        writeFile << "***Iteration " << iteration << "***\n";
        std::cout << "sigma: " << sigma << std::endl;
        writeFile << "sigma: " << sigma << std::endl;
        std::vector<double> x_candidate;
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
        for (auto it = model_parameters.c.vec.begin(); it != model_parameters.c.vec.end(); ++it) {
            expr_obj += (it -> second) * x_temp[it -> first];
        }
        for (int x_index = 0; x_index < A_colsize; ++x_index) {
            expr_obj += 0.5 * sigma * (x_temp[x_index] * x_temp[x_index] - 2.0 * x_temp[x_index] * x_incumbent[x_index] + x_incumbent[x_index] * x_incumbent[x_index]);
        }
        expr_obj += eta;
        IloObjective obj = IloMinimize(env,expr_obj); // objective function
        mod.add(obj);
        // constraints
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
        std::cout << "Number of minorants used in promxial mapping: " << active_minorant_collection.size() << std::endl;
        writeFile << "Number of minorants used in promxial mapping: " << active_minorant_collection.size() << std::endl;
        for (int index_cons = 0; index_cons < active_minorant_collection.size(); ++index_cons) {
            IloExpr expr(env);
            expr += active_minorant_collection[index_cons].alpha - eta;
            for (int index_x = 0; index_x < x_size; ++index_x ) {
                expr += active_minorant_collection[index_cons].beta[index_x] * x_temp[index_x];
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
        std::vector<minorant> active_minorant_collection_new;
        //std::vector<minorant> active_minorant_collection_temp;
        int num_active_minorants = 0;
        for (int index = 0; index < active_minorant_collection.size(); ++index) {
            if (duals[index] < SOLVER_PRECISION_LOWER || duals[index] > SOLVER_PRECISION_UPPER) { // only store the active minorants whose duals are significantly different from 0
                //std::cout << "dual: " << duals[index] << std::endl;
                active_minorant_collection_new.push_back(active_minorant_collection[index]);
                //active_minorant_collection_temp.push_back(active_minorant_collection[index]);
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
        // kNN in after adding a batch of samples
        std::cout << "Start batch sampling\n";
        writeFile << "Start batch sampling\n";
        for (int idx_bacth_datapoint = 0; idx_bacth_datapoint < batch_size; ++idx_bacth_datapoint) {
            N += 1; // increase sample size
            idx_datapoint += 1; // go to the next data point
            k_new = (int) pow(N, beta); // calculate new k
            std::cout << "N: " << N << std::endl;
            writeFile << "N: " << N << std::endl;
            std::cout << "k (number of nearest neighbor): " << k_new << std::endl;
            writeFile << "k (number of nearest neighbor): " << k_new << std::endl;
            std::cout << "k old: " << k << std::endl;// debug
            writeFile << "k old: " << k << std::endl;
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
                            kNNSet_new.push_back(index);
                        }
                        std::cout << std::endl;
                        writeFile << std::endl;
                    }
                }
                else {
                    // update the kNN set
                    for (int index = 0; index < orderSet.size(); ++index) {
                        if (orderSet[index] <= k_new) {
                            kNNSet_new.push_back(index);
                        }
                    }
                }
                 */
            }
            //*********************
            //end non-parametric estimation (kNN)
            //std::cout << "===(END) kNN ESTIMATION===\n";
            //writeFile << "===(END) kNN ESTIMATION===\n";
            // DUAL SPACE EXPLORATION
            //std::cout << "===DUAL SPACE EXPLORATION===\n";
            //writeFile << "===DUAL SPACE EXPLORATION===\n";
            // calculate the dual multipliers
            dualMultipliers dualsTemp = twoStageLP_secondStageDual(x_candidate, model_parameters, RHS_datapoint, RHSmap);
            if (dualsTemp.feasible_flag == false) { // if second stage subproblem is infeasible
                //
                std::cout << "Warning: An infeasible case is captured in iteration " << iteration << ".\n";
                writeFile << "Warning: An infeasible case is captured in iteration " << iteration << ".\n";
                std::cout << "Warning: A feasibility cut will be constructed.\n";
                writeFile << "Warning: A feasibility cut will be constructed.\n";
                dualMultipliers extremeRay_scenario = twoStageLP_secondStageExtremRay(x_candidate, model_parameters, RHS_datapoint, RHSmap);
                feasibilityCut feasibilityCut_scenario = twoStageLP_feasibilityCutGeneration(extremeRay_scenario, model_parameters, RHS_datapoint, RHSmap);
                // add feasibility cut
                feasibility_cuts.push_back(feasibilityCut_scenario);
                //int idx_dual = rand() % (k_new) + 1;
                //explored_duals.push_back(explored_duals[kNNSet[idx_dual]]);// still randomly add an existed dual
            }
            else {
                // second stage subproblem is feasible
                std::cout << "Computation Log: Subproblem " << idx_bacth_datapoint << " in iteration " << iteration << " is feasible.\n";
                writeFile << "Computation Log: Subproblem " << idx_bacth_datapoint << " in iteration " << iteration << " is feasible.\n";
                /*
                // check if the new dual is found
                bool flag_new_dual_explored = if_new_dual(explored_duals, dualsTemp);
                if (flag_new_dual_explored == true) {
                    std::cout << "Computation Log: New dual is found.\n";
                    writeFile << "Computation Log: New dual is found.\n";
                    explored_duals.push_back(dualsTemp);
                }*/
                if (if_new_dual(explored_duals, dualsTemp)) { // if new dual is found
                    explored_duals.push_back(dualsTemp);
                    double pi_e = model_parameters.e * dualsTemp.equality; // store new dual times e
                    pi_e_collection.push_back(pi_e);
                    // determinitsic C
                    std::vector<double> pi_C = dualsTemp.equality * model_parameters.C;
                    pi_C_collection.push_back(pi_C);
                }
            }
            std::cout << "Number of unique duals: " << explored_duals.size() << std::endl;
            writeFile << "Number of unique duals: " << explored_duals.size() << std::endl;
            //std::cout << "===(END) DUAL SPACE EXPLORATION===\n";
            //writeFile << "===(END) DUAL SPACE EXPLORATION===\n";
            // MINORANT UPDATES
            // update old minorants
            //std::cout << "Update old active minorants.\n";
            //writeFile << "Update old active minorants.\n";
            if (k == k_new) { // will have more advanced version to store the radius of kNN set
                for (int minorant_index = 0; minorant_index < active_minorant_collection_new.size(); ++minorant_index) {
                    minorant minorant_new;
                    active_minorant_collection_new[minorant_index].alpha = active_minorant_collection_new[minorant_index].alpha +  (f_lowerbound - f_upperbound) / ((double) k);
                    //minorant_new.beta = minorant_collection[active_minorants_new[minorant_index]].beta;
                    //minorant_collection[active_minorants_new[minorant_index]].alpha -= f_upperbound / ((double) k);
                    //minorant_collection_new.push_back(minorant_new);
                }
            }
            else {
                for (int minorant_index = 0; minorant_index < active_minorant_collection_new.size(); ++minorant_index) {
                    minorant minorant_new;
                    active_minorant_collection_new[minorant_index].alpha = active_minorant_collection_new[minorant_index].alpha * ((double) k) / ((double) k_new) + ((double) (k_new - k)) / ((double) k_new) * f_lowerbound;
                    //minorant_collection[active_minorants_new[minorant_index]].alpha = minorant_collection[active_minorants_new[minorant_index]].alpha * ((double) k) / ((double) k_new);
                    active_minorant_collection_new[minorant_index].beta = ((double) k) / ((double) k_new) * active_minorant_collection_new[minorant_index].beta;
                }
            } // end minorant update
            // update k
            k = k_new;
        }
        // update the kNN set after batch sampling
        kNNSet.clear(); // clear the old kNN set
        for (int index = 0; index < orderSet.size(); ++index) {
            if (orderSet[index] <= k_new) {
                kNNSet.push_back(index);
            }
        }
        std::cout << "End batch sampling\n";
        writeFile << "End batch sampling\n";
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
        for (int index = 0; index < k_new; ++index) {
            double max_value = -99999; // NOTE: need to make it smaller
            int max_index = -1;
            int max_index_incumbent = -1;
            double alpha_candidate = 0;
            double alpha_incumbent = 0;
            std::vector<double> beta_candidate;
            std::vector<double> beta_incumbent;
            // incumbent
            double max_value_incumbent = -99999; // NOTE: need to make it smaller
            for (int dual_index = 0; dual_index < explored_duals.size(); ++dual_index) {
                // find optimal dual based on the given set of unique duals
                double current_value = 0;
                // deterministic e
                //double pi_e = model_parameters.e * explored_duals[dual_index].equality; // store somewhere for improevement (IMPORTANT)
                double pi_e = pi_e_collection[dual_index]; // deterministic pi * e
                // stochastic e
                // equality part
                for (int idx_eq = 0; idx_eq < RHS_dataset[kNNSet[index]].be.size(); ++idx_eq) {
                    pi_e += explored_duals[dual_index].equality[idx_eq] * RHS_dataset[kNNSet[index]].be[idx_eq];
                }
                // inequality part (before standardizing) inequality constraint is after the equality constraints
                for (int idx_ineq = 0; idx_ineq < RHS_dataset[kNNSet[index]].bi.size(); ++idx_ineq) {
                    pi_e += explored_duals[dual_index].equality[RHSmap.bi_map[idx_ineq] + model_parameters.num_eq] * RHS_dataset[kNNSet[index]].bi[idx_ineq];
                }
                current_value += pi_e;
                // determinitsic C
                //std::vector<double> pi_C = explored_duals[dual_index].equality * model_parameters.C; // store somewhere else for improvement (IMPORTANT)
                std::vector<double> pi_C = pi_C_collection[dual_index]; // deterministic pi * C
                // stochastic C
                // equality
                for (int idx_Ce = 0; idx_Ce < RHS_dataset[kNNSet[index]].Ce.size(); ++idx_Ce) {
                    pi_C[RHSmap.Ce_map[idx_Ce].second] += -1.0 * RHS_dataset[kNNSet[index]].Ce[idx_Ce] * explored_duals[dual_index].equality[RHSmap.Ce_map[idx_Ce].first];
                }
                // inequality before standardizing
                for (int idx_Ci = 0; idx_Ci < RHS_dataset[kNNSet[index]].Ci.size(); ++idx_Ci) {
                    pi_C[RHSmap.Ce_map[idx_Ci].second] += -1.0 * RHS_dataset[kNNSet[index]].Ci[idx_Ci] * explored_duals[dual_index].equality[RHSmap.Ci_map[idx_Ci].first + model_parameters.num_eq];
                }
                current_value += (-1.0) * (pi_C * x_candidate);
                double current_value_incumbent = 0;
                current_value_incumbent += pi_e;
                current_value_incumbent += (-1.0) * (pi_C * x_incumbent);
                if (dual_index < 1) {
                    max_index = dual_index;
                    max_value = current_value;
                    max_index_incumbent = dual_index;
                    max_value_incumbent = current_value_incumbent;
                    // store the intercept and slope
                    alpha_candidate = pi_e; // \pi^\top e
                    alpha_incumbent = pi_e;
                    beta_candidate = (-1.0) * pi_C; // -\pi^\top C
                    beta_incumbent = (-1.0) * pi_C;
                }
                else {
                    if (max_value < current_value) { // find the better dual for given candidate
                        max_index = dual_index;
                        max_value = current_value;
                        alpha_candidate = pi_e;
                        beta_candidate = (-1.0) * pi_C;
                    }
                    if (max_value_incumbent < current_value_incumbent) { // find the better dual for given incumbent
                        max_index_incumbent = dual_index;
                        max_value_incumbent = current_value_incumbent;
                        alpha_incumbent = pi_e;
                        beta_incumbent = (-1.0) * pi_C;
                    }
                }
            }
            // minorant on the candidate
            minorant_candidate.alpha += (1.0 / (double) k_new) * alpha_candidate;
            minorant_candidate.beta = minorant_candidate.beta + (1.0 / (double) k_new) * beta_candidate;
            // minorant on the incumbent
            minorant_incumbent.alpha += (1.0 / (double) k_new) * alpha_incumbent;
            minorant_incumbent.beta = minorant_incumbent.beta + (1.0 / (double) k_new) * beta_incumbent;
        }
        active_minorant_collection_new.push_back(minorant_candidate);
        active_minorant_collection_new.push_back(minorant_incumbent);
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
        bool flag_incumbent_selection = incumbent_selection_check_v2(q, x_candidate, x_incumbent, model_parameters.c, active_minorant_collection, active_minorant_collection_new);
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
        active_minorant_collection.clear();
        for (int idx_minorant = 0; idx_minorant < active_minorant_collection_new.size(); ++idx_minorant) {
            active_minorant_collection.push_back(active_minorant_collection_new[idx_minorant]);
        }
        //std::cout << "===(END) INCUMBENT SELECTION===\n";
        //writeFile << "===(END) INCUMBENT SELECTION===\n";
    } // end of main loop
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
    writeFile << "*******************************************\n";
    
    writeFile.close();
    return x_incumbent;
}
