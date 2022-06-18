//
//  NSD_utils.cpp
//  NSDv2.0
//
//  Created by Shuotao Diao on 8/5/21.
//

#include "NSD_utils.hpp"

double twoStageLP_secondStageCost(const std::vector<double>& x, standardTwoStageParameters& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap) {
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
    // cost in the second stage problem
    double second_stage_cost = cplex.getObjValue();
    env.end();
    return second_stage_cost;
}


validationResult twoStageLP_validation_outputResultsV2(const std::string& folder_path, const std::vector<double>& x_candidate) {
    // STEP 1: INITIALIZATION
    bool flag_be;
    bool flag_bi;
    bool flag_Ce;
    bool flag_Ci;
    // create directory paths for database and model
    std::string be_DB_path = folder_path + "/be_DB.txt";
    std::string bi_DB_path = folder_path + "/bi_DB.txt";
    std::string Ce_DB_path = folder_path + "/Ce_DB.txt";
    std::string Ci_DB_path = folder_path + "/Ci_DB.txt";
    std::string model_path = folder_path + "/model.txt";
    std::string sto_path = folder_path + "/sto.txt";
    // convert all the paths into constant chars
    const char* be_DB_path_const = be_DB_path.c_str();
    const char* bi_DB_path_const = bi_DB_path.c_str();
    const char* Ce_DB_path_const = Ce_DB_path.c_str();
    const char* Ci_DB_path_const = Ci_DB_path.c_str();
    const char* model_path_const = model_path.c_str();
    const char* sto_path_const = sto_path.c_str();
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
    long sample_size = 0;
    if (flag_be == true) {
        sample_size = be_DB[0].size();
    }
    else if (flag_bi == true) {
        sample_size = bi_DB[0].size();
    }
    else if (flag_Ce == true) {
        sample_size = Ce_DB[0].size();
    }
    else if (flag_Ci == true) {
        sample_size = Ci_DB[0].size();
    }
    else {
        throw std::invalid_argument("ERROR: Database is empty!\n");
    }
    // determine appropriate model
    double secondStageTotalCost = 0;
    double firstStageCost = model_parameters.c * x_candidate;
    double variance_P = 0; // intermediate component for calculating variance
    for (int idx_scenario = 0; idx_scenario < sample_size; ++idx_scenario) {
        // obtain a new data point
        dataPoint be_datapoint;
        if (flag_be == true) {
            be_datapoint = be_DB[0][idx_scenario];
        }
        dataPoint bi_datapoint;
        if (flag_bi == true) {
            bi_datapoint = bi_DB[0][idx_scenario];
        }
        dataPoint Ce_datapoint;
        if (flag_Ce == true) {
            Ce_datapoint = Ce_DB[0][idx_scenario];
        }
        dataPoint Ci_datapoint;
        if (flag_Ci == true) {
            Ci_datapoint = Ci_DB[0][idx_scenario];
        }
        // merge all the datapoints
        secondStageRHSpoint RHS_datapoint = merge_randomVector(be_datapoint, bi_datapoint, Ce_datapoint, Ci_datapoint);
        double tempCost = twoStageLP_secondStageCost(x_candidate, model_parameters, RHS_datapoint, RHSmap);
        secondStageTotalCost += tempCost / ((double) sample_size);
        double tempTotalCost = tempCost + firstStageCost;
        double _n_ = idx_scenario + 1;
        variance_P = variance_P * (_n_ - 1) / _n_ + tempTotalCost * tempTotalCost / _n_;
    }
    double _n_ = sample_size;
    validationResult result;
    result.mean = firstStageCost + secondStageTotalCost;
    result.variance = (variance_P - result.mean * result.mean) * _n_ / (_n_ - 1);
    result.num_dataPoint = sample_size;
    double halfMargin = result.Zalpha * sqrt(result.variance / _n_);
    result.CI_lower = result.mean - halfMargin;
    result.CI_upper = result.mean + halfMargin;
    // write computational results
    std::string outputResults_path = folder_path + "/validationResultsV2.txt";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    // current time
    std::time_t currTime = std::time(nullptr);
    writeFile << "***************************************************\n";
    writeFile << "Estimating the quality of candidate solution\n";
    writeFile << std::put_time(localtime(&currTime), "%c %Z") << "\n"; // write current time
    writeFile << "Candidate solution             : ";
    long x_size = x_candidate.size();
    for (int index = 0; index < x_size - 1; ++index) {
        writeFile << x_candidate[index] << ", ";
    }
    writeFile << x_candidate[x_size - 1] << "\n";
    writeFile << "Number of data points          : " << result.num_dataPoint << "\n";
    std::cout << "Average validation cost        : " << result.mean << "\n";
    writeFile << "Average validation cost        : " << result.mean << "\n";
    writeFile << "Variance                       : " << result.variance << "\n";
    writeFile << "Variance in estimating the mean: " << sqrt(result.variance/ _n_) << "\n";
    writeFile << result.alpha << "% confidence interval of expected cost: [" << result.CI_lower << ", " << result.CI_upper << "]\n";
    writeFile << "***************************************************\n";
    return result;
}

// solve the SAA problem to get the baseline solution and cost
void twoStageLP_empirical_cost(const std::string& folder_path) {
    // STEP 1: INITIALIZATION
    bool flag_be;
    bool flag_bi;
    bool flag_Ce;
    bool flag_Ci;
    // create directory paths for database and model
    std::string be_DB_path = folder_path + "/be_DB.txt";
    std::string bi_DB_path = folder_path + "/bi_DB.txt";
    std::string Ce_DB_path = folder_path + "/Ce_DB.txt";
    std::string Ci_DB_path = folder_path + "/Ci_DB.txt";
    std::string model_path = folder_path + "/model.txt";
    std::string sto_path = folder_path + "/sto.txt";
    // convert all the paths into constant chars
    const char* be_DB_path_const = be_DB_path.c_str();
    const char* bi_DB_path_const = bi_DB_path.c_str();
    const char* Ce_DB_path_const = Ce_DB_path.c_str();
    const char* Ci_DB_path_const = Ci_DB_path.c_str();
    const char* model_path_const = model_path.c_str();
    const char* sto_path_const = sto_path.c_str();
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
    long sample_size = 0;
    if (flag_be == true) {
        sample_size = be_DB[0].size();
    }
    else if (flag_bi == true) {
        sample_size = bi_DB[0].size();
    }
    else if (flag_Ce == true) {
        sample_size = Ce_DB[0].size();
    }
    else if (flag_Ci == true) {
        sample_size = Ci_DB[0].size();
    }
    else {
        throw std::invalid_argument("ERROR: Database is empty!\n");
    }
    // determine appropriate model
    // solve a large LP
    IloEnv env;
    IloModel mod(env);
    IloNumVarArray x(env,model_parameters.c.num_entry,-IloInfinity,IloInfinity,ILOFLOAT);
    mod.add(x);
    std::vector<IloNumVarArray> y;
    for (int idx_scenario = 0; idx_scenario < sample_size; ++idx_scenario) {
        IloNumVarArray y_scenario(env,model_parameters.D.num_col,0,IloInfinity,ILOFLOAT); // y >= 0
        y.push_back(y_scenario);
        mod.add(y[idx_scenario]);
    }
    IloExpr expr_obj(env);
    // first stage objective linear
    for (auto it = model_parameters.c.vec.begin(); it != model_parameters.c.vec.end(); ++it) {
        expr_obj += (it -> second) * x[it -> first];
    }
    for (int idx_scenario = 0; idx_scenario < sample_size; ++idx_scenario) {
        // second stage objective linear
        for (auto it = model_parameters.d.vec.begin(); it != model_parameters.d.vec.end(); ++it) {
            expr_obj += (1.0 / (double) sample_size) * (it -> second) * y[idx_scenario][it -> first];
        }
    }
    // add objective
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
        exprs[(it -> first).first] += (it -> second) * x[(it -> first).second];
    }
    // right hand side
    for (auto it = model_parameters.b.vec.begin(); it != model_parameters.b.vec.end(); ++it) {
        exprs[it -> first] -= (it -> second);
    }
    // add constraints
    for (int index_cons = 0; index_cons < model_parameters.A.num_row; ++index_cons) {
        mod.add(exprs[index_cons] <= 0);
    }
    for (int idx_scenario = 0; idx_scenario < sample_size; ++idx_scenario) {
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
            exprs_eq[(it -> first).first] += (it -> second) * y[idx_scenario][(it -> first).second];
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
        // obtain a new data point
        dataPoint be_datapoint;
        if (flag_be == true) {
            be_datapoint = be_DB[0][idx_scenario];
        }
        dataPoint bi_datapoint;
        if (flag_bi == true) {
            bi_datapoint = bi_DB[0][idx_scenario];
        }
        dataPoint Ce_datapoint;
        if (flag_Ce == true) {
            Ce_datapoint = Ce_DB[0][idx_scenario];
        }
        dataPoint Ci_datapoint;
        if (flag_Ci == true) {
            Ci_datapoint = Ci_DB[0][idx_scenario];
        }
        // merge all the datapoints
        secondStageRHSpoint RHS_datapoint = merge_randomVector(be_datapoint, bi_datapoint, Ce_datapoint, Ci_datapoint);
        // coefficients before x (stochastic part) equality (i.e., Cij * xj map: <i,j> )
        for (int idx_Ce = 0; idx_Ce < RHS_datapoint.Ce.size(); ++idx_Ce) {
            exprs_eq[RHSmap.Ce_map[idx_Ce].first] += RHS_datapoint.Ce[idx_Ce] * x[RHSmap.Ce_map[idx_Ce].second];
        }
        // coefficients before x (stochastic part) inequality (location is behind equality constraints)
        for (int idx_Ci = 0; idx_Ci < RHS_datapoint.Ci.size(); ++idx_Ci) {
            exprs_eq[RHSmap.Ci_map[idx_Ci].first + model_parameters.num_eq] += RHS_datapoint.Ci[idx_Ci] * x[RHSmap.Ci_map[idx_Ci].second];
        }
        // right hand side (stochastic part) equality be_(i) equality
        for (int idx_be = 0; idx_be < RHS_datapoint.be.size(); ++idx_be) {
            exprs_eq[RHSmap.be_map[idx_be]] -= RHS_datapoint.be[idx_be];
        }
        // right hand side (stochastic part) equality bi_(i) inequality
        for (int idx_bi = 0; idx_bi < RHS_datapoint.bi.size(); ++idx_bi) {
            exprs_eq[RHSmap.bi_map[idx_bi] + model_parameters.num_eq] -= RHS_datapoint.bi[idx_bi];
        }
        // add the equality constraints
        for (int index_eq = 0; index_eq < model_parameters.D.num_row; ++index_eq) {
            constraintsEquality.add(exprs_eq[index_eq] == 0);
        }
        mod.add(constraintsEquality);
    }
    // create cplex environment
    IloCplex cplex(env);
    cplex.extract(mod);
    cplex.solve();
    // write computational results
    std::string outputResults_path = folder_path + "/baseline_slp.txt";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    // current time
    std::time_t currTime = std::time(nullptr);
    writeFile << "***************************************************\n";
    writeFile << "Input Folder: " << folder_path << std::endl;
    writeFile << std::put_time(localtime(&currTime), "%c %Z") << "\n"; // write current time
    writeFile << "Baseline Solution             : ";
    long x_size = model_parameters.c.num_entry;
    for (int index = 0; index < x_size - 1; ++index) {
        writeFile << cplex.getValue(x[index]) << ", ";
    }
    writeFile << cplex.getValue(x[x_size - 1]) << "\n";
    writeFile << "Baseline Cost: " << cplex.getObjValue() << std::endl;
    env.end();
    writeFile.close();
}

void interface_dynamic_nsd_presolve(const std::string& folder_path, const std::string& validation_folder_path, int max_iterations, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug) {
    // obtain estimated solution from NSD solver (fast)
    std::vector<double> x_SDkNN = dynamic_sdknn_solver_presolve_fullDual(folder_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    // set up write file
    std::clock_t time_start;
    time_start = std::clock();
    validationResult res_val = twoStageLP_validation_outputResultsV2(validation_folder_path, x_SDkNN);
    double duration_val = (std::clock() - time_start ) / (double) CLOCKS_PER_SEC;
    // write the results of quality of candidate solution
    std::string outputResults_path = folder_path + "/validationResults(NSDv2.0presolve).txt";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    // current time
    std::time_t currTime = std::time(nullptr);
    std::cout << "**********************************************************************\n";
    writeFile << "**********************************************************************\n";
    std::cout << "Dynamic Stepsize and Presolve(v2) are used\n";
    writeFile << "Dynamic Stepsize and Presolve(v2) are used\n";
    std::cout << "Number of datapoints used in the presolve: " << N_pre << std::endl;
    writeFile << "Number of datapoints used in the presolve: " << N_pre << std::endl;
    writeFile << std::put_time(localtime(&currTime), "%c %Z") << "\n"; // write current time
    std::cout << std::put_time(localtime(&currTime), "%c %Z") << "\n"; // write current time
    std::cout << "Folder Path: " << folder_path << std::endl;
    writeFile << "Folder Path: " << folder_path << std::endl;
    writeFile << "Initial Point        : ";
    std::cout << "Initial Point        : ";
    std::cout << "Observed Predictor: ";
    writeFile << "Observed Predictor: ";
    for (int index = 0; index < observed_predictor.size() - 1; ++index) {
        std::cout << observed_predictor[index] << ", ";
        writeFile << observed_predictor[index] << ", ";
    }
    std::cout << observed_predictor[observed_predictor.size() - 1] << std::endl;
    writeFile << observed_predictor[observed_predictor.size() - 1] << std::endl;
    std::cout << "Number of iterations: " << max_iterations << std::endl;
    writeFile << "Number of iterations: " << max_iterations << std::endl;
    std::cout << "Solution obtained from SD-kNN with dynamic stepsize(fast): ";
    writeFile << "Solution obtained from SD-kNN with dynamic stepsize(fast): ";
    for (int index = 0; index < x_SDkNN.size() - 1; ++index) {
        std::cout << x_SDkNN[index] << ", ";
        writeFile << x_SDkNN[index] << ", ";
    }
    std::cout << x_SDkNN[x_SDkNN.size() - 1] << std::endl;
    writeFile << x_SDkNN[x_SDkNN.size() - 1] << std::endl;
    std::cout << "===Validation Results===\n";
    writeFile << "===Validation Results===\n";
    std::cout << "Validation Folder: " << validation_folder_path << std::endl;
    writeFile << "Validation Folder: " << validation_folder_path << std::endl;
    writeFile << "Number of data points: " << res_val.num_dataPoint << std::endl;
    std::cout << "Number of data points: " << res_val.num_dataPoint << std::endl;
    writeFile << "Average cost         : " << res_val.mean << std::endl;
    std::cout << "Average cost         : " << res_val.mean << std::endl;
    writeFile << "Variance             : " << res_val.variance << std::endl;
    std::cout << "Variance             : " << res_val.variance << std::endl;
    writeFile << "Error in estimating the expected cost: " << sqrt(res_val.variance / ((double)res_val.num_dataPoint)) << std::endl;
    std::cout << "Error in estimating the expected cost: " << sqrt(res_val.variance / ((double)res_val.num_dataPoint)) << std::endl;
    writeFile << res_val.alpha << "% CI Lower Bound: " << res_val.CI_lower << std::endl;
    std::cout << res_val.alpha << "% CI Lower Bound: " << res_val.CI_lower << std::endl;
    writeFile << res_val.alpha << "% CI Upper Bound: " << res_val.CI_upper << std::endl;
    std::cout << res_val.alpha << "% CI Upper Bound: " << res_val.CI_upper << std::endl;
    writeFile << "Time Elapsed(s) for Validation: " << duration_val << std::endl;
    std::cout << "Time Elapsed(s) for Validation: " << duration_val << std::endl;
    std::cout << "**********************************************************************\n";
    writeFile << "**********************************************************************\n";
    // close write file
    writeFile.close();
}
void interface_sdknn(const std::string& folder_path, const std::string& validation_folder_path, int max_iterations, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug){
    // set up start time
    std::clock_t time_start;
    time_start = std::clock();
    // obtain estimated solution from NSD solver (fast)
    std::vector<double> x_SDkNN = dynamic_sdknn_solver_presolve_fullDual_v3(folder_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    double time_elapse = (std::clock() - time_start) / (double) CLOCKS_PER_SEC;
    validationResult res_val = twoStageLP_validation_outputResultsV2(validation_folder_path, x_SDkNN);
    // write the results of quality of candidate solution
    std::string outputResults_path = folder_path + "/nsd_summary.csv";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    int sample_size = N_pre + max_iterations;
    writeFile << max_iterations << ", ";
    writeFile << sample_size << ", ";
    writeFile << res_val.mean << ", ";
    writeFile << time_elapse << std::endl;
}


void interface_sdknn_v2(const std::string& folder_path, const std::string& validation_folder_path, int max_iterations, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug) {
    // set up start time
    std::clock_t time_start;
    time_start = std::clock();
    // obtain estimated solution from NSD solver (fast)
    std::vector<double> x_SDkNN = dynamic_sdknn_solver_v4(folder_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    double time_elapse = (std::clock() - time_start) / (double) CLOCKS_PER_SEC;
    validationResult res_val = twoStageLP_validation_outputResultsV2(validation_folder_path, x_SDkNN);
    // write the results of quality of candidate solution
    std::string outputResults_path = folder_path + "/nsd_summary.csv";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    int sample_size = N_pre + max_iterations;
    writeFile << max_iterations << ", ";
    writeFile << sample_size << ", ";
    writeFile << res_val.mean << ", ";
    writeFile << time_elapse << std::endl;
}


void interface_dynamic_nsd_presolve_v3(const std::string& folder_path, const std::string& validation_folder_path, int max_iterations, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug) {
    // obtain estimated solution from NSD solver (fast)
    std::vector<double> x_SDkNN = dynamic_sdknn_solver_presolve_fullDual_v3(folder_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    // set up write file
    std::clock_t time_start;
    time_start = std::clock();
    validationResult res_val = twoStageLP_validation_outputResultsV2(validation_folder_path, x_SDkNN);
    double duration_val = (std::clock() - time_start ) / (double) CLOCKS_PER_SEC;
    // write the results of quality of candidate solution
    std::string outputResults_path = folder_path + "/validationResults(NSDv3.0basic).txt";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    // current time
    std::time_t currTime = std::time(nullptr);
    std::cout << "**********************************************************************\n";
    writeFile << "**********************************************************************\n";
    std::cout << "Dynamic Stepsize and Presolve(v3) are used\n";
    writeFile << "Dynamic Stepsize and Presolve(v3) are used\n";
    std::cout << "Number of datapoints used in the presolve: " << N_pre << std::endl;
    writeFile << "Number of datapoints used in the presolve: " << N_pre << std::endl;
    writeFile << std::put_time(localtime(&currTime), "%c %Z") << "\n"; // write current time
    std::cout << std::put_time(localtime(&currTime), "%c %Z") << "\n"; // write current time
    std::cout << "Folder Path: " << folder_path << std::endl;
    writeFile << "Folder Path: " << folder_path << std::endl;
    writeFile << "Initial Point        : ";
    std::cout << "Initial Point        : ";
    std::cout << "Observed Predictor: ";
    writeFile << "Observed Predictor: ";
    for (int index = 0; index < observed_predictor.size() - 1; ++index) {
        std::cout << observed_predictor[index] << ", ";
        writeFile << observed_predictor[index] << ", ";
    }
    std::cout << observed_predictor[observed_predictor.size() - 1] << std::endl;
    writeFile << observed_predictor[observed_predictor.size() - 1] << std::endl;
    std::cout << "Number of iterations: " << max_iterations << std::endl;
    writeFile << "Number of iterations: " << max_iterations << std::endl;
    std::cout << "Solution obtained from SD-kNN v3 with dynamic stepsize(fast): ";
    writeFile << "Solution obtained from SD-kNN v3 with dynamic stepsize(fast): ";
    for (int index = 0; index < x_SDkNN.size() - 1; ++index) {
        std::cout << x_SDkNN[index] << ", ";
        writeFile << x_SDkNN[index] << ", ";
    }
    std::cout << x_SDkNN[x_SDkNN.size() - 1] << std::endl;
    writeFile << x_SDkNN[x_SDkNN.size() - 1] << std::endl;
    std::cout << "===Validation Results===\n";
    writeFile << "===Validation Results===\n";
    std::cout << "Validation Folder: " << validation_folder_path << std::endl;
    writeFile << "Validation Folder: " << validation_folder_path << std::endl;
    writeFile << "Number of data points: " << res_val.num_dataPoint << std::endl;
    std::cout << "Number of data points: " << res_val.num_dataPoint << std::endl;
    writeFile << "Average cost         : " << res_val.mean << std::endl;
    std::cout << "Average cost         : " << res_val.mean << std::endl;
    writeFile << "Variance             : " << res_val.variance << std::endl;
    std::cout << "Variance             : " << res_val.variance << std::endl;
    writeFile << "Error in estimating the expected cost: " << sqrt(res_val.variance / ((double)res_val.num_dataPoint)) << std::endl;
    std::cout << "Error in estimating the expected cost: " << sqrt(res_val.variance / ((double)res_val.num_dataPoint)) << std::endl;
    writeFile << res_val.alpha << "% CI Lower Bound: " << res_val.CI_lower << std::endl;
    std::cout << res_val.alpha << "% CI Lower Bound: " << res_val.CI_lower << std::endl;
    writeFile << res_val.alpha << "% CI Upper Bound: " << res_val.CI_upper << std::endl;
    std::cout << res_val.alpha << "% CI Upper Bound: " << res_val.CI_upper << std::endl;
    writeFile << "Time Elapsed(s) for Validation: " << duration_val << std::endl;
    std::cout << "Time Elapsed(s) for Validation: " << duration_val << std::endl;
    std::cout << "**********************************************************************\n";
    writeFile << "**********************************************************************\n";
    // close write file
    writeFile.close();
}


void interface_dynamic_nsd(const std::string& folder_path, const std::string& validation_folder_path, int max_iterations, int batch_size, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug) {
    // obtain estimated solution from NSD solver (fast)
    std::vector<double> x_SDkNN = dynamic_sdknn_solver(folder_path, max_iterations, batch_size, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    // set up write file
    std::clock_t time_start;
    time_start = std::clock();
    validationResult res_val = twoStageLP_validation_outputResultsV2(validation_folder_path, x_SDkNN);
    double duration_val = (std::clock() - time_start ) / (double) CLOCKS_PER_SEC;
    // write the results of quality of candidate solution
    std::string outputResults_path = folder_path + "/validationResults(NSDv2.0presolve_batch).txt";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    // current time
    std::time_t currTime = std::time(nullptr);
    std::cout << "**********************************************************************\n";
    writeFile << "**********************************************************************\n";
    std::cout << "Dynamic Stepsize, Presolve, and Batch Size(v2) are used\n";
    writeFile << "Dynamic Stepsize, Presolve, and Batch Size(v2) are used\n";
    std::cout << "Number of datapoints used in the presolve: " << N_pre << std::endl;
    writeFile << "Number of datapoints used in the presolve: " << N_pre << std::endl;
    writeFile << std::put_time(localtime(&currTime), "%c %Z") << "\n"; // write current time
    std::cout << std::put_time(localtime(&currTime), "%c %Z") << "\n"; // write current time
    std::cout << "Folder Path: " << folder_path << std::endl;
    writeFile << "Folder Path: " << folder_path << std::endl;
    writeFile << "Initial Point        : ";
    std::cout << "Initial Point        : ";
    std::cout << "Observed Predictor: ";
    writeFile << "Observed Predictor: ";
    for (int index = 0; index < observed_predictor.size() - 1; ++index) {
        std::cout << observed_predictor[index] << ", ";
        writeFile << observed_predictor[index] << ", ";
    }
    std::cout << observed_predictor[observed_predictor.size() - 1] << std::endl;
    writeFile << observed_predictor[observed_predictor.size() - 1] << std::endl;
    std::cout << "Number of iterations: " << max_iterations << std::endl;
    writeFile << "Number of iterations: " << max_iterations << std::endl;
    std::cout << "Batch Size: " << batch_size << std::endl;
    writeFile << "Batch Size: " << batch_size << std::endl;
    std::cout << "Solution obtained from SD-kNN v2 with dynamic stepsize(fast): ";
    writeFile << "Solution obtained from SD-kNN v2 with dynamic stepsize(fast): ";
    for (int index = 0; index < x_SDkNN.size() - 1; ++index) {
        std::cout << x_SDkNN[index] << ", ";
        writeFile << x_SDkNN[index] << ", ";
    }
    std::cout << x_SDkNN[x_SDkNN.size() - 1] << std::endl;
    writeFile << x_SDkNN[x_SDkNN.size() - 1] << std::endl;
    std::cout << "===Validation Results===\n";
    writeFile << "===Validation Results===\n";
    std::cout << "Validation Folder: " << validation_folder_path << std::endl;
    writeFile << "Validation Folder: " << validation_folder_path << std::endl;
    writeFile << "Number of data points: " << res_val.num_dataPoint << std::endl;
    std::cout << "Number of data points: " << res_val.num_dataPoint << std::endl;
    writeFile << "Average cost         : " << res_val.mean << std::endl;
    std::cout << "Average cost         : " << res_val.mean << std::endl;
    writeFile << "Variance             : " << res_val.variance << std::endl;
    std::cout << "Variance             : " << res_val.variance << std::endl;
    writeFile << "Error in estimating the expected cost: " << sqrt(res_val.variance / ((double)res_val.num_dataPoint)) << std::endl;
    std::cout << "Error in estimating the expected cost: " << sqrt(res_val.variance / ((double)res_val.num_dataPoint)) << std::endl;
    writeFile << res_val.alpha << "% CI Lower Bound: " << res_val.CI_lower << std::endl;
    std::cout << res_val.alpha << "% CI Lower Bound: " << res_val.CI_lower << std::endl;
    writeFile << res_val.alpha << "% CI Upper Bound: " << res_val.CI_upper << std::endl;
    std::cout << res_val.alpha << "% CI Upper Bound: " << res_val.CI_upper << std::endl;
    writeFile << "Time Elapsed(s) for Validation: " << duration_val << std::endl;
    std::cout << "Time Elapsed(s) for Validation: " << duration_val << std::endl;
    std::cout << "**********************************************************************\n";
    writeFile << "**********************************************************************\n";
    // close write file
    writeFile.close();
}


void interface_dynamic_nsd_v2(const std::string& folder_path, const std::string& validation_folder_path, int max_iterations, int batch_size, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug) {
    // obtain estimated solution from NSD solver (fast)
    std::vector<double> x_SDkNN = dynamic_sdknn_solver_v2(folder_path, max_iterations, batch_size, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    // set up write file
    std::clock_t time_start;
    time_start = std::clock();
    validationResult res_val = twoStageLP_validation_outputResultsV2(validation_folder_path, x_SDkNN);
    double duration_val = (std::clock() - time_start ) / (double) CLOCKS_PER_SEC;
    // write the results of quality of candidate solution
    std::string outputResults_path = folder_path + "/validationResults(NSDv2.0).txt";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    // current time
    std::time_t currTime = std::time(nullptr);
    std::cout << "**********************************************************************\n";
    writeFile << "**********************************************************************\n";
    std::cout << "Dynamic Stepsize, Presolve, and Batch Size(v2) are used\n";
    writeFile << "Dynamic Stepsize, Presolve, and Batch Size(v2) are used\n";
    std::cout << "Number of datapoints used in the presolve: " << N_pre << std::endl;
    writeFile << "Number of datapoints used in the presolve: " << N_pre << std::endl;
    writeFile << std::put_time(localtime(&currTime), "%c %Z") << "\n"; // write current time
    std::cout << std::put_time(localtime(&currTime), "%c %Z") << "\n"; // write current time
    std::cout << "Folder Path: " << folder_path << std::endl;
    writeFile << "Folder Path: " << folder_path << std::endl;
    writeFile << "Initial Point        : ";
    std::cout << "Initial Point        : ";
    std::cout << "Observed Predictor: ";
    writeFile << "Observed Predictor: ";
    for (int index = 0; index < observed_predictor.size() - 1; ++index) {
        std::cout << observed_predictor[index] << ", ";
        writeFile << observed_predictor[index] << ", ";
    }
    std::cout << observed_predictor[observed_predictor.size() - 1] << std::endl;
    writeFile << observed_predictor[observed_predictor.size() - 1] << std::endl;
    std::cout << "Number of iterations: " << max_iterations << std::endl;
    writeFile << "Number of iterations: " << max_iterations << std::endl;
    std::cout << "Batch Size: " << batch_size << std::endl;
    writeFile << "Batch Size: " << batch_size << std::endl;
    std::cout << "Solution obtained from SD-kNN v2 with dynamic stepsize(fast): ";
    writeFile << "Solution obtained from SD-kNN v2 with dynamic stepsize(fast): ";
    for (int index = 0; index < x_SDkNN.size() - 1; ++index) {
        std::cout << x_SDkNN[index] << ", ";
        writeFile << x_SDkNN[index] << ", ";
    }
    std::cout << x_SDkNN[x_SDkNN.size() - 1] << std::endl;
    writeFile << x_SDkNN[x_SDkNN.size() - 1] << std::endl;
    std::cout << "===Validation Results===\n";
    writeFile << "===Validation Results===\n";
    std::cout << "Validation Folder: " << validation_folder_path << std::endl;
    writeFile << "Validation Folder: " << validation_folder_path << std::endl;
    writeFile << "Number of data points: " << res_val.num_dataPoint << std::endl;
    std::cout << "Number of data points: " << res_val.num_dataPoint << std::endl;
    writeFile << "Average cost         : " << res_val.mean << std::endl;
    std::cout << "Average cost         : " << res_val.mean << std::endl;
    writeFile << "Variance             : " << res_val.variance << std::endl;
    std::cout << "Variance             : " << res_val.variance << std::endl;
    writeFile << "Error in estimating the expected cost: " << sqrt(res_val.variance / ((double)res_val.num_dataPoint)) << std::endl;
    std::cout << "Error in estimating the expected cost: " << sqrt(res_val.variance / ((double)res_val.num_dataPoint)) << std::endl;
    writeFile << res_val.alpha << "% CI Lower Bound: " << res_val.CI_lower << std::endl;
    std::cout << res_val.alpha << "% CI Lower Bound: " << res_val.CI_lower << std::endl;
    writeFile << res_val.alpha << "% CI Upper Bound: " << res_val.CI_upper << std::endl;
    std::cout << res_val.alpha << "% CI Upper Bound: " << res_val.CI_upper << std::endl;
    writeFile << "Time Elapsed(s) for Validation: " << duration_val << std::endl;
    std::cout << "Time Elapsed(s) for Validation: " << duration_val << std::endl;
    std::cout << "**********************************************************************\n";
    writeFile << "**********************************************************************\n";
    // close write file
    writeFile.close();
}


void interface_dynamic_nsd_v3(const std::string& folder_path, const std::string& validation_folder_path, int max_iterations, int batch_size, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug) {
    // obtain estimated solution from NSD solver (fast)
    std::vector<double> x_SDkNN = dynamic_sdknn_solver_v3(folder_path, max_iterations, batch_size, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    // set up write file
    std::clock_t time_start;
    time_start = std::clock();
    validationResult res_val = twoStageLP_validation_outputResultsV2(validation_folder_path, x_SDkNN);
    double duration_val = (std::clock() - time_start ) / (double) CLOCKS_PER_SEC;
    // write the results of quality of candidate solution
    std::string outputResults_path = folder_path + "/validationResults(NSDv3.0).txt";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    // current time
    std::time_t currTime = std::time(nullptr);
    std::cout << "**********************************************************************\n";
    writeFile << "**********************************************************************\n";
    std::cout << "Dynamic Stepsize, Presolve, and Batch Size(v3) are used\n";
    writeFile << "Dynamic Stepsize, Presolve, and Batch Size(v3) are used\n";
    std::cout << "Number of datapoints used in the presolve: " << N_pre << std::endl;
    writeFile << "Number of datapoints used in the presolve: " << N_pre << std::endl;
    writeFile << std::put_time(localtime(&currTime), "%c %Z") << "\n"; // write current time
    std::cout << std::put_time(localtime(&currTime), "%c %Z") << "\n"; // write current time
    std::cout << "Folder Path: " << folder_path << std::endl;
    writeFile << "Folder Path: " << folder_path << std::endl;
    writeFile << "Initial Point        : ";
    std::cout << "Initial Point        : ";
    std::cout << "Observed Predictor: ";
    writeFile << "Observed Predictor: ";
    for (int index = 0; index < observed_predictor.size() - 1; ++index) {
        std::cout << observed_predictor[index] << ", ";
        writeFile << observed_predictor[index] << ", ";
    }
    std::cout << observed_predictor[observed_predictor.size() - 1] << std::endl;
    writeFile << observed_predictor[observed_predictor.size() - 1] << std::endl;
    std::cout << "Number of iterations: " << max_iterations << std::endl;
    writeFile << "Number of iterations: " << max_iterations << std::endl;
    std::cout << "Batch Size: " << batch_size << std::endl;
    writeFile << "Batch Size: " << batch_size << std::endl;
    std::cout << "Solution obtained from SD-kNN-Batch v3 with dynamic stepsize(fast): ";
    writeFile << "Solution obtained from SD-kNN-Batch v3 with dynamic stepsize(fast): ";
    for (int index = 0; index < x_SDkNN.size() - 1; ++index) {
        std::cout << x_SDkNN[index] << ", ";
        writeFile << x_SDkNN[index] << ", ";
    }
    std::cout << x_SDkNN[x_SDkNN.size() - 1] << std::endl;
    writeFile << x_SDkNN[x_SDkNN.size() - 1] << std::endl;
    std::cout << "===Validation Results===\n";
    writeFile << "===Validation Results===\n";
    std::cout << "Validation Folder: " << validation_folder_path << std::endl;
    writeFile << "Validation Folder: " << validation_folder_path << std::endl;
    writeFile << "Number of data points: " << res_val.num_dataPoint << std::endl;
    std::cout << "Number of data points: " << res_val.num_dataPoint << std::endl;
    writeFile << "Average cost         : " << res_val.mean << std::endl;
    std::cout << "Average cost         : " << res_val.mean << std::endl;
    writeFile << "Variance             : " << res_val.variance << std::endl;
    std::cout << "Variance             : " << res_val.variance << std::endl;
    writeFile << "Error in estimating the expected cost: " << sqrt(res_val.variance / ((double)res_val.num_dataPoint)) << std::endl;
    std::cout << "Error in estimating the expected cost: " << sqrt(res_val.variance / ((double)res_val.num_dataPoint)) << std::endl;
    writeFile << res_val.alpha << "% CI Lower Bound: " << res_val.CI_lower << std::endl;
    std::cout << res_val.alpha << "% CI Lower Bound: " << res_val.CI_lower << std::endl;
    writeFile << res_val.alpha << "% CI Upper Bound: " << res_val.CI_upper << std::endl;
    std::cout << res_val.alpha << "% CI Upper Bound: " << res_val.CI_upper << std::endl;
    writeFile << "Time Elapsed(s) for Validation: " << duration_val << std::endl;
    std::cout << "Time Elapsed(s) for Validation: " << duration_val << std::endl;
    std::cout << "**********************************************************************\n";
    writeFile << "**********************************************************************\n";
    // close write file
    writeFile.close();
}


// SQQP
double twoStageQP_secondStageCost(const std::vector<double>& x, standardTwoStageParameters_QP& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap) {
    int y_size = model_parameters.D.num_col;
    // set up the model
    IloEnv env;
    IloModel mod(env);
    IloNumVarArray y(env,y_size,0,IloInfinity,ILOFLOAT); // second stage decision variables
    mod.add(y);
    // objective function
    IloExpr expr_obj(env);
    // linear
    for (auto it = model_parameters.d.vec.begin(); it != model_parameters.d.vec.end(); ++it) {
        expr_obj += (it -> second) * y[it -> first];
    }
    // quadratic
    for (std::map<std::pair<int,int>, double>::iterator it = model_parameters.P.mat.begin(); it != model_parameters.P.mat.end(); ++it) {
        expr_obj += y[(it -> first).first] * y[(it -> first).second] * (it -> second) * 0.5;
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
    // set up cplex solver
    IloCplex cplex(env);
    cplex.extract(mod);
    cplex.setOut(env.getNullStream());
    IloBool solvable_flag = cplex.solve();
    // cost in the second stage problem
    double second_stage_cost = cplex.getObjValue();
    env.end();
    return second_stage_cost;
}


validationResult twoStageQP_validation_outputResultsV2(const std::string& folder_path, const std::vector<double>& x_candidate) {
    // STEP 1: INITIALIZATION
    bool flag_be;
    bool flag_bi;
    bool flag_Ce;
    bool flag_Ci;
    // create directory paths for database and model
    std::string be_DB_path = folder_path + "/be_DB.txt";
    std::string bi_DB_path = folder_path + "/bi_DB.txt";
    std::string Ce_DB_path = folder_path + "/Ce_DB.txt";
    std::string Ci_DB_path = folder_path + "/Ci_DB.txt";
    std::string model_path = folder_path + "/model.txt";
    std::string sto_path = folder_path + "/sto.txt";
    // convert all the paths into constant chars
    const char* be_DB_path_const = be_DB_path.c_str();
    const char* bi_DB_path_const = bi_DB_path.c_str();
    const char* Ce_DB_path_const = Ce_DB_path.c_str();
    const char* Ci_DB_path_const = Ci_DB_path.c_str();
    const char* model_path_const = model_path.c_str();
    const char* sto_path_const = sto_path.c_str();
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
    // read model file
    model_parameters = readStandardTwoStageParameters_QP(model_path);
    // read sto file
    RHSmap = readStochasticMap(sto_path);
    long sample_size = 0;
    if (flag_be == true) {
        sample_size = be_DB[0].size();
    }
    else if (flag_bi == true) {
        sample_size = bi_DB[0].size();
    }
    else if (flag_Ce == true) {
        sample_size = Ce_DB[0].size();
    }
    else if (flag_Ci == true) {
        sample_size = Ci_DB[0].size();
    }
    else {
        throw std::invalid_argument("ERROR: Database is empty!\n");
    }
    // determine appropriate model
    double secondStageTotalCost = 0;
    double firstStageCost = model_parameters.c * x_candidate;
    firstStageCost += 0.5 * (x_candidate * model_parameters.Q * x_candidate);
    double variance_P = 0; // intermediate component for calculating variance
    for (int idx_scenario = 0; idx_scenario < sample_size; ++idx_scenario) {
        // obtain a new data point
        dataPoint be_datapoint;
        if (flag_be == true) {
            be_datapoint = be_DB[0][idx_scenario];
        }
        dataPoint bi_datapoint;
        if (flag_bi == true) {
            bi_datapoint = bi_DB[0][idx_scenario];
        }
        dataPoint Ce_datapoint;
        if (flag_Ce == true) {
            Ce_datapoint = Ce_DB[0][idx_scenario];
        }
        dataPoint Ci_datapoint;
        if (flag_Ci == true) {
            Ci_datapoint = Ci_DB[0][idx_scenario];
        }
        // merge all the datapoints
        secondStageRHSpoint RHS_datapoint = merge_randomVector(be_datapoint, bi_datapoint, Ce_datapoint, Ci_datapoint);
        double tempCost = twoStageQP_secondStageCost(x_candidate, model_parameters, RHS_datapoint, RHSmap);
        secondStageTotalCost += tempCost / ((double) sample_size);
        double tempTotalCost = tempCost + firstStageCost;
        double _n_ = idx_scenario + 1;
        variance_P = variance_P * (_n_ - 1) / _n_ + tempTotalCost * tempTotalCost / _n_;
    }
    double _n_ = sample_size;
    validationResult result;
    result.mean = firstStageCost + secondStageTotalCost;
    result.variance = (variance_P - result.mean * result.mean) * _n_ / (_n_ - 1);
    result.num_dataPoint = sample_size;
    double halfMargin = result.Zalpha * sqrt(result.variance / _n_);
    result.CI_lower = result.mean - halfMargin;
    result.CI_upper = result.mean + halfMargin;
    // write computational results
    std::string outputResults_path = folder_path + "/validationResultsV2.txt";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    // current time
    std::time_t currTime = std::time(nullptr);
    writeFile << "***************************************************\n";
    writeFile << "Estimating the quality of candidate solution\n";
    writeFile << std::put_time(localtime(&currTime), "%c %Z") << "\n"; // write current time
    writeFile << "Candidate solution             : ";
    long x_size = x_candidate.size();
    for (int index = 0; index < x_size - 1; ++index) {
        writeFile << x_candidate[index] << ", ";
    }
    writeFile << x_candidate[x_size - 1] << "\n";
    writeFile << "Number of data points          : " << result.num_dataPoint << "\n";
    std::cout << "Average validation cost        : " << result.mean << "\n";
    writeFile << "Average validation cost        : " << result.mean << "\n";
    writeFile << "Variance                       : " << result.variance << "\n";
    writeFile << "Variance in estimating the mean: " << sqrt(result.variance/ _n_) << "\n";
    writeFile << result.alpha << "% confidence interval of expected cost: [" << result.CI_lower << ", " << result.CI_upper << "]\n";
    writeFile << "***************************************************\n";
    return result;
}


void twoStageQP_empirical_cost(const std::string& folder_path) {
    // STEP 1: INITIALIZATION
    bool flag_be;
    bool flag_bi;
    bool flag_Ce;
    bool flag_Ci;
    // create directory paths for database and model
    std::string be_DB_path = folder_path + "/be_DB.txt";
    std::string bi_DB_path = folder_path + "/bi_DB.txt";
    std::string Ce_DB_path = folder_path + "/Ce_DB.txt";
    std::string Ci_DB_path = folder_path + "/Ci_DB.txt";
    std::string model_path = folder_path + "/model.txt";
    std::string sto_path = folder_path + "/sto.txt";
    // convert all the paths into constant chars
    const char* be_DB_path_const = be_DB_path.c_str();
    const char* bi_DB_path_const = bi_DB_path.c_str();
    const char* Ce_DB_path_const = Ce_DB_path.c_str();
    const char* Ci_DB_path_const = Ci_DB_path.c_str();
    const char* model_path_const = model_path.c_str();
    const char* sto_path_const = sto_path.c_str();
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
    // read model file
    model_parameters = readStandardTwoStageParameters_QP(model_path);
    // read sto file
    RHSmap = readStochasticMap(sto_path);
    long sample_size = 0;
    if (flag_be == true) {
        sample_size = be_DB[0].size();
    }
    else if (flag_bi == true) {
        sample_size = bi_DB[0].size();
    }
    else if (flag_Ce == true) {
        sample_size = Ce_DB[0].size();
    }
    else if (flag_Ci == true) {
        sample_size = Ci_DB[0].size();
    }
    else {
        throw std::invalid_argument("ERROR: Database is empty!\n");
    }
    // determine appropriate model
    // solve a large LP
    IloEnv env;
    IloModel mod(env);
    IloNumVarArray x(env,model_parameters.c.num_entry,-IloInfinity,IloInfinity,ILOFLOAT);
    mod.add(x);
    std::vector<IloNumVarArray> y;
    for (int idx_scenario = 0; idx_scenario < sample_size; ++idx_scenario) {
        IloNumVarArray y_scenario(env,model_parameters.D.num_col,0,IloInfinity,ILOFLOAT); // y >= 0
        y.push_back(y_scenario);
        mod.add(y[idx_scenario]);
    }
    IloExpr expr_obj(env);
    // first stage objective linear
    for (auto it = model_parameters.c.vec.begin(); it != model_parameters.c.vec.end(); ++it) {
        expr_obj += (it -> second) * x[it -> first];
    }
    // first stage objective quadratic
    for (std::map<std::pair<int,int>, double>::iterator it = model_parameters.Q.mat.begin(); it != model_parameters.Q.mat.end(); ++it) {
        expr_obj += x[(it -> first).first] * x[(it -> first).second] * (it -> second) * 0.5;
    }
    for (int idx_scenario = 0; idx_scenario < sample_size; ++idx_scenario) {
        // second stage objective linear
        for (auto it = model_parameters.d.vec.begin(); it != model_parameters.d.vec.end(); ++it) {
            expr_obj += (1.0 / (double) sample_size) * (it -> second) * y[idx_scenario][it -> first];
        }
        // second stage objective quadratic
        for (std::map<std::pair<int,int>, double>::iterator it = model_parameters.P.mat.begin(); it != model_parameters.P.mat.end(); ++it) {
            expr_obj += y[idx_scenario][(it -> first).first] * y[idx_scenario][(it -> first).second] * (it -> second) * 0.5 * (1.0 / (double) sample_size);
        }
    }
    // add objective
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
        exprs[(it -> first).first] += (it -> second) * x[(it -> first).second];
    }
    // right hand side
    for (auto it = model_parameters.b.vec.begin(); it != model_parameters.b.vec.end(); ++it) {
        exprs[it -> first] -= (it -> second);
    }
    // add constraints
    for (int index_cons = 0; index_cons < model_parameters.A.num_row; ++index_cons) {
        mod.add(exprs[index_cons] <= 0);
    }
    for (int idx_scenario = 0; idx_scenario < sample_size; ++idx_scenario) {
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
            exprs_eq[(it -> first).first] += (it -> second) * y[idx_scenario][(it -> first).second];
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
        // obtain a new data point
        dataPoint be_datapoint;
        if (flag_be == true) {
            be_datapoint = be_DB[0][idx_scenario];
        }
        dataPoint bi_datapoint;
        if (flag_bi == true) {
            bi_datapoint = bi_DB[0][idx_scenario];
        }
        dataPoint Ce_datapoint;
        if (flag_Ce == true) {
            Ce_datapoint = Ce_DB[0][idx_scenario];
        }
        dataPoint Ci_datapoint;
        if (flag_Ci == true) {
            Ci_datapoint = Ci_DB[0][idx_scenario];
        }
        // merge all the datapoints
        secondStageRHSpoint RHS_datapoint = merge_randomVector(be_datapoint, bi_datapoint, Ce_datapoint, Ci_datapoint);
        // coefficients before x (stochastic part) equality (i.e., Cij * xj map: <i,j> )
        for (int idx_Ce = 0; idx_Ce < RHS_datapoint.Ce.size(); ++idx_Ce) {
            exprs_eq[RHSmap.Ce_map[idx_Ce].first] += RHS_datapoint.Ce[idx_Ce] * x[RHSmap.Ce_map[idx_Ce].second];
        }
        // coefficients before x (stochastic part) inequality (location is behind equality constraints)
        for (int idx_Ci = 0; idx_Ci < RHS_datapoint.Ci.size(); ++idx_Ci) {
            exprs_eq[RHSmap.Ci_map[idx_Ci].first + model_parameters.num_eq] += RHS_datapoint.Ci[idx_Ci] * x[RHSmap.Ci_map[idx_Ci].second];
        }
        // right hand side (stochastic part) equality be_(i) equality
        for (int idx_be = 0; idx_be < RHS_datapoint.be.size(); ++idx_be) {
            exprs_eq[RHSmap.be_map[idx_be]] -= RHS_datapoint.be[idx_be];
        }
        // right hand side (stochastic part) equality bi_(i) inequality
        for (int idx_bi = 0; idx_bi < RHS_datapoint.bi.size(); ++idx_bi) {
            exprs_eq[RHSmap.bi_map[idx_bi] + model_parameters.num_eq] -= RHS_datapoint.bi[idx_bi];
        }
        // add the equality constraints
        for (int index_eq = 0; index_eq < model_parameters.D.num_row; ++index_eq) {
            constraintsEquality.add(exprs_eq[index_eq] == 0);
        }
        mod.add(constraintsEquality);
    }
    // create cplex environment
    IloCplex cplex(env);
    cplex.extract(mod);
    cplex.solve();
    // write computational results
    std::string outputResults_path = folder_path + "/baseline_slp.txt";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    // current time
    std::time_t currTime = std::time(nullptr);
    writeFile << "***************************************************\n";
    writeFile << "Input Folder: " << folder_path << std::endl;
    writeFile << std::put_time(localtime(&currTime), "%c %Z") << "\n"; // write current time
    writeFile << "Baseline Solution             : ";
    long x_size = model_parameters.c.num_entry;
    for (int index = 0; index < x_size - 1; ++index) {
        writeFile << cplex.getValue(x[index]) << ", ";
    }
    writeFile << cplex.getValue(x[x_size - 1]) << "\n";
    writeFile << "Baseline Cost: " << cplex.getObjValue() << std::endl;
    env.end();
    writeFile.close();
}


void interface_dynamic_nsd_qq_presolve(const std::string& folder_path, const std::string& validation_folder_path, int max_iterations, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug) {
    // obtain estimated solution from NSD solver (fast)
    std::vector<double> x_SDkNN = dynamic_sdknn_qq_solver_presolve_fullFace(folder_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    // set up write file
    std::clock_t time_start;
    time_start = std::clock();
    validationResult res_val = twoStageQP_validation_outputResultsV2(validation_folder_path, x_SDkNN);
    double duration_val = (std::clock() - time_start ) / (double) CLOCKS_PER_SEC;
    // write the results of quality of candidate solution
    std::string outputResults_path = folder_path + "/validationResults(NSD_QQv2.0presolve).txt";
    const char* outputResults_path_const = outputResults_path.c_str();
    std::fstream writeFile;
    writeFile.open(outputResults_path_const,std::fstream::app);
    // current time
    std::time_t currTime = std::time(nullptr);
    std::cout << "**********************************************************************\n";
    writeFile << "**********************************************************************\n";
    std::cout << "Dynamic Stepsize and Presolve(v2) are used\n";
    writeFile << "Dynamic Stepsize and Presolve(v2) are used\n";
    std::cout << "Number of datapoints used in the presolve: " << N_pre << std::endl;
    writeFile << "Number of datapoints used in the presolve: " << N_pre << std::endl;
    writeFile << std::put_time(localtime(&currTime), "%c %Z") << "\n"; // write current time
    std::cout << std::put_time(localtime(&currTime), "%c %Z") << "\n"; // write current time
    std::cout << "Folder Path: " << folder_path << std::endl;
    writeFile << "Folder Path: " << folder_path << std::endl;
    writeFile << "Initial Point        : ";
    std::cout << "Initial Point        : ";
    std::cout << "Observed Predictor: ";
    writeFile << "Observed Predictor: ";
    for (int index = 0; index < observed_predictor.size() - 1; ++index) {
        std::cout << observed_predictor[index] << ", ";
        writeFile << observed_predictor[index] << ", ";
    }
    std::cout << observed_predictor[observed_predictor.size() - 1] << std::endl;
    writeFile << observed_predictor[observed_predictor.size() - 1] << std::endl;
    std::cout << "Number of iterations: " << max_iterations << std::endl;
    writeFile << "Number of iterations: " << max_iterations << std::endl;
    std::cout << "Solution obtained from SD-kNN-QQ with dynamic stepsize(fast): ";
    writeFile << "Solution obtained from SD-kNN-QQ with dynamic stepsize(fast): ";
    for (int index = 0; index < x_SDkNN.size() - 1; ++index) {
        std::cout << x_SDkNN[index] << ", ";
        writeFile << x_SDkNN[index] << ", ";
    }
    std::cout << x_SDkNN[x_SDkNN.size() - 1] << std::endl;
    writeFile << x_SDkNN[x_SDkNN.size() - 1] << std::endl;
    std::cout << "===Validation Results===\n";
    writeFile << "===Validation Results===\n";
    std::cout << "Validation Folder: " << validation_folder_path << std::endl;
    writeFile << "Validation Folder: " << validation_folder_path << std::endl;
    writeFile << "Number of data points: " << res_val.num_dataPoint << std::endl;
    std::cout << "Number of data points: " << res_val.num_dataPoint << std::endl;
    writeFile << "Average cost         : " << res_val.mean << std::endl;
    std::cout << "Average cost         : " << res_val.mean << std::endl;
    writeFile << "Variance             : " << res_val.variance << std::endl;
    std::cout << "Variance             : " << res_val.variance << std::endl;
    writeFile << "Error in estimating the expected cost: " << sqrt(res_val.variance / ((double)res_val.num_dataPoint)) << std::endl;
    std::cout << "Error in estimating the expected cost: " << sqrt(res_val.variance / ((double)res_val.num_dataPoint)) << std::endl;
    writeFile << res_val.alpha << "% CI Lower Bound: " << res_val.CI_lower << std::endl;
    std::cout << res_val.alpha << "% CI Lower Bound: " << res_val.CI_lower << std::endl;
    writeFile << res_val.alpha << "% CI Upper Bound: " << res_val.CI_upper << std::endl;
    std::cout << res_val.alpha << "% CI Upper Bound: " << res_val.CI_upper << std::endl;
    writeFile << "Time Elapsed(s) for Validation: " << duration_val << std::endl;
    std::cout << "Time Elapsed(s) for Validation: " << duration_val << std::endl;
    std::cout << "**********************************************************************\n";
    writeFile << "**********************************************************************\n";
    // close write file
    writeFile.close();
}
