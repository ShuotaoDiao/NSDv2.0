//
//  main.cpp
//  NSDv2.0
//
//  Created by Shuotao Diao on 8/4/21.
//

#include <iostream>
#include <ilcplex/ilocplex.h>

#include "NSD_ioStochastic.hpp"
#include "NSD_solver.hpp"
#include "NSD_QQ_solver.hpp"
#include "NSD_utils.hpp"

void qp_dual_test() {
    std::cout << "Test on basic solver performance!\n";
    IloEnv env;
    IloModel mod(env);
    IloNumVarArray x(env,2,-5,5,ILOFLOAT);
    mod.add(x);
    IloExpr expr_obj(env);
    expr_obj = x[0] * x[0] + x[1] * x[1];
    IloObjective obj = IloMinimize(env, expr_obj);
    mod.add(obj);
    // constrant
    IloRangeArray constraints(env);
    IloExpr expr_con(env);
    expr_con = x[0] + x[1];
    constraints.add(expr_con == 1);
    // extra constraints
    constraints.add(x[0] + x[1] <= 2);
    mod.add(constraints);
    // set up cplex solver
    IloCplex cplex(env);
    cplex.extract(mod);
    cplex.solve();
    // output solutions
    std::cout << "Optimal value: " << cplex.getObjValue() << std::endl;
    std::cout << "x[0] = " << cplex.getValue(x[0]) << std::endl;
    std::cout << "x[1] = " << cplex.getValue(x[1]) << std::endl;
    // dual variables
    IloNumArray duals(env);
    cplex.getDuals(duals, constraints);
    std::cout << "duals[0] = " << duals[0] << std::endl;
    std::cout << "duals[1] = " << duals[1] << std::endl;
    env.end();
}

void minorant_test() {
    minorant cut1;
    cut1.alpha = 0.5;
    std::vector<double> b(4,1.2);
    cut1.beta = b;
    std::cout << "cut1\n";
    std::cout << cut1.beta[0] << " " << cut1.beta[1] << " " << cut1.beta[2] << " " << cut1.beta[3] << "\n";
    minorant cut2 = cut1;
    std::vector<double> a(4,2.3);
    cut2.beta = b + a;
    std::cout << "cut1\n";
    std::cout << cut1.alpha << "\n";
    std::cout << cut1.beta[0] << " " << cut1.beta[1] << " " << cut1.beta[2] << " " << cut1.beta[3] << "\n";
    std::cout << "cut2\n";
    std::cout << cut2.alpha << "\n";
    std::cout << cut2.beta[0] << " " << cut2.beta[1] << " " << cut2.beta[2] << " " << cut2.beta[3] << "\n";
}

void nsd_bk19() {
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN/twoStageShipment/experiment6/case9";
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN/twoStageShipment/experiment6/trueValidation";
    int max_iterations[] = {1, 104, 266, 496, 806, 1211, 1729, 2381, 3191, 4186, 5369, 6854, 8596, 10661, 13091}; // 200
    std::vector<double> observed_predictor(3,0.0);
    observed_predictor[0] = -0.3626;//(-0.3626, 0.5871, -0.2987)
    observed_predictor[1] = 0.5871;
    observed_predictor[2] = -0.2987;
    double f_upperbound = 2000;
    double f_lowerbound = 0;
    double sigma_upperbound = 100;
    double sigma_lowerbound = 1;
    bool flag_debug = false;
    int N_pre = 49;
    for (int idx = 8; idx < 9; ++idx) {
        //interface_sdknn(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
        interface_sdknn_v2(folder_path, validation_path, max_iterations[idx], f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    }
    
} // end nsd_bk19

int main(int argc, const char * argv[]) {
    //qp_dual_test();
    // test on reading sto file
    //std::string path_sto = "/Users/sonny/Documents/numericalExperiment/SDkNN2/sqqp/sto.txt";
    //readStochasticMap(path_sto);
    
    // bk19
    //nsd_bk19();
    // baa 99 small
    /*
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99small/experiment2/case10";
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99small/experiment1/kNNValidation1";
    int max_iterations = 1000; // 200
    int batch_size = 5;
    std::vector<double> observed_predictor;
    observed_predictor.push_back(108);
    observed_predictor.push_back(106);
    observed_predictor.push_back(104);
    observed_predictor.push_back(102);
    observed_predictor.push_back(100);
    double f_upperbound = 10000;
    double f_lowerbound = -3000;
    double sigma_upperbound = 100;
    double sigma_lowerbound = 1;
    bool flag_debug = false;
    int N_pre = 15000;
    //std::vector<double> x_est = dynamic_sdknn_solver_presolve_fullDual(folder_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    //twoStageLP_validation_outputResultsV2(validation_path, x_est);
    //twoStageLP_empirical_cost(validation_path);
    //interface_dynamic_nsd_presolve(folder_path, validation_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    interface_dynamic_nsd_presolve_v3(folder_path, validation_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug); // SD-kNN Basic
    //interface_dynamic_nsd_v2(folder_path, validation_path, max_iterations, batch_size, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    //interface_dynamic_nsd_v3(folder_path, validation_path, max_iterations, batch_size, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug); // SD-kNN-Batch
     */
    // baa 99 large
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99large/experiment8/case1";
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/baa99large/experiment8/kNNValidation2";
    int max_iterations = 266; // 200
    int batch_size = 5;
    int num_predictor = 25;
    std::vector<double> observed_predictor;
    for (int idx = 0; idx < num_predictor; ++idx) {
        std::cout << 108 - ((double) idx) * (1.0 / 3.0)  << std::endl;
        observed_predictor.push_back(108 - ((double) idx) * (1.0 / 3.0));
    }
    double f_upperbound = 10000;
    double f_lowerbound = -20000; // -20000
    double sigma_upperbound = 100;
    double sigma_lowerbound = 1;
    bool flag_debug = true;
    int N_pre = 49;// 15000
    //std::vector<double> x_est = dynamic_sdknn_solver_presolve_fullDual(folder_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    //twoStageLP_validation_outputResultsV2(validation_path, x_est);
    //twoStageLP_empirical_cost(validation_path); // find baseline cost and solutions
    //interface_dynamic_nsd_presolve(folder_path, validation_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    interface_dynamic_nsd_presolve_v3(folder_path, validation_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug); // SD-kNN Basic
    //interface_dynamic_nsd_v2(folder_path, validation_path, max_iterations, batch_size, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    //interface_dynamic_nsd_v3(folder_path, validation_path, max_iterations, batch_size, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug); // SD-kNN-Batch
    // bk 19
    /*
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/bk19/experiment5/case10";
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/slp/bk19/trueValidation";
    int max_iterations = 100; // 200
    int batch_size = 50;
    std::vector<double> observed_predictor(3,0.0);
    observed_predictor[0] = -0.3626;//(-0.3626, 0.5871, -0.2987)
    observed_predictor[1] = 0.5871;
    observed_predictor[2] = -0.2987;
    double f_upperbound = 2000;
    double f_lowerbound = 0;
    double sigma_upperbound = 100;
    double sigma_lowerbound = 1;
    bool flag_debug = false;
    int N_pre = 80000; // 80000
    //std::vector<double> x_est = dynamic_sdknn_solver_presolve_fullDual(folder_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    //std::vector<double> x_est(4,10.0);
    //twoStageLP_validation_outputResultsV2(validation_path, x_est);
    //twoStageLP_empirical_cost(validation_path);
    //interface_dynamic_nsd_presolve(folder_path, validation_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    //interface_dynamic_nsd_presolve_v3(folder_path, validation_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug); // SD-kNN basic
    //interface_dynamic_nsd(folder_path, validation_path, max_iterations, batch_size, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    //interface_dynamic_nsd_v2(folder_path, validation_path, max_iterations, batch_size, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    interface_dynamic_nsd_v3(folder_path, validation_path, max_iterations, batch_size, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug); // SD-kNN-Batch
    */
    //minorant_test();
    // small SQQP problem
    /*
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/sqqp/experiment1/case2";
    int max_iterations = 10;
    std::vector<double> observed_predictor;
    observed_predictor.push_back(2);
    observed_predictor.push_back(1);
    double f_upperbound = 100;
    double f_lowerbound = -160;
    double sigma_upperbound = 100;
    double sigma_lowerbound = 1;
    bool flag_debug = true;
    int N_pre = 1000;
    std::vector<double> x_est = dynamic_sdknn_qq_solver_presolve_fullFace(folder_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    */
    // baa99 SQQP problem
    /*
    std::string folder_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/sqqp/baa99small/experiment2/case19";
    std::string validation_path = "/Users/sonny/Documents/numericalExperiment/SDkNN2/sqqp/baa99small/experiment2/kNNValidation2";
    int max_iterations = 300;
    std::vector<double> observed_predictor;
    observed_predictor.push_back(0.9);
    observed_predictor.push_back(110);
    observed_predictor.push_back(107);
    double f_upperbound = 180000;
    double f_lowerbound = -1600;
    double sigma_upperbound = 100;
    double sigma_lowerbound = 1;
    bool flag_debug = true;
    int N_pre = 1000;
    //std::vector<double> x_est = dynamic_sdknn_qq_solver_presolve_fullFace(folder_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
    //twoStageQP_empirical_cost(validation_path);
    //twoStageQP_validation_outputResultsV2(validation_path,x_est);
    interface_dynamic_nsd_qq_presolve(folder_path, validation_path, max_iterations, f_upperbound, f_lowerbound, sigma_upperbound, sigma_lowerbound, observed_predictor, N_pre, flag_debug);
     */
    return 0;
}
