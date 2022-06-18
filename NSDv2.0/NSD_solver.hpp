//
//  NSD_solver.hpp
//  NSDv2.0
//
//  Created by Shuotao Diao on 8/4/21.
//

#ifndef NSD_solver_hpp
#define NSD_solver_hpp

#include <stdio.h>
#include <stdlib.h> // rand
#include <ctime>
#include <cmath>

#include "NSD_dataStructure.hpp"
#include "NSD_ioDB.hpp"
#include "NSD_ioModel.hpp"
#include "NSD_ioStochastic.hpp"

// compare duals
bool if_duals_equal(const dualMultipliers& dual1, const dualMultipliers& dual2);
bool if_new_dual(const std::vector<dualMultipliers>& duals, const std::vector<int>& indices, const dualMultipliers& candidate_dual);
bool if_new_dual(const std::vector<dualMultipliers>& duals, const dualMultipliers& candidate_dual);

// obtain dual multiplers of the second stage, given x (first stage decision variable)
dualMultipliers twoStageLP_secondStageDual(const std::vector<double>& x, standardTwoStageParameters& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap);

// functions for generating feasibility cut
// find extreme ray
dualMultipliers twoStageLP_secondStageExtremRay(const std::vector<double>& x, standardTwoStageParameters& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap);
// construct feasibility cut
feasibilityCut twoStageLP_feasibilityCutGeneration(const dualMultipliers& extremeRay, standardTwoStageParameters& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap);

// projection for the first stage in the two stage linear programming
std::vector<double> twoStageLP_projection(const std::vector<double>& x, standardTwoStageParameters& model_parameters);

// presolve
std::vector<double> twoStageLP_presolve(standardTwoStageParameters& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap);

// incumbent selection
bool incumbent_selection_check(double q, const std::vector<double>& x_candidate, const std::vector<double>& x_incumbent, sparseVector& c, const std::vector<minorant>& minorants, const std::vector<minorant>& minorants_new, const std::vector<int>& active_minorants);

bool incumbent_selection_check_v2(double q, const std::vector<double>& x_candidate, const std::vector<double>& x_incumbent, sparseVector& c, const std::vector<minorant>& minorants, const std::vector<minorant>& minorants_new);
// NSD solver with presolve and selected track of duals (initial k will be based on the presolve)
std::vector<double> dynamic_sdknn_solver_presolve(const std::string& folder_path, int max_iterations, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug);

// NSD solver with presolve all the explored duals will be used
std::vector<double> dynamic_sdknn_solver_presolve_fullDual(const std::string& folder_path, int max_iterations, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug);

// NSD solver with presolve, all the explored duals will be used, extra memory to store pi*C_det and pi*e_det
std::vector<double> dynamic_sdknn_solver_presolve_fullDual_v3(const std::string& folder_path, int max_iterations, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug);

// NSD solver with presolve and batch size (skip the incumbent selection and explored duals in the batch), all the explored duals will be used
std::vector<double> dynamic_sdknn_solver(const std::string& folder_path, int max_iterations, int batch_size,double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug);

// built-in batch, further refine the minorants
std::vector<double> dynamic_sdknn_solver_v2(const std::string& folder_path, int max_iterations, int batch_size,double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug);

// built-in batch, further refine the minorants, extra memory to store pi*C_det and pi*e_det
std::vector<double> dynamic_sdknn_solver_v3(const std::string& folder_path, int max_iterations, int batch_size,double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug);

#endif /* NSD_solver_hpp */
