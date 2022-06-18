//
//  NSD_utils.hpp
//  NSDv2.0
//
//  Created by Shuotao Diao on 8/5/21.
//

#ifndef NSD_utils_hpp
#define NSD_utils_hpp

#include <stdio.h>
#include "NSD_solver.hpp"
#include "NSD_QQ_solver.hpp"
#include "NSD_solver_v2.hpp"

double twoStageLP_secondStageCost(const std::vector<double>& x, standardTwoStageParameters& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap);

validationResult twoStageLP_validation_outputResultsV2(const std::string& folder_path, const std::vector<double>& x_candidate);

void twoStageLP_empirical_cost(const std::string& folder_path);

void interface_dynamic_nsd_presolve(const std::string& folder_path, const std::string& validation_folder_path, int max_iterations, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug);

void interface_dynamic_nsd_presolve_v3(const std::string& folder_path, const std::string& validation_folder_path, int max_iterations, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug);

void interface_dynamic_nsd(const std::string& folder_path, const std::string& validation_folder_path, int max_iterations, int batch_size, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug);

void interface_dynamic_nsd_v2(const std::string& folder_path, const std::string& validation_folder_path, int max_iterations, int batch_size, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug);

void interface_sdknn(const std::string& folder_path, const std::string& validation_folder_path, int max_iterations, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug);

void interface_sdknn_v2(const std::string& folder_path, const std::string& validation_folder_path, int max_iterations, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug); // correct one

void interface_dynamic_nsd_v3(const std::string& folder_path, const std::string& validation_folder_path, int max_iterations, int batch_size, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug);
// SQQP
double twoStageQP_secondStageCost(const std::vector<double>& x, standardTwoStageParameters_QP& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap);

validationResult twoStageQP_validation_outputResultsV2(const std::string& folder_path, const std::vector<double>& x_candidate);

void twoStageQP_empirical_cost(const std::string& folder_path);

void interface_dynamic_nsd_qq_presolve(const std::string& folder_path, const std::string& validation_folder_path, int max_iterations, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug);
#endif /* NSD_utils_hpp */
