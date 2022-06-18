//
//  NSD_QQ_solver.hpp
//  NSDv2.0
//
//  Created by Shuotao Diao on 8/5/21.
//

#ifndef NSD_QQ_solver_hpp
#define NSD_QQ_solver_hpp

#include <stdio.h>
#include <ctime>
#include <cmath>

#include "NSD_dataStructure.hpp"
#include "NSD_ioDB.hpp"
#include "NSD_ioModel.hpp"
#include "NSD_ioStochastic.hpp"

// check two face
bool if_face_equal(const face& face1, const face& face2);
// check if the face is new
bool if_face_new(const std::vector<face>& faces, const std::vector<int>& indices, const face& face_candidate);
bool if_face_new(const std::vector<face>& faces, const face& face_candidate);

// obtain dual multiplers of the second stage, given x (first stage decision variable)
dualMultipliers_QP twoStageQP_secondStagePrimal(const std::vector<double>& x, standardTwoStageParameters_QP& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap);
// obtain dual multipliers of the second stage, given x and face
dualMultipliers_QP twoStageQP_secondStageDual(const std::vector<double>& x, standardTwoStageParameters_QP& model_parameters, const secondStageRHSpoint& rhs, const face& face_cur, const secondStageRHSmap& RHSmap);

// functions for generating feasibility cut
dualMultipliers_QP twoStageQP_secondStageExtremRay(const std::vector<double>& x, standardTwoStageParameters_QP& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap);
feasibilityCut twoStageQP_feasibilityCutGeneration(const dualMultipliers_QP& extremeRay, standardTwoStageParameters_QP& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap);


// projection for the first stage in the two stage linear programming
std::vector<double> twoStageQP_projection(const std::vector<double>& x, standardTwoStageParameters_QP& model_parameters);


// presolve
std::vector<double> twoStageQP_presolve(standardTwoStageParameters_QP& model_parameters, const secondStageRHSpoint& rhs, const secondStageRHSmap& RHSmap);

// incumbent selection
bool incumbent_selection_check_QP(double q, const std::vector<double>& x_candidate, const std::vector<double>& x_incumbent, standardTwoStageParameters_QP& model_parameters, const std::vector<minorant>& minorants, const std::vector<minorant>& minorants_new, const std::vector<int>& active_minorants);

// NSD QP solver with presolve and unique duals (initial k will be based on the presolve)
std::vector<double> dynamic_sdknn_qq_solver_presolve_fullFace(const std::string& folder_path, int max_iterations, double f_upperbound, double f_lowerbound, double sigma_upperbound, double sigma_lowerbound, const std::vector<double>& observed_predictor, int N_pre, bool flag_debug);
#endif /* NSD_QQ_solver_hpp */
