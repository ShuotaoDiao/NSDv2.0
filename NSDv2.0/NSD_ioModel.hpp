//
//  NSD_ioModel.hpp
//  NSDv2.0
//
//  Created by Shuotao Diao on 8/4/21.
//

#ifndef NSD_ioModel_hpp
#define NSD_ioModel_hpp

#include <stdio.h>

#include "NSD_dataStructure.hpp"
#include "NSD_matrix_operation.hpp"

// convert into standard form
/* |De 0| |y| +  |Ce| x = |be|
   |Di I| |s|    |Ci|     |bi|
 y,s >=0
 */
/* D = |De 0|
       |Di I|
 */
std::vector<std::vector<double>> standard_D(const std::vector<std::vector<double>>& De, const std::vector<std::vector<double>>& Di);
/* C = |Ce|
       |Ci|
 */
std::vector<std::vector<double>> standard_C(const std::vector<std::vector<double>>& Ce,
                                            const std::vector<std::vector<double>>& Ci);
/* e = |be|
       |bi|
 */
std::vector<double> standard_e(const std::vector<double>& be, const std::vector<double>& bi);
// standard model parameters
standardTwoStageParameters readStandardTwoStageParameters(const std::string& parameterPath);
standardTwoStageParameters_QP readStandardTwoStageParameters_QP(const std::string& parameterPath); // currnetly, nonstandard QP can not be considered, since P after introducing slack variable is not positive definite

#endif /* NSD_ioModel_hpp */
