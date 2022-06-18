//
//  NSD_ioStochastic.hpp
//  NSDv2.0
//
//  Created by Shuotao Diao on 8/4/21.
//

#ifndef NSD_ioStochastic_hpp
#define NSD_ioStochastic_hpp

#include <stdio.h>

#include "NSD_dataStructure.hpp"

// including be, bi, Ce and Ci
secondStageRHSmap readStochasticMap(const std::string& stochasticPath);

// merge randomVector
secondStageRHSpoint merge_randomVector(const dataPoint& be_point, const dataPoint& bi_point, const dataPoint& Ce_point, const dataPoint& Ci_point);

#endif /* NSD_ioStochastic_hpp */
