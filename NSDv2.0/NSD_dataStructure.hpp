//
//  NSD_dataStructure.hpp
//  NSDv2.0
//
//  Created by Shuotao Diao on 8/4/21.
//

#ifndef NSD_dataStructure_hpp
#define NSD_dataStructure_hpp

#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <ilcplex/ilocplex.h>
#include <ctime>
#include <stdlib.h>
#include <cassert>
#include <unordered_map>
#include <map>
#include <utility>

// ****************************************************
// two dimensional sparse matrix
struct sparseMatrix {
    std::map<std::pair<int, int>, double>  mat;
    long num_row = 0;
    long num_col = 0;
};

// sparse vector
struct sparseVector {
    std::unordered_map<int, double> vec;
    long num_entry = 0;
};

// ****************************************************
// Target: NSD_solver
// dual multipliers
struct dualMultipliers {
    std::vector<double> equality;
    std::vector<double> inequality;
    bool feasible_flag;
};
// feasibility cut $a^\top x \leq b$
struct feasibilityCut {
    std::vector<double> A_newRow;
    double b_newRow;
};

// ****************************************************
// Target: NSD_QQ_solver
// dual multipliers
struct dualMultipliers_QP {
    std::vector<double> s;
    std::vector<double> t;
    std::vector<double> r; // used when directly solving the dual QP
    double obj_val = 0;
    bool feasible_flag;
};
// faces
struct face {
    std::vector<int> axis; // store indices where the components of s are 0, e.g.,
                            // axis = [0,3], then s[0] = 0, s[3] = 0
};

// ****************************************************
// Target: NSD_solver and NSD_ioModel
// data structure used in model setup
// data structure for the parameters in two stage linear programming
struct twoStageParameters {
    // first stage
    sparseVector c;
    sparseMatrix A;
    sparseVector b;
    // second stage
    sparseVector d;
    sparseMatrix De;
    sparseMatrix Ce;
    sparseMatrix Di;
    sparseMatrix Ci;
};
// standard two stage parameters
// A x <= b
// Dy = e - Cx
struct standardTwoStageParameters {
    // first stage
    sparseVector c;
    sparseMatrix A;
    sparseVector b;
    // second stage
    sparseVector d;
    sparseMatrix D;
    sparseMatrix C;
    sparseVector e; // right hand side deterministic part
    // extra paramters
    long num_eq = 0;
    long num_ineq = 0;
};
// standard two stage parameters for SQQP
// A x <= b
// Dy = e - Cx
struct standardTwoStageParameters_QP {
    // first stage
    sparseMatrix Q;
    sparseVector c;
    sparseMatrix A;
    sparseVector b;
    // second stage
    sparseMatrix P;
    sparseVector d;
    sparseMatrix D;
    sparseMatrix C;
    sparseVector e; // right hand side deterministic part
    // inverse
    sparseMatrix P_inv;
    // transpose
    sparseMatrix D_trans;
    sparseMatrix C_trans;
    // extra paramters
    int num_eq = 0;
    int num_ineq = 0;
};

// ****************************************************
// Target: ioNDB
// data structures for the dataPoint
struct dataPoint { // definition of dataPoint
    std::vector<double> predictor;
    std::vector<double> response;
    double weight;
    //int order = -1;
    //double distance;
    // default constructor
    //dataPoint();
    // copy constructor
    //dataPoint(const dataPoint& targetPoint);
    // assignment
    //dataPoint operator=(const dataPoint& targetPoint);
};

// ****************************************************
// Target:ioStochastic
// data structure
struct randomVector {
    std::vector<double> component; // all the entries of a random vector
    std::vector<int> randomIndices; // indices of random entries
};

struct randomScalar {
    double component = 0;
    bool flag_random = false; // flag which tells whether this scalar is random
};

// vectors on the right hand side of second stage problem
struct secondStageRHS {
    randomVector be;
    randomVector bi;
};
// database of vectors on the right hand side of second stage (new)
struct secondStageRHSDB {
    std::vector<std::vector<std::vector<double>>> be_database;
    std::vector<std::vector<std::vector<double>>> bi_database;
    std::vector<std::vector<std::vector<double>>> Ce_database;
    std::vector<std::vector<double>> weight_database;
};

struct secondStageRHSpoint {
    std::vector<double> be;
    std::vector<double> bi;
    std::vector<double> Ce;
    std::vector<double> Ci;
    std::vector<double> predictor;
};

// store the location of randomness 
struct secondStageRHSmap {
    std::vector<int> be_map;
    std::vector<int> bi_map;
    std::vector<std::pair<int,int>> Ce_map;
    std::vector<std::pair<int,int>> Ci_map;
};

// ****************************************************
// Target: NSD_solver
struct validationResult {
    double mean;
    double variance;
    const double alpha = 95;
    const double Zalpha = 1.96;
    double CI_lower;
    double CI_upper;
    int num_dataPoint;
};


// ****************************************************
// Target: NSD_solver
// minorant
struct minorant {
    double alpha = 0;
    std::vector<double> beta;
    bool if_active = true;
};

// matrix operations in sparse matrix and sparse vector
// matrix multiplication
sparseMatrix operator*(sparseMatrix mat1, sparseMatrix mat2);
// vector multiplication
double operator*(sparseVector vec1, sparseVector vec2);
double operator*(sparseVector vec1, const std::vector<double>& vec2);
double operator*(const std::vector<double>& vec1, const std::vector<double>& vec2);
// matrix times vector
sparseVector operator*(sparseMatrix mat1, sparseVector vec1);
std::vector<double> operator*(sparseMatrix& mat1, const std::vector<double>& vec1);
sparseVector operator*(sparseVector vec1, sparseMatrix mat1);
std::vector<double> operator*(std::vector<double> vec1, sparseMatrix mat1);

// scalar times sparse vector
sparseVector operator*(double a, sparseVector vec1);
std::vector<double> operator*(double a, const std::vector<double>& vec1);

// matrix plus
sparseMatrix operator+(sparseMatrix mat1, sparseMatrix mat2);
sparseVector operator+(sparseVector vec1, sparseVector vec2);
std::vector<double> operator+(const std::vector<double>& vec1, const std::vector<double>& vec2);
sparseVector operator+(sparseVector vec1, double a);
std::vector<double> operator+(const std::vector<double>& vec1, double a);
// sparse vector + vector
sparseVector operator+(sparseVector vec1, const std::vector<double>& vec2);

// vector minus
std::vector<double> operator-(std::vector<double> vec1, std::vector<double> vec2);

// matrix negate
sparseMatrix negate(sparseMatrix& mat1);
// vector nagate
sparseVector negate(sparseVector& vec1);
std::vector<double> negate(const std::vector<double>& vec1);

// matrix transpose
sparseMatrix transpose(sparseMatrix& mat1);

// element sum
double element_sum(sparseMatrix& mat1);

// comparsion
// need precision, otherwise 1.0000001 is different from 1.0
bool if_equal(sparseVector& vec1, sparseVector& vec2);
bool if_equal(const std::vector<double>& vec1, const std::vector<double>& vec2);

// convert to sparse matrix
sparseMatrix convert2sparseMatrix(const std::vector<std::vector<double>>& input_matrix);

// convert to sparse vector
sparseVector convert2sparseVector(const std::vector<double>& input_vector);

// utility functions
void print(sparseMatrix& mat1);
void print(sparseVector& mat2);

// other supplemental functions
double max(double x, double y);
int max(int x, int y);
long max(long x, long y);
double min(double x, double y);

// test functions
void test_sparse_vector();
void test_sparse_matrix();
void test_sparse_matrix_multiplication();
void test_sparse_vector_negate();

#endif /* NSD_dataStructure_hpp */
