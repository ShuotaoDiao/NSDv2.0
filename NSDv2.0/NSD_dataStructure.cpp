//
//  NSD_dataStructure.cpp
//  NSDv2.0
//
//  Created by Shuotao Diao on 8/4/21.
//

#include "NSD_dataStructure.hpp"

// declare global variables (need to be defined in the source file)
double PRECISION_LOWER = -1e-6;
double PRECISION_UPPER = 1e-6;

sparseMatrix operator*(sparseMatrix mat1, sparseMatrix mat2) {
    // initialize sparse matrix
    sparseMatrix res;
    res.num_row = mat1.num_row;
    res.num_col = mat2.num_col;
    std::map<std::pair<int,int>, double>::iterator it_find;
    for (std::map<std::pair<int,int>, double>::iterator it = mat1.mat.begin(); it != mat1.mat.end(); ++it) {
        for (int index = 0; index < mat2.num_col; ++index) {
            double val = 0;
            it_find = mat2.mat.find(std::make_pair((it -> first).second, index));
            if (it_find != mat2.mat.end()) {
                val = (it -> second) * (it_find -> second); // calculate the product of two entries
            }
            std::pair<int, int> loc_key((it -> first).first,index);
            if (res.mat.find(loc_key) != res.mat.end()) { // location key exists
                double temp = res.mat[loc_key] + val;
                if (temp >= PRECISION_LOWER && temp <= PRECISION_UPPER) { // results after summation is significantly close to 0, then remove the entry
                    res.mat.erase(res.mat.find(loc_key));
                }
                else {
                    res.mat[loc_key] = temp;
                }
            }
            else { // location key does not exist
                res.mat[loc_key] = val;
            }
        }
    }
    return res;
}

double operator*(sparseVector vec1, sparseVector vec2) {
    double res = 0;
    std::unordered_map<int, double>::const_iterator it_find;
    for (auto it = vec1.vec.begin(); it != vec1.vec.end(); ++it) {
        it_find = vec2.vec.find(it -> first);
        if (it_find != vec2.vec.end()) {
            res += (it -> second) * (it_find -> second);
        }
    }
    return res;
}

double operator*(sparseVector vec1, const std::vector<double>& vec2) {
    double res = 0;
    if (vec1.num_entry != vec2.size()) {
        throw std::invalid_argument("Vector sizes do not match.\n");
    }
    for (auto it = vec1.vec.begin(); it != vec1.vec.end(); ++it) {
        res += (it -> second) * (vec2[it -> first]);
    }
    return res;
}

double operator*(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    double res = 0;
    if (vec1.size() != vec2.size()) {
        std::cout << "Warning: vec1.size() = " << vec1.size() << std::endl;
        std::cout << "Warning: vec2.size() = " << vec2.size() << std::endl;
        throw std::invalid_argument("Vector sizes do not match.\n");
    }
    for (int index = 0; index < vec1.size(); ++index) {
        res += vec1[index] * vec2[index];
    }
    return res;
}

// matrix times vector
sparseVector operator*(sparseMatrix mat1, sparseVector vec1) {
    sparseVector res;
    if (mat1.num_col != vec1.num_entry) {
        throw std::invalid_argument("Matrix column size does not match vector size.\n");
    }
    res.num_entry = mat1.num_row;
    std::unordered_map<int,double>::const_iterator it_find;
    for (std::map<std::pair<int,int>, double>::iterator it = mat1.mat.begin(); it != mat1.mat.end(); ++it) {
        it_find = vec1.vec.find((it -> first).second);
        if (it_find != vec1.vec.end()) {
            double val = (it -> second) * (it_find -> second);
            if (res.vec.find((it -> first).first) != res.vec.end()) {
                double temp = res.vec[(it -> first).first] + val;
                if (temp >= PRECISION_LOWER && temp <= PRECISION_UPPER) { // the sum is not siginificantly different from 0
                    res.vec.erase(res.vec.find((it -> first).first)); // remove the entry to preserve sparsity
                }
                else {
                    res.vec[(it -> first).first] = temp;
                }
            }
            else {
                res.vec[(it -> first).first] = val;
            }
        }
        
    }
    return res;
}


std::vector<double> operator*(sparseMatrix& mat1, const std::vector<double>& vec1) {
    std::vector<double> res(mat1.num_row,0.0);
    if (mat1.num_col != vec1.size()) {
        throw std::invalid_argument("Matrix column size does not match vector size.\n");
    }
    for (auto it = mat1.mat.begin(); it != mat1.mat.end(); ++it) {
        // get the location of the entry
        res[(it -> first).first] += (it -> second) * vec1[(it -> first).second];
    }
    return res;
}


// scalar times sparse vector
sparseVector operator*(double a, sparseVector vec1) {
    sparseVector res;
    res.num_entry = vec1.num_entry;
    for (auto it = vec1.vec.begin(); it != vec1.vec.end(); ++it) {
        double temp = a * (it -> second);
        if (temp < PRECISION_LOWER || temp > PRECISION_UPPER) {
            res.vec[it -> first] = temp;
        }
    }
    return res;
}
std::vector<double> operator*(double a, const std::vector<double>& vec1) {
    std::vector<double> res;
    for (int index = 0; index < vec1.size(); ++index) {
        res.push_back(a * vec1[index]);
    }
    return res;
}

// v_transpose * mat
sparseVector operator*(sparseVector vec1, sparseMatrix mat1) {
    sparseVector res;
    if (mat1.num_row != vec1.num_entry) {
        throw std::invalid_argument("Matrix column size does not match vector size.\n");
    }
    res.num_entry = mat1.num_col;
    std::unordered_map<int,double>::const_iterator it_find;
    for (std::map<std::pair<int,int>, double>::iterator it = mat1.mat.begin(); it != mat1.mat.end(); ++it) {
        it_find = vec1.vec.find((it -> first).first);
        if (it_find != vec1.vec.end()) {
            double val = (it -> second) * (it_find -> second);
            if (res.vec.find((it -> first).second) != res.vec.end()) { // entry already exists in res
                double temp = res.vec[(it -> first).second] + val;
                if (temp >= PRECISION_LOWER && temp <= PRECISION_UPPER) { // the sum is not siginificantly different from 0
                    res.vec.erase(res.vec.find((it -> first).second)); // remove the entry to preserve sparsity
                }
                else {
                    res.vec[(it -> first).second] = temp;
                }
            }
            else {
                res.vec[(it -> first).second] = val;
            }
        }
    }
    return res;
}

std::vector<double> operator*(std::vector<double> vec1, sparseMatrix mat1) {
    if (mat1.num_row != vec1.size()) {
        throw std::invalid_argument("Matrix column size does not match vector size.\n");
    }
    std::vector<double> res(mat1.num_col,0);
    for (std::map<std::pair<int,int>, double>::iterator it = mat1.mat.begin(); it != mat1.mat.end(); ++it) {
        double val = (it -> second) * vec1[(it -> first).first];
        res[(it -> first).second] += val;
    }
    return res;
}

// matrix plus
sparseMatrix operator+(sparseMatrix mat1, sparseMatrix mat2) {
    sparseMatrix res;
    std::map<std::pair<int,int>, double>::iterator it_find;
    if (mat1.num_row != mat2.num_row || mat1.num_col != mat2.num_col) {
        throw std::invalid_argument("Matrix sizes are not matched.\n");
    }
    res.num_row = mat1.num_row;
    res.num_col = mat1.num_col;
    for (std::map<std::pair<int,int>, double>::iterator it = mat1.mat.begin(); it != mat1.mat.end(); ++it) {
        it_find = mat2.mat.find(it -> first);
        if (it_find != mat2.mat.end()) { // the entries of the same position are different from 0
            double temp = (it -> second) + (it_find -> second);
            if (temp > PRECISION_UPPER || temp < PRECISION_LOWER) { // summation is significantly far from 0
                res.mat[it -> first] = temp;
            }
        }
        else {
            res.mat[it -> first] = it -> second;
        }
    }
    for (auto it2 = mat2.mat.begin(); it2 != mat1.mat.end(); ++it2) {
        it_find = mat1.mat.find(it2 -> first);
        if (it_find == mat1.mat.end()) { // the entry of the same position in matrix 1 is 0
            res.mat[it2 -> first] = it2 -> second;
        }
    }
    return res;
}


sparseVector operator+(sparseVector vec1, sparseVector vec2) {
    sparseVector res;
    if (vec1.num_entry != vec2.num_entry) {
        throw std::invalid_argument("Vector sizes are not matched.\n");
    }
    res.num_entry = vec1.num_entry;
    std::unordered_map<int,double>::const_iterator it_find;
    for (auto it = vec1.vec.begin(); it != vec1.vec.end(); ++it) {
        it_find = vec2.vec.find(it -> first);
        if (it_find != vec2.vec.end()) { // the entries of the same location are different from zero
            double temp = (it -> second) + (it_find -> second);
            if (temp < PRECISION_LOWER || temp > PRECISION_UPPER) { // if the sum is significantly different from 0
                res.vec[it -> first] = temp;
            }
        }
        else {
            res.vec[it -> first] = it -> second;
        }
    }
    for (auto it2 = vec2.vec.begin(); it2 != vec2.vec.end(); ++it2) {
        it_find = vec1.vec.find(it2 -> first);
        if (it_find == vec1.vec.end()) { // the entry of the same position in the first vector is 0
            res.vec[it2 -> first] = it2 -> second;
        }
    }
    return res;
}

std::vector<double> operator+(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vector sizes are not matched.\n");
    }
    std::vector<double> res;
    for (int index = 0; index < vec1.size(); ++index) {
        res.push_back(vec1[index] + vec2[index]);
    }
    return res;
}

sparseVector operator+(sparseVector vec1, double a) {
    sparseVector res;
    res.num_entry = vec1.num_entry;
    for (auto it = vec1.vec.begin(); it != vec1.vec.end(); ++it) {
        double temp = (it -> second) * a;
        if (temp > PRECISION_UPPER || temp < PRECISION_LOWER) {
            res.vec[it -> first] = temp;
        }
    }
    return res;
}

std::vector<double> operator+(const std::vector<double>& vec1, double a) {
    std::vector<double> res;
    for (int index = 0; index < vec1.size(); ++index) {
        res.push_back(vec1[index] + a);
    }
    return res;
}

sparseVector operator+(sparseVector vec1, const std::vector<double>& vec2) {
    sparseVector res;
    if (vec1.num_entry != vec2.size()) {
        throw std::invalid_argument("Vector sizes are not matched.\n");
    }
    res.num_entry = vec1.num_entry;
    for (auto it = vec1.vec.begin(); it != vec1.vec.end(); ++it) {
        double temp = (it -> second) + vec2[it -> first];
        if (temp < PRECISION_LOWER || temp > PRECISION_UPPER) { // if the sum is significantly different from 0
            res.vec[it -> first] = temp;
        }
    }
    std::unordered_map<int,double>::const_iterator it_find;
    for (int index = 0; index < vec2.size(); ++index) {
        it_find = vec1.vec.find(index);
        if (it_find == vec1.vec.end()) {
            if (vec2[index] < PRECISION_LOWER || vec2[index] > PRECISION_UPPER) {
                res.vec[index] = vec2[index];
            }
        }
    }
    return res;
}

// vector minus
std::vector<double> operator-(std::vector<double> vec1, std::vector<double> vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Vector sizes are not matched.\n");
    }
    std::vector<double> res;
    for (int index = 0 ; index < vec1.size(); ++index) {
        res.push_back(vec1[index] - vec2[index]);
    }
    return res;
}

// matrix negate
sparseMatrix negate(sparseMatrix& mat1) {
    sparseMatrix res;
    res.num_col = mat1.num_col;
    res.num_row = mat1.num_row;
    for (std::map<std::pair<int,int>, double>::iterator it = mat1.mat.begin(); it != mat1.mat.end(); ++it) {
        res.mat[it -> first] = (it -> second) * (-1.0);
    }
    return res;
}

// vector nagate
sparseVector negate(sparseVector& vec1) {
    sparseVector res;
    res.num_entry = vec1.num_entry;
    for (auto it = vec1.vec.begin(); it != vec1.vec.end(); ++it) {
        res.vec[it -> first] = (it -> second) * (-1.0);
    }
    return res;
}

std::vector<double> negate(const std::vector<double>& vec1) {
    std::vector<double> res;
    for (int index = 0; index < vec1.size(); ++index) {
        res.push_back(-vec1[index]);
    }
    return res;
}

// matrix transpose
sparseMatrix transpose(sparseMatrix& mat1) {
    sparseMatrix res;
    res.num_col = mat1.num_row;
    res.num_row = mat1.num_col;
    for (std::map<std::pair<int,int>, double>::iterator it = mat1.mat.begin(); it != mat1.mat.end(); ++it) {
        // get the location of the entry
        res.mat[std::make_pair((it -> first).second, (it -> first).first)] = it -> second; // assign the value
    }
    return res;
}


// element sum
double element_sum(sparseMatrix& mat1) {
    double res = 0;
    for (auto it = mat1.mat.begin(); it != mat1.mat.end(); ++it) {
        res += it -> second;
    }
    return res;
}

// comparsion
bool if_equal(sparseVector& vec1, sparseVector& vec2) {
    if (vec1.num_entry != vec2.num_entry) {
        return false;
    }
    // both are empty
    if (vec1.num_entry == 0 && vec2.num_entry == 0) {
        return true;
    }
    bool res = true;
    std::unordered_map<int,double>::const_iterator it_find;
    for (auto it = vec1.vec.begin(); it != vec1.vec.end(); ++it) {
        it_find = vec2.vec.find(it -> first);
        if (it_find == vec2.vec.end()) {
            return false;
        }
        else if ((it -> second) != (it_find -> second)) {
            return false;
        }
    }
    return res;
}
// compare two vector with certain tolerance
bool if_equal(const std::vector<double>& vec1, const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        return false;
    }
    // use L1 distance to measure the distance between two vectors 
    double diff = 0;
    for (int index = 0; index < vec1.size(); ++index) {
        diff += std::abs(vec1[index] - vec2[index]);
        if (diff > PRECISION_UPPER) { // difference is significant
            //std::cout << "Debug check if two vector are equal.\n";
            //std::cout << "index: " << index;
            //std::cout << "  diff: " << diff << std::endl;
            return false;
        }
    }
    return true;
}

// convert to sparse matrix
sparseMatrix convert2sparseMatrix(const std::vector<std::vector<double>>& input_matrix) {
    sparseMatrix res;
    res.num_row = input_matrix.size();
    if (input_matrix.size() > 0) {
        res.num_col = input_matrix[0].size();
    }
    for (int row = 0; row < res.num_row; ++row) {
        for (int col = 0; col < res.num_col; ++col) {
            if (input_matrix[row][col] > PRECISION_UPPER || input_matrix[row][col] < PRECISION_LOWER) {
                res.mat[std::make_pair(row, col)] = input_matrix[row][col];
            }
            
        }
    }
    return res;
}


// convert to sparse vector
sparseVector convert2sparseVector(const std::vector<double>& input_vector) {
    sparseVector res;
    res.num_entry = input_vector.size();
    for (int index = 0; index < input_vector.size(); ++index) {
        if (input_vector[index] > PRECISION_UPPER || input_vector[index] < PRECISION_LOWER) { // value is significantly greater than 0
            res.vec[index] = input_vector[index];
        }
    }
    return res;
}

// utility functions
void print(sparseMatrix& mat1) {
    std::cout << "Output Sparse Matrix\n";
    if (mat1.mat.size() < 1) {
        std::cout << "All elements are zero.\n";
    }
    for (std::map<std::pair<int,int>, double>::iterator it = mat1.mat.begin(); it != mat1.mat.end(); ++it) {
        std::cout << "[" << (it -> first).first << "," << (it -> first).second << "]" << " => " << it -> second << std::endl;
    }
    std::cout << "Number of rows: " << mat1.num_row << std::endl;
    std::cout << "NUmber of columns: " << mat1.num_col << std::endl;
}

void print(sparseVector& vec1) {
    std::cout << "Output Sparse Vector\n";
    if (vec1.vec.size() < 1) {
        std::cout << "All elements are zero.\n";
    }
    for (auto it = vec1.vec.begin(); it != vec1.vec.end(); ++it) {
        std::cout << it -> first << " => " << it -> second << std::endl;
    }
    std::cout << "Vector size (including 0s): " << vec1.num_entry << std::endl;
}

// other supplemental functions
double max(double x, double y){
    if (x > y){
        return x;
    }
    else {
        return y;
    }
}

int max(int x, int y){
    if (x > y){
        return x;
    }
    else {
        return y;
    }
}

long max(long x, long y){
    if (x > y){
        return x;
    }
    else {
        return y;
    }
}

double min(double x, double y) {
    if (x < y) {
        return x;
    }
    else {
        return y;
    }
}

// test functions
void test_sparse_vector() {
    std::vector<double> vec1;
    vec1.push_back(0);
    vec1.push_back(1.5e-7);
    vec1.push_back(20);
    vec1.push_back(-2);
    std::cout << "Original Vector:\n";
    for (int index = 0; index < vec1.size(); ++index) {
        std::cout << vec1[index] << std::endl;
    }
    std::cout << "Sparse Vector:\n";
    sparseVector vec2 = convert2sparseVector(vec1);
    /*
    for (std::map<int,double>::iterator it = vec2.vec.begin() ; it != vec2.vec.end(); ++it) {
        std::cout << it -> first << " => " << it -> second << std::endl;
    }
     */
    print(vec2);
}

void test_sparse_matrix() {
    std::vector<std::vector<double>> matrix1;
    std::vector<double> row1;
    row1.push_back(1);
    row1.push_back(0);
    row1.push_back(1);
    std::vector<double> row2;
    row2.push_back(-1);
    row2.push_back(0);
    row2.push_back(3);
    matrix1.push_back(row1);
    matrix1.push_back(row2);
    std::cout << "Original Matrix\n";
    for (int row = 0; row < 2; ++row) {
        std::cout << "| ";
        for (int col = 0; col < 3; ++col) {
            std::cout << matrix1[row][col] << " ";
        }
        std::cout << "|\n";
    }
    // convert to sparse matrix
    sparseMatrix matrix2 = convert2sparseMatrix(matrix1);
    std::cout << "Sparse Matrix\n";
    print(matrix2);
}

void test_sparse_matrix_multiplication() {
    std::vector<std::vector<double>> matrix1;
    std::vector<double> row1;
    row1.push_back(1);
    row1.push_back(0);
    row1.push_back(1);
    std::vector<double> row2;
    row2.push_back(-1);
    row2.push_back(0);
    row2.push_back(3);
    matrix1.push_back(row1);
    matrix1.push_back(row2);
    std::cout << "Original Matrix 1\n";
    for (int row = 0; row < 2; ++row) {
        std::cout << "| ";
        for (int col = 0; col < 3; ++col) {
            std::cout << matrix1[row][col] << " ";
        }
        std::cout << "|\n";
    }
    // matrix 2
    std::vector<std::vector<double>> matrix2;
    std::vector<double> row3;
    row3.push_back(7);
    row3.push_back(0);
    row3.push_back(-1);
    std::vector<double> row4;
    row4.push_back(0);
    row4.push_back(0);
    row4.push_back(0);
    std::vector<double> row5;
    row5.push_back(2);
    row5.push_back(0);
    row5.push_back(1);
    matrix2.push_back(row3);
    matrix2.push_back(row4);
    matrix2.push_back(row5);
    std::cout << "Original Matrix 2\n";
    for (int row = 0; row < 3; ++row) {
        std::cout << "| ";
        for (int col = 0; col < 3; ++col) {
            std::cout << matrix2[row][col] << " ";
        }
        std::cout << "|\n";
    }
    // convert to sparse matrix
    std::cout << "Print Sparse Matrices\n";
    sparseMatrix matrix1_sparse = convert2sparseMatrix(matrix1);
    sparseMatrix matrix2_sparse = convert2sparseMatrix(matrix2);
    print(matrix1_sparse);
    print(matrix2_sparse);
    // Output
    std::cout << "Output the results of sparse matrix multiplication\n";
    sparseMatrix matrix_mul = matrix1_sparse * matrix2_sparse;
    print(matrix_mul);
}

void test_sparse_vector_negate() {
    std::vector<double> vec1;
    vec1.push_back(0);
    vec1.push_back(1.5e-7);
    vec1.push_back(20);
    vec1.push_back(-2);
    std::cout << "Original Vector:\n";
    for (int index = 0; index < vec1.size(); ++index) {
        std::cout << vec1[index] << std::endl;
    }
    std::cout << "Sparse Vector:\n";
    sparseVector vec2 = convert2sparseVector(vec1);
    /*
    for (std::map<int,double>::iterator it = vec2.vec.begin() ; it != vec2.vec.end(); ++it) {
        std::cout << it -> first << " => " << it -> second << std::endl;
    }
     */
    print(vec2);
    std::cout << "Negate Sparse Vector:\n";
    sparseVector vec3 = negate(vec2);
    print(vec3);
    std::cout << "Sparse Vector + its Negate:\n";
    sparseVector vec4 = vec2 + vec3;
    print(vec4);
    std::cout << "Sparse Vector * its Negate:\n";
    double value = vec2 * vec3;
    std::cout << value << std::endl;
    if (if_equal(vec2, vec3)) {
        std::cout << "vec2 is equal to vec3\n";
    }
    else {
        std::cout << "vec2 is not equal to vec3\n";
    }
    if (if_equal(vec2, vec2)) {
        std::cout << "vec2 is equal to vec2\n";
    }
    else {
        std::cout << "vec2 is not equal to vec2\n";
    }
}
