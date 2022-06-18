//
//  NSD_matrix_operation.hpp
//  NSDv2.0
//
//  Created by Shuotao Diao on 8/4/21.
//

#ifndef NSD_matrix_operation_hpp
#define NSD_matrix_operation_hpp

#include <stdio.h>
#include <cmath>
#include <vector>

// LUP decomposition source code website:https://en.wikipedia.org/wiki/LU_decomposition#C_code_example

/* INPUT: A - array of pointers to rows of a square matrix having dimension N
 *        Tol - small tolerance number to detect failure when the matrix is near degenerate
 * OUTPUT: Matrix A is changed, it contains a copy of both matrices L-E and U as A=(L-E)+U such that P*A=L*U.
 *        The permutation matrix is not stored as a matrix, but in an integer vector P of size N+1
 *        containing column indexes where the permutation matrix has "1". The last element P[N]=S+N,
 *        where S is the number of row exchanges needed for determinant computation, det(P)=(-1)^S
 */
// Note: E is identity matrix, since all the diagonal terms in L are set to be 1
int LUPDecompose(std::vector<std::vector<double>>& A, int N, double Tol, std::vector<int>& P);

/* INPUT: A,P filled in LUPDecompose; N - dimension
 * OUTPUT: IA is the inverse of the initial matrix
 */
std::vector<std::vector<double>> LUPInvert(std::vector<std::vector<double>>& A, std::vector<int>& P, int N);
#endif /* NSD_matrix_operation_hpp */
