//
//  NSD_matrix_operation.cpp
//  NSDv2.0
//
//  Created by Shuotao Diao on 8/4/21.
//

#include "NSD_matrix_operation.hpp"

int LUPDecompose(std::vector<std::vector<double>>& A, int N, double Tol, std::vector<int>& P){

    int i, j, k, imax;
    double maxA, absA;
    std::vector<double> ptr;
    
    for (i = 0; i <= N; i++)
        P[i] = i; //Unit permutation matrix, P[N] initialized with N

   for (i = 0; i < N; i++) {
       maxA = 0.0;
       imax = i;

       for (k = i; k < N; k++)
           if ((absA = std::abs(A[k][i])) > maxA) {
               maxA = absA;
               imax = k;
           }

       if (maxA < Tol) return 0; //failure, matrix is degenerate

       if (imax != i) {
           //pivoting P
           j = P[i];
           P[i] = P[imax];
           P[imax] = j;

           //pivoting rows of A
           ptr = A[i];
           A[i] = A[imax];
           A[imax] = ptr;

           //counting pivots starting from N (for determinant)
           P[N]++;
       }

       for (j = i + 1; j < N; j++) {
           A[j][i] /= A[i][i];

           for (k = i + 1; k < N; k++)
               A[j][k] -= A[j][i] * A[i][k];
       }
   }

   return 1;  //decomposition done
}

std::vector<std::vector<double>> LUPInvert(std::vector<std::vector<double>>& A, std::vector<int>& P, int N) {
    std::vector<std::vector<double>> IA;
    for (int i = 0; i < N; ++i) {
        std::vector<double> IA_row(N,0.0);
        IA.push_back(IA_row);
    }
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            IA[i][j] = P[i] == j ? 1.0 : 0.0;

            for (int k = 0; k < i; k++)
                IA[i][j] -= A[i][k] * IA[k][j];
        }

        for (int i = N - 1; i >= 0; i--) {
            for (int k = i + 1; k < N; k++)
                IA[i][j] -= A[i][k] * IA[k][j];

            IA[i][j] /= A[i][i];
        }
    }
    return  IA;
}
