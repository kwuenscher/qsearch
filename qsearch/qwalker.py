import numpy as np
import math
import networkx as nx
from scipy.sparse.linalg import expm, eigsh
from collections import defaultdict
import matplotlib.pyplot as plt

class Qwalker(object):

    def __init__(self, graph):

        self.graph = graph

    def getFactors(self):

        return self.eigenvalues, self.eigenprojectors, self.eigenvectors

    def specdecomp(self, A, sparse = False):

        '''

        For sparse matrices Lanczos algorithm is used in order to compute
        the invariant subspace. WARNIN: Lanczos algorithm is not stable!

        '''

        if sparse == True:
            [eigenvalue_list, row_eigenmatrix] = eigsh(np.array(A, dtype="float"))#, k=int(np.ceil(len(A)*0.01)))
        else:
            [eigenvalue_list, row_eigenmatrix] = np.linalg.eigh(np.array(A, dtype="float"))
        [num_rows, num_cols] = row_eigenmatrix.shape #A.shape

        eigenmatrix = np.asmatrix(row_eigenmatrix)
        eigenmatrix = eigenmatrix.H
        eigen_vec = []

        eigenvalues = []
        eigenprojectors = []
        for i in range(num_cols):
            found = False
            for j in range(len(eigenvalues)):
                if abs(eigenvalue_list[i] - eigenvalues[j]) < 0.0001:
                    v = eigenmatrix[i].H
                    #eigen_vec[j] += eigenmatrix[i]
                    eigenprojectors[j] += (v * v.H)
                    found = True

            if found == False:
                eigenvalues.append(eigenvalue_list[i])
                eigen_vec.append(eigenmatrix[i])
                v = eigenmatrix[i].H
                eigenprojectors.append(v * v.H)

        return np.asarray(eigenvalues).astype(np.float64), np.asarray(eigenprojectors), np.asarray(eigen_vec).astype(np.float64)

    def unitary(self, eig_values, eig_proj, t):

        '''

        Using Sylvester's formular to carry out matrix exponentiation.

        '''
        n_rows = len(eig_proj)
        n_col = len(eig_proj)


        u_hat = np.array(np.zeros((n_rows, n_col)), dtype = complex)

        u_hat = np.sum([(np.exp(-1j * complex(t * eig_values[j])) * eig_proj[j].astype(complex)) for j in range(n_col)],  axis=0)

        return u_hat


    def unitary2(self, A, t):

        '''

        Using Pade approximation to compute the unitary. (slow

        '''

        [num_rows, num_cols] = A.shape
        U = (np.zeros(shape = (num_rows, num_cols))).astype(complex)
        U = expm(-1j * A.astype(complex) * t)
        return U

    def getEigenstateOverlap(self, mv, opt_lamb):

        '''

        Computes the overlap of the smallest two eigenstates i.e. the marked vertex
        and the inital superpostion.

        '''

        adj_matrix = np.array(nx.adjacency_matrix(self.graph).todense())
        [num_rows, num_cols] = adj_matrix.shape

        gamma_range = np.arange(0.001, 2*opt_lamb, 0.0001)

        s = [1/np.sqrt(num_rows)]*num_rows

        mv_overlaps = defaultdict(list)
        s_overlaps = defaultdict(list)

        for gamma in gamma_range:

            m = -gamma * adj_matrix - np.outer(mv, mv)
            eig_values, eig_proj, eig_vectors = self.specdecomp(m)

            for i, v in enumerate(eig_vectors[:2]):
                mv_overlaps[i].append(abs((np.inner(v, mv))**2)[0])
                s_overlaps[i].append(abs((np.inner(v, s))**2)[0])

        gamma_N = gamma_range/opt_lamb

        return mv_overlaps

    def getAdjMatrix(self):
        return np.array(nx.adjacency_matrix(self.graph).todense())

    def getLaplacianMatrix(self):
        return np.array(nx.laplacian_matrix(self.graph).todense())

    def updateGraph(self, newGraph):
        self.graph = newGraph
