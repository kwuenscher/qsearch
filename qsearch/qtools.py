import numpy as np
import math
import random
#from Tests.tests import *
import pickle
import time
import datetime
import networkx as nx
from qsearch.qwalker import Qwalker
from collections import defaultdict


class QSearchSession(object):

    def __init__(self, time_steps, lmnda_range, graph_meta):
        self.session = datetime.datetime.now()
        self.time_steps = time_steps
        self.lmnda_range = lmnda_range
        self.success_prob = []
        self.grah_meta = graph_meta

    def addData(self, data):
        self.success_prob.append(data)

def walk(walker, h, time_steps, res, sparse = False):

    adj_matrix = walker.getAdjMatrix()
    adj_matrix = -h * adj_matrix

    [num_rows, num_cols] = adj_matrix.shape
    init_condition = [1/np.sqrt(num_rows)]*num_rows

    eig_values, eig_proj, eig_vec = walker.specdecomp(adj_matrix, sparse=sparse)

    time_line = np.arange(0, time_steps, res)

    probabilities = []

    for t in time_line:
        u_hat = walker.unitary(eig_values, eig_proj, t)

        state_vector = u_hat @ init_condition
        prob = (state_vector * state_vector.conjugate()).real
        probabilities.append(prob)

    return probabilities

def evolve(walker, h, mv, time_steps, res, sparse = False):

    # Eovlving the system up to time t.

    # walker     - quantum walker instance
    # mv         - marked vertex
    # time_steps - specify for how many time steps
    #               the walker should run.
    # res        - resolutions of the search.

    v = np.argmax(mv)
    adj_matrix = walker.getAdjMatrix()
    adj_matrix = -h * adj_matrix - np.outer(mv, mv)

    [num_rows, num_cols] = adj_matrix.shape
    init_condition = [1/np.sqrt(num_rows)]*num_rows

    #print("Computing eigenvalue decomposition for graph with {} vertices...".format(num_rows))

    eig_values, eig_proj, eig_vec = walker.specdecomp(adj_matrix, sparse = sparse)

    # Unit-Test
    #TestWalker().testBase(adj_matrix, eig_proj)
    time_line = np.arange(0, time_steps, res)

    success_prob = []

    for t in time_line:
        u_hat = walker.unitary(eig_values, eig_proj, t)

        #TestWalker().testProbability(u_hat)
        state_vector = u_hat @ init_condition
        prob = (state_vector * state_vector.conjugate()).real

        success_prob.append(prob[v])
        #print("Simulated {} out of {} timesteps.".format(t+1, time_line[-1]+1))

    return success_prob

def optimiseGamma(walker, lambda_range, mv, res, sparse = False):

    # Naive optimisation of the hopping rate.

    time_steps = np.sqrt(len(walker.graph))
    best_lambda = 1
    best_prob = 0
    time_of_best_prob = 0

    for lam in lambda_range:
        prob = evolve(walker, lam, mv, time_steps, res)
        m = np.max(prob)
        t = np.argmax(prob)
        if m > best_prob:
            best_lambda = lam
            best_prob = m
            time_of_best_prob = t

    return best_lambda, best_prob, time_of_best_prob

def groverClustering(walker, c):

    # Create clusters according to searchability of a
    # specific node.

    g = walker.graph
    n = len(g)
    gam_range = np.arange(0, 1, 0.1)
    best_prob = []
    for v in range(0, n):
        mv = createMarkedVertex(n, v)
        best_prob.append(optimiseGamma(walker, gam_range, mv, 0.1))


    freq, bins = np.histogram(best_prob, bins = c)
    pos = np.digitize(best_prob, bins)

    return pos

def makeDeterministicAdj(G):

    # Takes a weighted adjacency matrix and returns a binary adjacency matrix.

    A = np.array(nx.adjacency_matrix(G).todense())
    m = np.full([len(nx.nodes(G)), len(nx.nodes(G))], 0, dtype="float")
    for row, i in enumerate(A):
        for col, j in enumerate(i):
            rn = np.random.uniform()
            if rn < j:
                m[row, col] = 1

    np.fill_diagonal(m, 0)
    G = nx.from_numpy_matrix(m)
    G.remove_nodes_from(nx.isolates(G))
    G = list(nx.connected_component_subgraphs(G))
    g_sizes = [len(nx.nodes(g)) for g in G]
    largest = np.argsort(g_sizes)

    if len(largest) > 1:
        return G[largest[-1]]
    else:
        return G[largest[0]]

def getLargestSubgraph(G):

    # If the graph is not connected returns the largest subgraph.

    G = list(nx.connected_component_subgraphs(G))
    g_sizes = [len(nx.nodes(g)) for g in G]
    largest = np.argsort(g_sizes)

    if len(largest) > 1:
        return G[largest[-1]]
    else:
        return G[largest[0]]

def equallyEvolvingVertices(walker, sparse= False):

    # Input: Qwalker object
    # Returns: List of lables indicating which
    #           vertices evolve equally.

    N = len(nx.nodes(walker.graph))

    mv = createMarkedVertex(N, 0)
    adj_matrix = walker.getAdjMatrix()
    A = adj_matrix - np.outer(mv, mv)
    oracle = nx.from_numpy_matrix(A)
    gam = np.linalg.eigh(A)[0][-1]
    walker = Qwalker(oracle)


    p = walk(walker,1/gam , 10, 0.1, sparse=sparse)
    s = defaultdict(list)

    edge_list = nx.edges(walker.graph)

    edge = defaultdict(int)
    for i, v in enumerate(p[-1]):
        s[round(v, 10)].append(i)
        edge[round(v, 10)] +=1

    c = [s[vals] for vals in s.keys()]

    c_list = np.arange(len(nx.nodes(walker.graph)))
    for i, sn in enumerate(c):
        for n in sn:
            c_list[n] = i

    return c_list

def rmfLoops(G, nil = True):

    # Removes or inserts selfloops.

    adj = nx.adjacency_matrix(G).todense()
    if nil == True:
        np.fill_diagonal(adj, 0)
    else:
        np.fill_diagonal(adj, 1)
    return nx.from_numpy_matrix(adj)


def loadG6File(file_path):
    gs = nx.read_graph6(file_path)
    return gs

def getMaxSuccessProb(success_prob):
    t = np.argmax(success_prob)
    prob = success_prob[t]
    return [t, prob]

def getMinSuccessProb(success_prob):
    t = np.argmin(success_prob)
    prob = success_prob[t]
    return [t, prob]

def qSave(data, file_path):
    current_data = qRestore(file_path)
    if len(current_data) == 0:
        current_data = list([data])
    else:
        current_data.append(data)
    with open(file_path, 'wb') as handle:
        pickle.dump(current_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved data to file.")

def qRestore(file_path):

    # Restores previously saved data

    try:
        with open(file_path, 'rb') as handle:
            return pickle.load(handle)

    # If no data exists return empty list.

    except EOFError:
        return []

def qFlush(file_path):

    with open(file_path, 'wb') as handle:
        pickle.dump(list([]), handle, protocol=pickle.HIGHEST_PROTOCOL)

def createMarkedVertex(n, vertex):
    mv = [0] * n
    mv[vertex] = 1
    return mv

def sampleMarkedVertex(n):
    vertex = random.sample(list(np.arange(0, n)), 1)[0]
    return createMarkedVertex(n, vertex)

def getDegreeMatrix(G):
    return nx.laplacian_matrix(G).todense() + nx.adjacency_matrix(G).todense()

def makeKronGraph(G, size, ext_initator = None, adj_matrix = False):

    # Generates k'th order Kronecker graph from any base graph.

    if ext_initator == None:
        init_a = nx.adjacency_matrix(G).todense()
        inter_a = init_a

    else:
        init_a = nx.adjacency_matrix(ext_initator).todense()
        inter_a = nx.adjacency_matrix(G).todense()

    if size == 1:
        inter_a = init_a
    else:
        for it in range(size-1):
            inter_a = np.kron(inter_a, init_a)

    # For large graphs using networkx's class function to return the adjacency
    # matrix becomes intractable. Hence, its recommended to just return the
    # raw adjacency matrix of the graph.

    if adj_matrix == False:
        g = nx.from_numpy_matrix(inter_a)
        if not nx.is_connected(g):
            print("Warnign: Graph is not connected.")
    else:
        g = inter_a
    return g


if __name__ == '__main__':
    import sys

    if 'test' in sys.argv:
        print("Testing")
