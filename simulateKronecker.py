import numpy as np
from qsearch.qtools import *
from qsearch.qwalker import Qwalker
import networkx as nx
import matplotlib.pyplot as plt


def plotEquallyEvolvingVertices():

    # Plots the equally evolving vertices of the
    # the second order Kroncker graphs with complete graph
    # initiator matrix.  

    c = {0:"red", 1:"green", 2:"blue"}

    fig = plt.figure(figsize=(20, 6))

    for k in range(3, 6):
        G = nx.complete_graph(k)
        K = makeKronGraph(G, 2)
        mv = createMarkedVertex(len(K), 0)
        A = (nx.adjacency_matrix(K).todense() - np.outer(mv, mv))
        walker = Qwalker(K)
        c_list = equallyEvolvingVertices(walker)
        cm = []
        for i in c_list:
            cm.append(c[i])
        ax = fig.add_subplot(1,3, k-2)
        nx.draw_circular(K, node_color=cm, node_size = 400)

    plt.show()


if __name__ == "__main__":
    import sys

    plotEquallyEvolvingVertices()
