
import numpy as np
import sys
import networkx as nx
from qsearch.qtools import *


def plotSuccessLambda(x, y):

    app = QtGui.QApplication(sys.argv)
    win = pg.GraphicsWindow(title="Quantum Walk")
    win.resize(1000, 1000)

    p = win.addPlot(title="Success Probability vs Hopping Rate")

    p.plot(x = x, y = y)
    status = app.exec_()
    sys.exit(status)

def getDegreeSequence(G):

    # Returning the degree Sequence.

    degree_sequence=sorted(nx.degree(w).values(),reverse=True) # degree sequence

    return max(degree_sequence)

def convert(A):

    # Create networkx graph from numpy adjacency matrix.

    g = nx.to_networkx_graph(A)

    return g

def deleteSelfLoops(graph, nNodes):

    # used to take away self loops in final graph for stat purposes

    nNodes = int(nNodes)
    for i in range(nNodes):
        for j in range(nNodes):
            if(i ==  j):
                graph[i, j] = 0
    return graph

def waveLengthToRGB(wavelength):
    factor = 0
    Red = Green = Blue = 0

    if wavelength >= 380 and wavelength<440:
        Red = -(wavelength - 440) / (440 - 380)
        Green = 0.0
        Blue = 1.0
    elif ((wavelength >= 440) and (wavelength<490)):
        Red = 0.0
        Green = (wavelength - 440) / (490 - 440)
        Blue = 1.0
    elif((wavelength >= 490) and (wavelength<510)):
        Red = 0.0
        Green = 1.0
        Blue = -(wavelength - 510) / (510 - 490)
    elif((wavelength >= 510) and (wavelength<580)):
        Red = (wavelength - 510) / (580 - 510)
        Green = 1.0
        Blue = 0.0
    elif((wavelength >= 580) and (wavelength<645)):
        Red = 1.0
        Green = -(wavelength - 645) / (645 - 580)
        Blue = 0.0
    elif((wavelength >= 645) and (wavelength<781)):
        Red = 1.0
        Green = 0.0
        Blue = 0.0
    else:
        Red = 0.0
        Green = 0.0
        Blue = 0.0

    return Red, Green, Blue

def eigToWaveLength(eigs, n):
    wave = 400 + 300 * np.array(eigs)/n
    return wave

def generateStochasticKron(initMat, k, deleteSelfLoopsForStats=False, directed=False, customEdges=False, edges=0):
    initN = len(initMat)
    nNodes = int(math.pow(initN, k))
    mtxDim = len(initMat)
    mtxSum = np.sum(initMat)

    if(customEdges == True):
        nEdges = edges
        if(nEdges > (nNodes*nNodes)):
            raise ValueError("More edges than possible with number of Nodes")
    else:
        nEdges = math.pow(mtxSum, k)
    collisions = 0

    probToRCPosV = []
    cumProb = 0.0

    for i in range(mtxDim):
        for j in range(mtxDim):
            prob = initMat[i, j]
            if(prob > 0.0):
                cumProb += prob
                probToRCPosV.append((cumProb/mtxSum, i, j))

    finalGraph = np.zeros((nNodes, nNodes))

    e = 0

    while(e < nEdges):
        rng = nNodes
        row = 0
        col = 0
        for t in range(k):
            prob = random.uniform(0, 1)
            n = 0
            while(prob > probToRCPosV[n][0]):
                n += 1
            mrow = probToRCPosV[n][1]
            mcol = probToRCPosV[n][2]
            rng /= mtxDim
            row += int(mrow * rng)
            col += int(mcol * rng)

        if(finalGraph[row, col] == 0):
            finalGraph[row, col] = 1
            e += 1
            if(not directed):
                if(row != col):
                    finalGraph[col, row] = 1
                    e += 1
        else:
            collisions += 1

    if(deleteSelfLoopsForStats):
        finalGraph = deleteSelfLoops(finalGraph, nNodes)
    finalGraph = convert(finalGraph)
    return finalGraph
