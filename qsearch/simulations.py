import networkx as nx
import numpy as np
import math
from Qwalker import *
from qtools import *
import pickle
from snap import *
from queue import *
from graphFactory import *

class SimulateWattsStrogats():

    def __init__(self, graph_meta, saver_path=None):

        self.graph_meta = graph_meta
        self.saver_path = saver_path
        #self.queue = Queue(maxsize=5)

    def run(self, lambda_range, sample_size, max_time_steps, graph_size):

        validation = []
        self.sess = QSearchSession(max_time_steps, lambda_range, self.graph_meta)

        qFlush(self.saver_path)

        for i, l in enumerate(lambda_range):

            try:
                probs = []
                for s in range(sample_size):
                    mv = sampleMarkedVertex(graph_size)
                    g = nx.watts_strogatz_graph(graph_size, self.graph_meta["neighbours"], self.graph_meta["probability"] )
                    walker = Qwalker(g, l)
                    success_prob = getSuccessProb(walker, mv, max_time_steps)
                    probs.append(findMaxSuccessProb(success_prob))
                #self.queue.put([np.mean(np.array(probs), 0)[1], l])
                self.sess.addData(np.mean(np.array(probs), 0))

            except KeyboardInterrupt:
                print("------- Simulation Interrupted -------")
                print("Saving progress...")
                self.sess.addData(np.mean(np.array(probs), 0))
                print("Done.")

            print("Iteration: {} of {}".format(i + 1, len(lambda_range)))

        optimal_lambda = lambda_range[np.argmax(np.array(self.sess.success_prob)[:, 1])]
        print(self.sess.success_prob)
        print("------- Simulation Done -------")
        print("Optimal Lambda: ", optimal_lambda)
        print("Saving Progress...")
        qSave(self.sess, self.saver_path)
        print("Done")

class SimulateFBGraph():

    def __init__(self, graph, saver_path=None):

        self.graph = graph.network
        self.saver_path = saver_path
        self.queue = Queue(maxsize=5)

    def run(self, lambda_range, sample_size, max_time_steps, graph_size):

        validation = []
        self.sess = QSearchSession(max_time_steps, lambda_range,{})

        qFlush(self.saver_path)

        for i, l in enumerate(lambda_range):

            try:
                probs = []
                for s in range(sample_size):
                    g = RNNSampling(self.graph, graph_size)
                    graph_size = np.shape(nx.adj_matrix(g).toarray())[0]
                    mv = sampleMarkedVertex(graph_size)
                    walker = Qwalker(g, l)
                    success_prob = getSuccessProb(walker, mv, max_time_steps)
                    probs.append(findMaxSuccessProb(success_prob))
                #self.queue.put([np.mean(np.array(probs), 0)[1], l])
                self.sess.addData(np.mean(np.array(probs), 0))

            except KeyboardInterrupt:
                print("------- Simulation Interrupted -------")
                print("Saving progress...")
                self.sess.addData(np.mean(np.array(probs), 0))
                print("Done.")

            print("Iteration: {} of {}".format(i + 1, len(lambda_range)))

        optimal_lambda = lambda_range[argmax(np.array(sess.success_prob)[:, 1])]

        print("------- Simulation Done -------")
        print("Best Lambda: ", optimal_lambda)
        print("Saving Progress...")
        qSave(self.sess, self.saver_path)
        print("Done")

class OptimiseGraphs():

    def __init__(self, graphs, saver_path):
        self.graphs = graphs
        self.saver_path = saver_path
        self.n_graphs = len(graphs)


    def optimise(self, thresh):

        n_vertices = nx.number_of_nodes(self.graphs[0])
        max_t = n_vertices + 150
        lambda_range = np.arange(0, 1, 5)

        graph_probs = []
        for j, graph in enumerate(self.graphs[:2]):
            adj_matrix = nx.adj_matrix(graph).toarray()
            g = nx.watts_strogatz_graph(13, 3, 0.5 )
            probs = []
            for i, l in enumerate(lambda_range):
                mv = sampleMarkedVertex(n_vertices)

                walker = Qwalker(g, l)
                success_prob = getSuccessProb(walker, mv, max_t)
                print(success_prob)
                probs.append(findMaxSuccessProb(success_prob))

            graph_probs.append(np.mean(np.array(probs), 0))

            print("Graph {} out of {}".format(j, self.n_graphs-1))

        print(graph_probs)




# graphs = loadG6File("highlyirregular13.g6")
# graph_optimiser = OptimiseGraphs(graphs, "nothing")
# graph_optimiser.optimise(0.3)


if __name__ == '__main__':
    import sys

    if 'ws-test' in sys.argv:
        saver_path = "data_storage/swn_data.pickle"
        graph_meta = {"probability": 0.5, "neighbours": 6}
        lmnda_range = np.linspace(0.1, 1, 10)
        sim = SimulateWattsStrogats(graph_meta, saver_path=saver_path)
        sim.run(lmnda_range, sample_size = 2, max_time_steps = 50, graph_size= 15)
        #plotSuccessLambda(lmnda_range, np.array(sim.sess.success_prob)[:, 1])

    if 'fb-test' in sys.argv:
        saver_path = "data_storage/swn_data.pickle"
        fb_g = loadSnapGraph()
        lmnda_range = np.linspace(0.1, 1, 10)
        sim = SimulateFBGraph(fb_g, saver_path=saver_path)
        sim.run(lmnda_range, sample_size = 5, max_time_steps = 50, graph_size= 5)
