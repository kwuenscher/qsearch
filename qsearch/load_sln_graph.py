import networkx as nx
import numpy as np
import glob
import os, os.path
import math
import pickle

########################################################
#
# Adapted from https://github.com/jcatw/snap-facebook
#

class SnapToNetworkX():

    def __init__(self, source_path):

        self.source_path = source_path
        self.feature_index = {}
        self.inv_feature_index = {}
        self.network = nx.Graph()
        self.ego_nodes = []
        self.feature_count = 0
        self.name = os.path.split(source_path)[-1]
        self.feat_file = os.path.dirname(self.source_path) + "/" + self.name + "_feature_map.txt"

        print("------ Creating {} Graph ------".format(self.name))
        self.createNetwork()

    def parse_featname_line(self, line):

        try:
            line = line[(line.find(' '))+1:]  # chop first field
            split = line.split(';')
            name = ';'.join(split[:-1]) # feature name
            index = int(split[-1].split(" ")[-1]) #feature index
            return index, name

        except ValueError:
            print("Something went wrong while parsing.")
            line = line[(line.find(' '))+1:]  # chop first field
            split = line.rstrip()
            name = split
            index = self.feature_count
            self.feature_count +=1
            return index, name


    def load_features(self):
        # may need to build the index first
        if not os.path.exists(self.feat_file):
            feat_index = {}
            # build the index from data/*.featnames files
            featname_files = glob.iglob(self.source_path + "/*.featnames")

            for featname_file_name in featname_files:
                featname_file = open(featname_file_name, 'r')
                for line in featname_file:
                    # example line:
                    # 0 birthday;anonymized feature 376
                    if line != ' ':
                        index, name = self.parse_featname_line(line)
                        feat_index[index] = name
                featname_file.close()
            keys = list(feat_index.keys())
            keys.sort()
            out = open(self.feat_file,'w')
            for key in keys:
                out.write("%d %s\n" % (key, feat_index[key]))
            out.close()

        # index built, read it in (even if we just built it by scanning)

        index_file = open(self.feat_file,'r')
        for line in index_file:
            s = line.strip().split(' ')
            key = int(s[0])
            val = s[1]
            self.feature_index[key] = val
        index_file.close()

        for key in self.feature_index.keys():
            val = self.feature_index[key]
            self.inv_feature_index[val] = key

    def load_nodes(self):

        assert len(self.feature_index) > 0, "call load_features() first"

        # get all the node ids by looking at the files
        ego_nodes = [int(x.split("/")[-1].split('.')[0]) for x in glob.glob(self.source_path + "/*.featnames")]
        node_ids = ego_nodes

        # parse each node
        for node_id in node_ids:
            featname_file = open(self.source_path +"/%d.featnames" % (node_id), 'r')
            feat_file     = open(self.source_path +"/%d.feat"      % (node_id), 'r')
            egofeat_file  = open(self.source_path +"/%d.egofeat"   % (node_id), 'r')
            edge_file     = open(self.source_path +"/%d.edges"     % (node_id), 'r')

            # parse ego node
            self.network.add_node(node_id)
            # 0 1 0 0 0 ...
            ego_features = [int(x) for x in egofeat_file.readline().split(' ')]
            i = 0
            self.network.node[node_id]['features'] = np.zeros(len(self.feature_index))
            for line in featname_file:
                self.feature_count = 0
                key, val = self.parse_featname_line(line)
                self.network.node[node_id]['features'][key] = ego_features[i] + 1
                i += 1

            # parse neighboring nodes
            for line in feat_file:
                featname_file.seek(0)
                split = [int(x) for x in line.split(' ')]
                node_id = split[0]
                features = split[1:]
                self.network.add_node(node_id)
                self.network.node[node_id]['features'] = np.zeros(len(self.feature_index))
                i = 0
                for line in featname_file:
                    self.feature_count = 0
                    key, val = self.parse_featname_line(line)
                    self.network.node[node_id]['features'][key] = features[i]
                    i += 1

            featname_file.close()
            feat_file.close()
            egofeat_file.close()
            edge_file.close()

    def load_edges(self):

        self.network
        assert self.network.order() > 0, "call load_nodes() first"
        edge_file = open(os.path.dirname(self.source_path) +"/"+self.name +"_combined.txt","r")
        for line in edge_file:
            # nodefrom nodeto
            split = [int(x) for x in line.split(" ")]
            node_from = split[0]
            node_to = split[1]
            self.network.add_edge(node_from, node_to)

    def feature_matrix(self):
        n_nodes = self.network.number_of_nodes()
        n_features = len(self.feature_index)

        X = np.zeros((n_nodes, n_features))
        for i,node in enumerate(self.network.nodes()):
            X[i,:] = self.network.node[node]['features']

        return X

    def saveGraphToPickle(self):
        try:
            print("Saving Newtork...")
            with open("data_storage/sln_graph.pickle", 'wb') as handle:
                pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

        except:
            print("Something went wrong pickling.")

    def createNetwork(self):

        print("Load Features...")
        self.load_features()
        print("Load Nodes...")
        self.load_nodes()
        print("Load Edges...")
        self.load_edges()
        print("Pickle graph...")
        self.saveGraphToPickle()
        print("Done")


#g = SnapToNetworkX("quantum_walk/Snap-Datasets/twitter")

if __name__ == '__main__':
    import sys

    if 'debug' in sys.argv:

        g = SnapToNetworkX("quantum_walk/Snap-Datasets/facebook")

        failures = 0
        def test(actual, expected, test_name):
            global failures  #lol python scope
            try:
                print ("testing %s..." % (test_name,))
                assert actual == expected, "%s failed (%s != %s)!" % (test_name,actual, expected)
                print("%s passed (%s == %s)." % (test_name,actual,expected))
            except AssertionError as e:
                print(e)
                failures += 1

        test(g.network.order(), 4039, "order")
        test(g.network.size(), 88234, "size")
        test(round(nx.average_clustering(g.network),4), 0.6055, "clustering")
        print("%d tests failed." % (failures,))


    if 'facebook-test' in sys.argv:

        g = SnapToNetworkX("Snap-Datasets/facebook")

    if 'twitter-test' in sys.argv:

        g = SnapToNetworkX("Snap-Datasets/twitter")
