import numpy as np
import networkx as nx
import community.community_louvain as community
from matplotlib import pyplot as plt

from hipo_rank.similarities.cos import CosSimilarity


# Abhiroop's code
class LouvainClustering():
  def __init__(
      self,
      n_clusters : int,
      verbose: bool = False
      ):
    self.verbose = verbose
    self.forward_sent_to_sent_weight = 1
    self.backward_sent_to_sent_weight = 2
    self.u=0.6

  def _build_graph(self, embeds, graph_centrality=False):  
    G = nx.Graph()
    nodes =[]
    edges =[]
    Similarity = CosSimilarity()
    edge_sims,pids = Similarity._get_pairwise_similarities(embeds)

    num_nodes = len(embeds)

    for ((i,j),edge_weight) in zip(pids,edge_sims):
        nodes.append(i)
        edges.append((i,j,{"weight": edge_weight}))

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    if graph_centrality:
        centrality = nx.eigenvector_centrality(G ,max_iter= 1000, tol= 1e-03, nstart= None, weight= "weight")
        for k in centrality.keys():
          G.add_edges_from([(i,i,{"weight": centrality[i]})])
    return G
  
  def _plot_graph(self,G):
    pos=nx.spring_layout(G)
    nx.draw(G,pos)
    labels = nx.get_edge_attributes(G,'weight')
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    plt.savefig("Graph.png")

  def fit_predict(self, embeds ,debug=False):
    G = self._build_graph(embeds)
    if debug:
      print(" Plotting graph")
      print("nodes", G.nodes())
      print("edge_weight", nx.get_edge_attributes(G,'weight'))
      self._plot_graph(G)
    #Louvain
    pred_partition = community.best_partition(G, weight='weight') #
    #communities = self.find_communities(G)
    #print("communities", communities)
    # Fast Modularity
    # pred_partition = {}
    # communities = sorted(nx.algorithms.community.greedy_modularity_communities(G), key=len, reverse=True)
    # for c, v_c in enumerate(communities):
    #   for v in v_c:
    #       pred_partition[v]= c + 1
    # module_ids = list(set(pred_partition.values()))
    # part_list = [] #list of list of nodes in each group of the partition
    # for i in module_ids:
    #     part_list.append([k for k,v in pred_partition.items() if v==i])
    # modularity = nx.algorithms.community.modularity(G, part_list)
    # print("modularity", modularity)
    # print("partition", set(pred_partition.values()) )
    # print("partition", len (list(pred_partition.values()) ))
    return np.array(list(pred_partition.values()))
    #return communities 

  def _plot(self, G, partition):
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title("Draw networkx graph with louvain communities")
    plt.show()
