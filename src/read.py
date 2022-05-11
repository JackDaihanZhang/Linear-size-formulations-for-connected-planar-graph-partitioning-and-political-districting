import networkx as nx

###########################
# Read the primal and dual txt file for the main graphs in Williams' model
###########################
def read_Williams(primal_dual_pairs):
    primal_graph = nx.DiGraph()
    dual_graph = nx.DiGraph()
    primal_nodes = set([])
    dual_nodes = set([])
    for i in range(len(primal_dual_pairs)):
        primal_pair = primal_dual_pairs[i][0]
        dual_pair = primal_dual_pairs[i][1]
        primal_edge = []
        for primal_node in primal_pair:
            primal_edge.append(primal_node)
            primal_nodes.add(primal_node)
        primal_graph.add_node(i)
        primal_graph.add_nodes_from(primal_edge)
        primal_graph.add_edge(i, primal_edge[1])
        primal_graph.add_edge(i, primal_edge[0])
        dual_edge = []
        for dual_node in dual_pair:
            dual_edge.append(dual_node)
            dual_nodes.add(dual_node)
        dual_graph.add_nodes_from(dual_edge)
        dual_graph.add_node(i)
        dual_graph.add_edge(i, dual_edge[0])
        dual_graph.add_edge(i, dual_edge[1])
        if i == 0:
            primal_roots = primal_edge
            dual_roots = dual_edge
    return [primal_graph, dual_graph, primal_nodes, dual_nodes, primal_roots, dual_roots]

###########################
# Read the primal txt file for Hess' model
###########################
def read_hess(primal_txt):
    primal_file = open(primal_txt, 'r').readlines()
    primal_graph = nx.Graph()
    primal_nodes = set([])
    for i in range(len(primal_file)):
        primal_str = primal_file[i]
        primal_edge = []
        nodes = primal_str.split()
        primal_graph.add_edge(int(nodes[0]), int(nodes[1]))
    return primal_graph