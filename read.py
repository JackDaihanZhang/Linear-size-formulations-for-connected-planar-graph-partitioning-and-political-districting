import networkx as nx
import csv

# Define a new function for drawing graphs

def read(primal_txt, dual_txt):
    primal_file = open(primal_txt, 'r').readlines()
    primal_graph = nx.DiGraph()
    dual_file = open(dual_txt, 'r').readlines()
    dual_graph = nx.DiGraph()
    primal_nodes = set([])
    dual_nodes = set([])
    for i in range(len(primal_file)):
        primal_str = primal_file[i]
        dual_str = dual_file[i]
        primal_edge = []
        for primal_node in primal_str.split():
            primal_edge.append(primal_node)
            primal_nodes.add(primal_node)
        primal_graph.add_node(i)
        primal_graph.add_nodes_from(primal_edge)
        primal_graph.add_edge(i, primal_edge[1])
        primal_graph.add_edge(i, primal_edge[0])
        dual_edge = []
        for dual_node in dual_str.split():
            dual_edge.append(dual_node)
            dual_nodes.add(dual_node)
        dual_graph.add_nodes_from(dual_edge)
        dual_graph.add_node(i)
        dual_graph.add_edge(i, dual_edge[0])
        dual_graph.add_edge(i, dual_edge[1])
        if i == 0:
            primal_roots = primal_edge
            dual_roots = dual_edge
    return [primal_file, dual_file, primal_graph, dual_graph, primal_nodes, dual_nodes, primal_roots, dual_roots]

def read_draw(primal_txt):
    primal_file = open(primal_txt, 'r').readlines()
    primal_edges = []
    tree_nodes = set([])
    primal_draw = nx.Graph()
    for i in range(len(primal_file)):
        primal_str = primal_file[i]
        primal_edge = []
        for primal_node in primal_str.split():
            primal_edge.append(int(primal_node))
            tree_nodes.add(int(primal_node))
        primal_edges.append(primal_edge)
        primal_draw.add_edge(primal_edge[0], primal_edge[1])
    return [primal_draw, primal_edges, tree_nodes]

# Read the population txt file
def read_population(population):
    pop_file = open(population, 'r').readlines()
    pop_dict = {}
    total_pop = int(pop_file[0].split()[3])
    pop_dict["total_pop"] = total_pop
    for i in range(1, len(pop_file)):
        pop_str = pop_file[i]
        splitted = pop_str.split()
        pop_dict[splitted[0]] = int(splitted[1])
    return pop_dict

# Read the distance csv file
def read_distance(distance_file, primal_graph, primal_edges):
    d = []
    # reading csv file
    with open(distance_file, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)

        # extracting each data row one by one
        for row in csvreader:
            actual_row = [int(i) for i in row[1:]]
            d.append(actual_row)
        # get rid of the first row (the header row)
        d = d[1:]
        d_dict = {}
        for edge in primal_graph.edges:
            actual_edge = primal_edges[edge[0]]
            d_dict[edge] = d[actual_edge[0] - 1][actual_edge[1] - 1]
        return d_dict

