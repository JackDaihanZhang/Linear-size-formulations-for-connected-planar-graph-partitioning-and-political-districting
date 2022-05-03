import networkx as nx
import csv

# Define a new function for drawing graphs

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

"""
###########################
# Read the primal and dual txt file for the graphs used for plotting in Williams' model
###########################
def read_draw(primal_dual_pairs):
    primal_edges = []
    dual_edges = []
    tree_nodes = set([])
    primal_draw = nx.Graph()
    dual_draw = nx.MultiGraph()
    for i in range(len(primal_file)):
        primal_str = primal_file[i]
        primal_edge = []
        nodes = primal_str.split()
        for primal_node in nodes:
            primal_edge.append(int(primal_node))
            tree_nodes.add(int(primal_node))
        primal_edges.append(primal_edge)
        primal_draw.add_edge(primal_edge[0], primal_edge[1])
        dual_str = dual_file[i]
        dual_edge = []
        dual_str = dual_str.split()
        for dual_node in dual_str:
            dual_edge.append(dual_node)
        dual_edges.append(dual_edge)
        dual_draw.add_edge(dual_edge[0], dual_edge[1])

    return [primal_draw, dual_draw, primal_edges, dual_edges, tree_nodes]
"""

"""
###########################
# Read the population txt file
###########################
def read_population(population, model):
    pop_file = open(population, 'r').readlines()
    pop_dict = {}
    total_pop = int(pop_file[0].split()[2])
    if model == "Hess":
        pop_dict[-1] = total_pop
    else:
        pop_dict["total_pop"] = total_pop
    total_pop_added = 0
    for i in range(1, len(pop_file)):
        pop_str = pop_file[i]
        splitted = pop_str.split()
        if model == "Hess":
            pop_dict[int(splitted[0])] = int(splitted[1])
        else:
            pop_dict[splitted[0]] = int(splitted[1])
        total_pop_added = total_pop_added + int(splitted[1])
    if total_pop_added != total_pop:
        print("Population not adding up")
    return pop_dict
"""

"""
###########################
# Run the distance csv file
###########################
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
"""

"""
###########################
# Read the face file
###########################
def read_face(face_txt):
    faces = []
    face_file = open(face_txt, 'r').readlines()
    for i in range(len(face_file)):
        face = []
        face_str = face_file[i]
        if i == len(face_file) -1:
            face_peeled = face_str[1:len(face_str) - 1]
        else:
            face_peeled = face_str[1:len(face_str)-2]
        for node in face_peeled.split(','):
            node_strip = node.strip()
            face.append(node_strip)
        faces.append(face)
    return faces
"""

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