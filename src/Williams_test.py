import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from gerrychain import Graph
import face_finder
import read


def Williams_model(k, state,df,p,primal_draw):
    # Do we deal with a forest?
    is_forest = True
    # Is population balance considered?
    is_population_considered = True
    # Add an objective function to measure compactness?
    add_objective = True
    # Add symmetry-breaking constraints if it is set to True
    symmetry_break = False
    # Add strengthening constraints if it is set to True
    strengthening = False
    # Add supervalid cuts if it is set to True
    supervalid = False

    # Construct the graphs from JSON file
    dual_draw, primal_dual_pairs = face_finder.restricted_planar_dual(primal_draw,df)
    
    print("number of crossings:", len(primal_dual_pairs))
    
    [primal_graph, dual_graph, primal_nodes, dual_nodes, primal_roots, dual_roots] = read.read_Williams(primal_dual_pairs)
    primal_edges = []
    dual_edges = []
    for pair in primal_dual_pairs:
        primal_edges.append(pair[0])
        dual_edges.append(pair[1])
    tree_nodes = primal_draw.nodes
    #[primal_draw, dual_draw, primal_edges, dual_edges, tree_nodes] = read.read_draw(primal, dual)


    # Pick the roots
    dual_root = dual_roots[0]
    primal_root = primal_roots[0]
    print("this is the dual root:",dual_root)
    print("this is the primal root:",primal_root)
    # draw the input primal graph
    print("Here is the input graph: ")
    nx.draw(primal_draw, with_labels= True)
    # plt.show()

    # Model
    m = gp.Model("Williams")

    # Open log file
    #m.setParam('OutputFlag', 0)
    m.Params.LogFile = "williams_3"+state
    #logfile = open('Williams', 'w')
    #m.Params.LogToConsole = 0

    # Create decision variables w
    m._w = m.addVars(primal_graph.edges, vtype = GRB.BINARY, name="w")

    # Create decision variables y
    m._w_prime = m.addVars(dual_graph.edges, vtype = GRB.BINARY, name="w_prime")

    #set objective function sense
    m.modelSense = GRB.MINIMIZE

    # Set a time limit
    m.setParam('TimeLimit', 3600)

    add_regular_constraints(m, primal_dual_pairs, primal_graph, primal_nodes, primal_root, dual_graph, dual_nodes, dual_root)

    # Let Gurobi know that the model has changed
    m.update()
    #m.display()


    # Subgraph division
    if is_forest == True:
        subgraph_division(m, primal_graph, primal_nodes, primal_dual_pairs, k)
        # Let Gurobi know that the model has changed
        m.update()
        #m.display()

    # add population constraints here
    if is_population_considered and is_forest:
        p = [primal_draw.nodes[i]['P0010001'] for i in primal_draw.nodes()]
        add_population_constraints(m, p, primal_nodes, primal_graph, primal_edges, add_objective, k)
        # L is the lower bound of each node's population, and U is the upper bound (change this later)

    """
    # Add symmetry-breaking constraints here
    if symmetry_break:
        symmetry_break_constraints(m, face_txt, primal_edges, primal_graph)
    m.update()
    m.display()

    if strengthening:
        strengthening_constraints(m, p, face_txt, primal_edges, primal_graph, U)
    m.update()
    m.display()


    # Add supervalid constraints here
    if supervalid:
        supervalid_constraints(m, face_txt, primal_edges, primal_graph)
    m.update()
    m.display()
    """

    # Optimize model
    m.optimize()
    run_time = m.Runtime
    node_count = 0
    # Print the solution if optimality is achieved
    if m.status == GRB.OPTIMAL or m.status == 9:
        spanning_tree_edges = []
        forest_edges = []
        for primal_edge in primal_graph.edges:
            if m._w[primal_edge].X > 0.5:
                spanning_tree_edges.append(tuple(primal_edges[primal_edge[0]]))
        spanning_tree = primal_draw.edge_subgraph(spanning_tree_edges)
        print("# of selected edges in the spanning tree: ", len(spanning_tree_edges))
        nx.draw(spanning_tree, with_labels = True)
        # plt.show()
        for dual_edge in dual_graph.edges:
            if m._w_prime[dual_edge].X > 0.5:
                print("dual edge in the dual tree: ", dual_edge)
        if is_forest == True:
            for primal_edge in primal_graph.edges:
                if m._x[primal_edge].X > 0.5:
                    if int(primal_edge[1]) == primal_edges[primal_edge[0]][1]:
                        forest_edge = tuple(primal_edges[primal_edge[0]])
                    else:
                        forest_edge = tuple([primal_edges[primal_edge[0]][1], primal_edges[primal_edge[0]][0]])
                    forest_edges.append(forest_edge)
            forest = nx.DiGraph()
            forest.add_nodes_from(tree_nodes)
            for node in forest.nodes():
                forest.nodes[node]["pos"]=primal_draw.nodes[node]["pos"]
            forest.add_edges_from(forest_edges)
            nx.draw(forest, with_labels = True)
            # plt.show()
            undirected_forest = forest.to_undirected()
            print("Number of connected components:")
            num = len(list(nx.connected_components(undirected_forest)))
            print(num)
            node_count = m.NodeCount
            if m.status == GRB.OPTIMAL:
                obj_bound = m.objVal
                obj_val = m.objVal
            else:
                obj_bound = m.ObjBound
                obj_val = m.objVal
    return [run_time, node_count, undirected_forest, forest, obj_val, obj_bound]


def add_regular_constraints(m, primal_dual_pairs, primal_graph, primal_nodes, primal_root, dual_graph, dual_nodes, dual_root):
    # Constraints 1 & 2
    for primal_node in primal_nodes:
        if primal_node == primal_root:
            m.addConstr(gp.quicksum(m._w[neighbor, primal_node] for neighbor in primal_graph.predecessors(primal_node)) == 0)
        else:
            m.addConstr(gp.quicksum(m._w[neighbor, primal_node] for neighbor in primal_graph.predecessors(primal_node)) == 1)

    # Constraints 3 & 4
    for dual_node in dual_nodes:
        if dual_node == dual_root:
            m.addConstr(gp.quicksum(m._w_prime[neighbor, dual_node] for neighbor in dual_graph.predecessors(dual_node)) == 0)
        else:
            m.addConstr(gp.quicksum(m._w_prime[neighbor, dual_node] for neighbor in dual_graph.predecessors(dual_node)) == 1)

    # Constraint 5
    for i in range(len(primal_dual_pairs) - 1):
        w_sum = 0
        w_nodes = list(primal_graph.neighbors(i))
        w_sum += m._w[i, w_nodes[0]] + m._w[i, w_nodes[1]]
        w_prime_nodes = list(dual_graph.neighbors(i))
        w_sum += m._w_prime[i, w_prime_nodes[0]]
        if len(w_prime_nodes) != 1:
            w_sum += m._w_prime[i, w_prime_nodes[1]]
        m.addConstr(w_sum == 1)

def subgraph_division(m, primal_graph, primal_nodes, primal_dual_pairs, k):
    # Create variables for the selected edges in forest
    m._x = m.addVars(primal_graph.edges, vtype = GRB.BINARY, name="x")
    # Create root variables
    m._r = m.addVars(primal_nodes, name="r")
    # Constraint 1
    m.addConstrs(gp.quicksum(m._x[out_edge] for out_edge in primal_graph.out_edges(i))  <= gp.quicksum(m._w[out_edge]
                                            for out_edge in primal_graph.out_edges(i)) for i in range(len(primal_dual_pairs) - 1))
    # Constraint 2
    m.addConstr(gp.quicksum(m._r[node] for node in primal_nodes) == k)
    # Constraint 3
    m.addConstrs(m._r[node] + gp.quicksum(m._x[predecessor, node] for predecessor in primal_graph.predecessors(node)) == 1
                 for node in primal_nodes)

def add_population_constraints(m, p, primal_nodes, primal_graph, primal_edges, add_objective, k):
    # L is the lower bound of each node's population, and U is the upper bound
    total_pop = sum(p)
    L = (total_pop/k)*(0.995)
    U = (total_pop/k)*(1.005)
    out_edges = {}
    print(primal_edges)
    for edge in primal_graph.edges:
        true_edge = primal_edges[edge[0]]
        if int(edge[1]) == true_edge[0]:
            head_node = str(true_edge[1])
        else:
            head_node = str(true_edge[0])
        if head_node in out_edges.keys():
            out_edges[head_node].append(edge)
        else:
            out_edges[head_node] = [edge]

    # add variables: p is the population variable, g is the generated flow variable, and f is the arc flow variable
    m._g = m.addVars(primal_nodes, name = 'g')
    m._f = m.addVars(primal_graph.edges, name = 'f')

    # Have the option to add an objective function (skipped because we don't have the distance files right now)
    if add_objective:
        m.setObjective(gp.quicksum(m._f[edge] for edge in primal_graph.edges))

    # add constraints
    m.addConstrs(m._g[node] - m._r[node]*L >= 0 for node in primal_nodes)
    m.addConstrs(m._g[node] - m._r[node]*U <= 0 for node in primal_nodes)
    m.addConstrs(m._g[node] + gp.quicksum(m._f[predecessor, node] for predecessor in primal_graph.predecessors(node)) -
                  gp.quicksum(m._f[out_edge] for out_edge in out_edges[str(node)]) - p[node] == 0 for node in primal_nodes)
    m.addConstrs(m._f[edge] <= m._x[edge] * (U - p[int(head_node)]) for head_node in out_edges.keys() for edge in out_edges[head_node])

    m.update()
    #m.display()

"""
def symmetry_break_constraints(m, face_txt, primal_edges, primal_graph):
    faces = read_face(face_txt)
    for face in faces:
        edge_index_list = []
        for vertex_pair in itertools.combinations(face, 2):
            print("vertex_pair: ", vertex_pair)
            list_pair = []
            list_pair.append(min(int(vertex_pair[0]), int(vertex_pair[1])))
            list_pair.append(max(int(vertex_pair[0]), int(vertex_pair[1])))
            edge_index_list.append(primal_edges.index(list_pair))
        max_index = max(edge_index_list)
        for edge_pair in itertools.combinations(edge_index_list, 2):
            if max_index in edge_pair:
                m.addConstr(gp.quicksum(m._x[edge, list(primal_graph.neighbors(edge))[0]] + m._x[
                    edge, list(primal_graph.neighbors(edge))[1]] for edge in edge_pair) <= 1)
"""

"""
def strengthening_constraints(m, p, face_txt, primal_edges, primal_graph, U):
    faces = read_face(face_txt)
    for face in faces:
        edge_index_list = []
        for vertex_pair in itertools.combinations(face, 2):
            print("vertex_pair: ", vertex_pair)
            list_pair = []
            list_pair.append(min(int(vertex_pair[0]), int(vertex_pair[1])))
            list_pair.append(max(int(vertex_pair[0]), int(vertex_pair[1])))
            edge_index_list.append(primal_edges.index(list_pair))
        if sum(p[node] for node in face) > U:
            m.addConstr(gp.quicksum(
                m._x[edge, list(primal_graph.neighbors(edge))[0]] + m._x[edge, list(primal_graph.neighbors(edge))[1]]
                for edge in edge_index_list) <= 1)
"""

"""
def supervalid_constraints(m, face_txt, primal_edges, primal_graph):
    faces = read_face(face_txt)
    for face in faces:
        edge_index_list = []
        for vertex_pair in itertools.combinations(face, 2):
            print("vertex_pair: ", vertex_pair)
            list_pair = []
            list_pair.append(min(int(vertex_pair[0]), int(vertex_pair[1])))
            list_pair.append(max(int(vertex_pair[0]), int(vertex_pair[1])))
            edge_index_list.append(primal_edges.index(list_pair))
        # max_index = max(edge_index_list)
        for edge in edge_index_list:
            new_edge_index = [primal_edge for primal_edge in edge_index_list if primal_edge != edge]
            end_nodes = list(primal_graph.neighbors(edge))
            if end_nodes[0] in list(primal_graph.neighbors(new_edge_index[0])):
                start_node = end_nodes[0]
                end_node = end_nodes[1]
            else:
                start_node = end_nodes[1]
                end_node = end_nodes[0]
            for node in face:
                if node != start_node and node != end_node:
                    intermediate_node = node
            order_forward = [intermediate_node, end_node]
            order_backward = [intermediate_node, start_node]
            m.addConstr(gp.quicksum(m._w[new_edge_index[i], order_forward[i]] for i in range(len(new_edge_index))) <= 1)
            m.addConstr(gp.quicksum(m._w[new_edge_index[len(new_edge_index) - i - 1], order_backward[i]] for i in
                                    range(len(new_edge_index))) <= 1)
"""