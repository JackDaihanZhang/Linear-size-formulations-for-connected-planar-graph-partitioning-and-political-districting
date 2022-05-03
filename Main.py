###########################
# Imports
###########################
import csv
import itertools
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from gerrychain import Graph
import face_finder

################################################
# Summarize computational results to csv file
################################################
def write_to_csv(state_rows, state_results, filename, fields):
    rows = []
    # Create an index to keep track of the
    result_index = 0
    for state in state_rows:
        [run_time, node_count, _, val, bound] = state_results[result_index]
        result_index += 1
        row = states_rows[state]
        row.insert(0, state)
        row.append(run_time)
        row.append(node_count)
        row.append(val)
        row.append(bound)
        rows.append(row)

    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        # writing the data rows
        csvwriter.writerows(rows)

def write_to_txt(forest, num_district, file_location, model):
    output_list = []
    if model == "Williams":
        components = sorted(list(nx.connected_components(forest)))
        for node in sorted(list(forest.nodes)):
            for index in range(num_district):
                if node in components[index]:
                    real_index = index + 1
                    output_list.append([str(node) + " " + str(real_index) + "\n"])
    else:
        for i in range(1,num_district+1):
            for node in forest[i-1]:
                output_list.append([str(node) + " " + str(i) + "\n"])
    with open(file_location, 'w') as f:
        for line in output_list:
            f.writelines(line)

###########################
# Hard-coded inputs
###########################

#states_rows = {"AL": [67, 106, 171, 7], "AR": [75, 119, 192, 4], "IA": [99, 125, 222, 4], "KS": [105, 160, 263, 4],
#                "ME": [16, 20, 34, 2], "MS": [82, 122, 202, 4], "NE": [93, 140, 231, 3], "NM": [33, 47, 78, 3],
#               "WV": [55, 72, 125, 2], "ID":[44, 60, 102, 2]}
states_rows = {"ME": [16, 20, 34, 2]}
fields = ["State", "Primal Vertices", "Dual Vertices", "Edges", "Districts", "Run Time (Seconds)", "Branch and Bound Nodes", "Objective Value", "Objective Bound"]


###########################
# Run An Instance using Williams' model
###########################
def run_williams(state):
    num_district = states_rows[state][3]
    population_file = "Population/" + state + "_population.txt"
    face_txt = "Faces/" + state + "_faces.txt"
    return Williams_model(population_file, face_txt, num_district, state)

###########################
# Run An Instance using Hess' model
###########################
def run_Hess(state):
    num_district = states_rows[state][3]
    return Hess_model(state,num_district)


###########################
# Run the complete experiment
###########################
state_results = []
# Specify the model
model = "Hess"
for state in states_rows:
    if model == "Hess":
        result = run_Hess(state)
    else:
        result = run_williams(state)
    state_results.append(result)
    write_to_txt(result[2],states_rows[state][3], "Experiment_" + model + "/" + state + "_solution.txt", model)
write_to_csv(states_rows, state_results, "Experiment_" + model + "/" + "result.csv", fields)


###########################
# Williams' model
###########################
def Williams_model(population_file, face_txt, k, state):
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
    primal_draw = Graph.from_json("C:/Rice/junior/Spring Semester/Research/JSON/County/" + state + "_counties.json")
    [dual_draw, primal_dual_pairs] = face_finder.restricted_planar_dual(primal_draw)

    [primal_graph, dual_graph, primal_nodes, dual_nodes, primal_roots, dual_roots] = read_williams(primal_dual_pairs)
    primal_edges = primal_draw.edges
    dual_edges = dual_draw.edges
    tree_nodes = primal_draw.nodes
    #[primal_draw, dual_draw, primal_edges, dual_edges, tree_nodes] = read.read_draw(primal, dual)


    # Pick the roots
    dual_root = dual_roots[0]
    primal_root = primal_roots[0]
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
    m._w = m.addVars(primal_graph.edges, name="w")

    # Create decision variables y
    m._y = m.addVars(dual_graph.edges, name="y")

    #set objective function sense
    m.modelSense = GRB.MINIMIZE

    # Set a time limit
    m.setParam('TimeLimit', 3600)

    add_regular_constraints(m, primal_dual_pairs, primal_graph, primal_nodes, primal_root, dual_graph, dual_nodes, dual_root)

    # Let Gurobi know that the model has changed
    m.update()
    m.display()


    # Subgraph division
    if is_forest == True:
        subgraph_division(m, primal_graph, primal_nodes, primal_dual_pairs, k)
        # Let Gurobi know that the model has changed
        m.update()
        m.display()

    # add population constraints here
    if is_population_considered and is_forest:
        p = [primal_draw.nodes[i]['P0010001'] for i in primal_draw.nodes()]
        add_population_constraints(m, p, primal_nodes, primal_graph, primal_edges, add_objective, k)
        # L is the lower bound of each node's population, and U is the upper bound (change this later)
        total_pop = sum(p)
        L = (total_pop / k) * (0.995)
        U = (total_pop / k) * (1.005)

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
            if m._y[dual_edge].X > 0.5:
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
    return [run_time, node_count, undirected_forest, obj_val, obj_bound]


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
            m.addConstr(gp.quicksum(m._y[neighbor, dual_node] for neighbor in dual_graph.predecessors(dual_node)) == 0)
        else:
            m.addConstr(gp.quicksum(m._y[neighbor, dual_node] for neighbor in dual_graph.predecessors(dual_node)) == 1)

    # Constraint 5
    for i in range(len(primal_dual_pairs) - 1):
        wy_sum = 0
        w_nodes = list(primal_graph.neighbors(i))
        wy_sum += m._w[i, w_nodes[0]] + m._w[i, w_nodes[1]]
        y_nodes = list(dual_graph.neighbors(i))
        wy_sum += m._y[i, y_nodes[0]]
        if len(y_nodes) != 1:
            wy_sum += m._y[i, y_nodes[1]]
        m.addConstr(wy_sum == 1)

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
    L = (total_pop/k)*(0.0995)
    U = (total_pop/k)*(1.005)
    out_edges = {}
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
    m._g = m.addVars(primal_nodes, vtype = GRB.INTEGER, name = 'g')
    m._f = m.addVars(primal_graph.edges, vtype = GRB.INTEGER, name = 'f')

    # Have the option to add an objective function (skipped because we don't have the distance files right now)
    if add_objective:
        m.setObjective(gp.quicksum(m._f[edge] for edge in primal_graph.edges))

    # add constraints
    m.addConstrs(m._g[node] - m._r[node]*L >= 0 for node in primal_nodes)
    m.addConstrs(m._g[node] - m._r[node]*U <= 0 for node in primal_nodes)
    m.addConstrs(m._g[node] + gp.quicksum(m._f[predecessor, node] for predecessor in primal_graph.predecessors(node)) -
                  gp.quicksum(m._f[out_edge] for out_edge in out_edges[node]) - p[node] == 0 for node in primal_nodes)
    m.addConstrs(m._f[edge] <= m._x[edge] * (U - p[head_node]) for head_node in out_edges.keys() for edge in out_edges[head_node])

    m.update()
    m.display()

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

###########################
# Hess' model
###########################
# Source: https://github.com/hamidrezavalidi/Political-Districting-to-Minimize-Cut-Edges/blob/master/src/hess.py
def add_base_constraints(m, population, L, U, k):
    DG = m._DG
    # Each vertex i assigned to one district
    m.addConstrs(gp.quicksum(m._X[i, j] for j in DG.nodes) == 1 for i in DG.nodes)

    # Pick k centers
    m.addConstr(gp.quicksum(m._X[j, j] for j in DG.nodes) == k)

    # Population balance: population assigned to vertex j should be in [L,U], if j is a center
    m.addConstrs(gp.quicksum(population[i] * m._X[i, j] for i in DG.nodes) <= U * m._X[j, j] for j in DG.nodes)
    m.addConstrs(gp.quicksum(population[i] * m._X[i, j] for i in DG.nodes) >= L * m._X[j, j] for j in DG.nodes)

    # Add coupling inequalities for added model strength
    couplingConstrs = m.addConstrs(m._X[i, j] <= m._X[j, j] for i in DG.nodes for j in DG.nodes)

    # Make them user cuts
    for i in DG.nodes:
        for j in DG.nodes:
            couplingConstrs[i, j].Lazy = -1

    # Set branch priority on center vars
    for j in DG.nodes:
        m._X[j, j].BranchPriority = 1


def add_shir_constraints(m):
    DG = m._DG

    # F[j,u,v] tells how much flow (from source j) is sent across arc (u,v)
    F = m.addVars(DG.nodes, DG.edges, vtype=GRB.CONTINUOUS)

    # compute big-M
    M = most_possible_nodes_in_one_district(m._population, m._U) - 1

    m.addConstrs(gp.quicksum(F[j, u, j] for u in DG.neighbors(j)) == 0 for j in DG.nodes)
    m.addConstrs(
        gp.quicksum(F[j, u, i] - F[j, i, u] for u in DG.neighbors(i)) == m._X[i, j] for i in DG.nodes for j in DG.nodes
        if i != j)
    m.addConstrs(
        gp.quicksum(F[j, u, i] for u in DG.neighbors(i)) <= M * m._X[i, j] for i in DG.nodes for j in DG.nodes if
        i != j)
    m.update()

def most_possible_nodes_in_one_district(population, U):
    cumulative_population = 0
    num_nodes = 0
    for ipopulation in sorted(population):
        cumulative_population += ipopulation
        num_nodes += 1
        if cumulative_population > U:
            return num_nodes - 1

def add_objective(m, G):
    # Create the distance file
    D = {}
    for i in G.nodes:
        D[i] =  nx.shortest_path_length(G, source=i)
    # Y[i,j] = 1 if edge {i,j} is cut
    # m._Y = m.addVars(G.edges, vtype=GRB.BINARY)
    # m.addConstrs( m._X[i,v]-m._X[j,v] <= m._Y[i,j] for i,j in G.edges for v in G.nodes)
    # m.setObjective(gp.quicksum(m._Y), GRB.MINIMIZE)
    m.setObjective(gp.quicksum(gp.quicksum(m._population[i]*D[i][j]*m._X[i,j] for j in G.nodes) for i in G.nodes), GRB.MINIMIZE)


def Hess_model(state, k):
    ############################
    # Build base model
    ############################
    G = Graph.from_json("C:/Rice/junior/Spring Semester/Research/JSON/County/"+state+"_counties.json")
    population = [G.nodes[i]['P0010001'] for i in G.nodes()]
    total_pop = sum(population)
    U = 1.005*(total_pop/k)
    L = 0.995*(total_pop/k)
    DG = nx.DiGraph(G)
    m = gp.Model()
    m._DG = DG
    m._U = U
    m._population = population
    # Set a time limit
    m.setParam('TimeLimit', 3600)

    # X[i,j]=1 if vertex i is assigned to (district centered at) vertex j
    m._X = m.addVars(DG.nodes, DG.nodes, vtype=GRB.BINARY)
    add_base_constraints(m, population, L, U, k)
    add_objective(m, G)
    add_shir_constraints(m)
    m.update()

    m.optimize()
    run_time = m.Runtime
    node_count = 0
    # Print the solution if optimality is achieved
    if m.status == GRB.OPTIMAL or m.status == 9:
        forests = []
        nodes = [i for i in range(len(G.nodes()))]
        added_nodes = {}
        k = 0
        for i in nodes:
            for j in nodes:
                if m._X[i,j].X > 0.5:
                    # It must be the first time node i is being added, while node j, as the root of a district, could
                    # have already been added
                    if j in added_nodes:
                        if i != j:
                            forests[added_nodes[j]].append(i)
                    else:
                        if i != j:
                            forests.append([i,j])
                        else:
                            forests.append([j])
                        added_nodes[j] = k
                        k = k + 1
            node_count = m.NodeCount
            if m.status == GRB.OPTIMAL:
                obj_bound = m.objVal
                obj_val = m.objVal
            else:
                obj_bound = m.ObjBound
                obj_val = m.objVal
    return [run_time, node_count, forests, obj_val, obj_bound]

############################
    # Read functions
#############################
###########################
# Read the primal and dual txt file for the main graphs in Williams' model
###########################
def read_williams(primal_dual_pairs):
    primal_graph = nx.DiGraph()
    dual_graph = nx.DiGraph()
    primal_nodes = set([])
    dual_nodes = set([])
    for i in range(primal_dual_pairs):
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