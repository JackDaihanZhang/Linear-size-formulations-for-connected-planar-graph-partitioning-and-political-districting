import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import face_finder
import read
import Population_cuts
import RCI_cuts
from itertools import combinations


def Williams_model(m):
    # Do we deal with a forest?
    is_forest = True
    # Is population balance considered?
    is_population_considered = True
    # Add an objective function to measure compactness?
    add_objective = True
    
    # Retrieve model inputs
    primal_draw = m._G
    df = m._df 
    primal_dual_pairs = m._pdp
    # Construct the graphs from JSON file
    # If running on county level, use face_finder.py to construct the original dual
    # graph (dual_draw) and primal_dual_pairs
    if m._level == "county":
        primal_dual_pairs = face_finder.restricted_planar_dual(primal_draw, df, m._state)[1]
     
    # Obtain all necessary inputs for the model
    [primal_graph, dual_graph, primal_nodes, dual_nodes, primal_roots, dual_roots] = read.read_Williams(primal_dual_pairs)
    
    # Attach newly constructed parameters to the model for convenience
    m._primalgraph = primal_graph
    m._primaldraw = primal_draw
    
    # Construct the edge lists for convenience later
    primal_edges = []
    dual_edges = []
    for pair in primal_dual_pairs:
        primal_edges.append(pair[0])
        dual_edges.append(pair[1])
    tree_nodes = primal_draw.nodes
    m._primaledges = primal_edges

    # Pick the roots
    dual_root = dual_roots[0]
    primal_root = primal_roots[0]

    # Create decision variables w
    m._w = m.addVars(primal_graph.edges, vtype = GRB.BINARY, name = "w")

    # Create decision variables w_prime
    m._w_prime = m.addVars(dual_graph.edges, vtype = GRB.BINARY, name = "w_prime")

    add_regular_constraints(m, primal_dual_pairs, primal_graph, primal_nodes, primal_root, dual_graph, dual_nodes, dual_root)

    m.update()

    # Subgraph division
    if is_forest == True:
        subgraph_division(m, primal_graph, primal_nodes, primal_dual_pairs)
        m.update()

    # Add population constraints here
    if is_population_considered and is_forest:
        add_population_constraints(m, primal_nodes, primal_graph, primal_edges, add_objective)
            
    # Add max-clique constraints here:
    if m._maxclique:
        add_max_clique_constraints(m)
    
    # Optimize model
    m.optimize(m._callback)
    run_time = m.Runtime
    # Print the solution if optimality if a feasible solution has been found
    if m.SolCount > 0:
        spanning_tree_edges = []
        for primal_edge in primal_graph.edges:
            if m._w[primal_edge].X > 0.5:
                spanning_tree_edges.append(tuple(primal_edges[primal_edge[0]]))
        # Construct the forest graph and obtain experiment result
        if is_forest == True:
            forest_edges = []
            for primal_edge in primal_graph.edges:
                if m._x[primal_edge].X > 0.5:
                    if primal_edge[1] == primal_edges[primal_edge[0]][1]:
                        forest_edge = tuple(primal_edges[primal_edge[0]])
                    else:
                        forest_edge = tuple([primal_edges[primal_edge[0]][1], primal_edges[primal_edge[0]][0]])
                    forest_edges.append(forest_edge)
            directed_forest = nx.DiGraph()
            directed_forest.add_nodes_from(tree_nodes)
            if m._level == "county":
                for node in directed_forest.nodes():
                    directed_forest.nodes[node]["pos"] = primal_draw.nodes[node]["pos"]
            directed_forest.add_edges_from(forest_edges)
            forest = directed_forest.to_undirected()
            node_count = m.NodeCount
            obj_bound = m.ObjBound
            obj_val = m.objVal
    # Make all the solution attributes 0 if no feasible solution is found
    else:
        node_count = 0
        directed_forest = 0
        forest = 0
        obj_val = 0
        obj_bound = m.ObjBound
    return [run_time, node_count, forest, directed_forest, obj_val, obj_bound]


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
    for i in range(len(primal_dual_pairs)):
        w_sum = 0
        w_nodes = list(primal_graph.neighbors(i))
        w_sum += m._w[i, w_nodes[0]] + m._w[i, w_nodes[1]]
        w_prime_nodes = list(dual_graph.neighbors(i))
        w_sum += m._w_prime[i, w_prime_nodes[0]]
        # If the dual edge is not a loop
        if len(w_prime_nodes) != 1:
            w_sum += m._w_prime[i, w_prime_nodes[1]]
        m.addConstr(w_sum == 1)

def subgraph_division(m, primal_graph, primal_nodes, primal_dual_pairs):
    # Create variables for the selected edges in forest
    m._x = m.addVars(primal_graph.edges, vtype = GRB.BINARY, name= "x")
    # Create root variables
    m._s = m.addVars(primal_nodes, name= "s")
    
    m._primalnodes = primal_nodes
    
    # Set branch priority on root vars
    for j in primal_nodes:
        m._s[j].BranchPriority = 1
        
    ####################################   
    # Inject heuristic warm start
    ####################################    
    if m._heuristic:
        for cut_edge in m._cuts:
            if cut_edge in m._primaledges:
                index = m._primaledges.index(cut_edge)
            else:
                reversed_edge = [cut_edge[1], cut_edge[0]]
                index = m._primaledges.index(reversed_edge)
            m._x[index, cut_edge[0]].start = 0
            m._x[index, cut_edge[1]].start = 0
        
    # Coupling constraints
    m.addConstrs(gp.quicksum(m._x[out_edge] for out_edge in primal_graph.out_edges(i))  <= gp.quicksum(m._w[out_edge]
                                            for out_edge in primal_graph.out_edges(i)) for i in range(len(primal_dual_pairs)))
    # Set number of roots to k
    m.addConstr(gp.quicksum(m._s[node] for node in primal_nodes) == m._k)
    # Constraint 3
    m.addConstrs(m._s[node] + gp.quicksum(m._x[predecessor, node] for predecessor in primal_graph.predecessors(node)) == 1
                 for node in primal_nodes)

def add_population_constraints(m, primal_nodes, primal_graph, primal_edges, add_objective):
    # Create a dictionary that stores the outbound edges of every node
    out_edges = {}
    for edge in primal_graph.edges:
        true_edge = primal_edges[edge[0]]
        if edge[1] == true_edge[0]:
            head_node = str(true_edge[1])
        else:
            head_node = str(true_edge[0])
        if head_node in out_edges.keys():
            out_edges[head_node].append(edge)
        else:
            out_edges[head_node] = [edge]
    m._incident = out_edges
    
    #add variables: p is the population variable, g is the generated flow variable, and f is the arc flow variable
    m._g = m.addVars(primal_nodes, name = 'g')
    m._f = m.addVars(primal_graph.edges, name = 'f')
    
    if m._populationparam == "flow":      
        ####################################   
        # Inject heuristic warm start
        ####################################   
        if m._heuristic:
            for district in m._hdistricts:                
                H = m._primaldraw.subgraph(district)
                min_score = nx.diameter(H) * max(m._p) * len(district)
                min_root = -1
                min_path = []
                for vertex in H.nodes:
                    length, path = nx.single_source_dijkstra(H, vertex)
                    score = sum(length[node]*m._p[node] for node in H.nodes)
                    if score < min_score:
                        min_score = score
                        min_root = vertex
                        min_path = path
                # warm start root var        
                m._s[min_root].start = 1
                
                # warm start generated population
                district_population = sum(m._p[i] for i in district)
                m._g[min_root].start = district_population
                
                # warm start selected edges inside a district
                for vertex in H.nodes:
                    if vertex == min_root: continue
                    current_path = min_path[vertex]
                    for i in range(len(current_path)-1):
                        current_node = current_path[i]
                        next_node = current_path[i+1]
                        edge = [current_node, next_node]
                        opposite_edge = [next_node, current_node]
                        if edge in m._primaledges:
                            index = m._primaledges.index(edge)
                        else:
                            index =  m._primaledges.index(opposite_edge)
                        m._x[index, next_node].start = 1
                        m._x[index, current_node].start = 0
                           
        # Have the option to add an objective function
        if add_objective:
            m.setObjective(gp.quicksum(m._f[edge] for edge in primal_graph.edges))
    
        # add constraints
        m.addConstrs(m._g[node] - m._s[node]*m._L >= 0 for node in primal_nodes)
        m.addConstrs(m._g[node] - m._s[node]*m._U <= 0 for node in primal_nodes)
        m.addConstrs(m._g[node] + gp.quicksum(m._f[predecessor, node] for predecessor in primal_graph.predecessors(node)) -
                      gp.quicksum(m._f[out_edge] for out_edge in out_edges[str(node)]) - m._p[node] == 0 for node in primal_nodes)
        m.addConstrs(m._f[edge] <= m._x[edge] * (m._U - m._p[int(head_node)]) for head_node in out_edges.keys() for edge in out_edges[head_node])
        
        if m._RCI:
            #tell Gurobi that we will be adding (lazy) constraints
            m.Params.lazyConstraints = 1

            # designate the callback routine 
            m._callback = RCI_cuts.RCI_inequalities
    
    elif m._populationparam == "cuts":
        if add_objective:
            m.setObjective(gp.quicksum(m._f[edge] for edge in primal_graph.edges))   
        
        m.addConstrs(m._g[node] >= m._s[node]*m._L for node in primal_nodes)  
    
        m.addConstrs(m._g[node] <= m._s[node]*m._total_pop for node in primal_nodes)
        
        m.addConstrs(m._g[node] + gp.quicksum(m._f[predecessor, node] for predecessor in primal_graph.predecessors(node)) -
                     gp.quicksum(m._f[out_edge] for out_edge in out_edges[str(node)]) - m._p[node] == 0 for node in primal_nodes)
        m.addConstrs(m._f[edge] <= m._x[edge] * (m._U - m._p[int(head_node)]) for head_node in out_edges.keys() for edge in out_edges[head_node])        

        #tell Gurobi that we will be adding (lazy) constraints
        m.Params.lazyConstraints = 1

        # designate the callback routine 
        m._callback = Population_cuts.rounded_capacity_ineq
           
    m.update()
    
def add_max_clique_constraints(m):
    G  = m._G
    cliques = nx.find_cliques(G)
    for clique in cliques:
        if sum([m._p[i] for i in clique]) <= m._U:
            continue
        real_edges = combinations(clique, 2)
        digraph_edges = []
        # Convert the real-graph edges to the digraph representation used in the model
        for real_edge in real_edges:
            if list(real_edge) in m._primaledges:
                index = m._primaledges.index(list(real_edge))
            else:
                reversed_edge = [real_edge[1], real_edge[0]]
                index = m._primaledges.index(reversed_edge)
            digraph_edges.append((index, real_edge[0]))
            digraph_edges.append((index, real_edge[1]))
        m.addConstr(gp.quicksum(m._x[e] for e in digraph_edges) <= len(clique) - 2)
        m._numMaxClique += 1
    m.update()
    
