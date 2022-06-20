import gurobipy as gp
import networkx as nx
from gurobipy import GRB

# Source: https://github.com/hamidrezavalidi/Political-Districting-to-Minimize-Cut-Edges/blob/master/src/hess.py
def add_base_constraints(m, k):
    DG = m._DG
    # Each vertex i assigned to one district
    m.addConstrs(gp.quicksum(m._X[i, j] for j in DG.nodes) == 1 for i in DG.nodes)

    # Pick k centers
    m.addConstr(gp.quicksum(m._X[j, j] for j in DG.nodes) == k)

    # Population balance: population assigned to vertex j should be in [L,U], if j is a center
    m.addConstrs(gp.quicksum(m._p[i] * m._X[i, j] for i in DG.nodes) <= m._U * m._X[j, j] for j in DG.nodes)
    m.addConstrs(gp.quicksum(m._p[i] * m._X[i, j] for i in DG.nodes) >= m._L * m._X[j, j] for j in DG.nodes)

    # Add coupling inequalities for added model strength
    m.addConstrs(m._X[i, j] <= m._X[j, j] for i in DG.nodes for j in DG.nodes)

    # Set branch priority on center vars
    for j in DG.nodes:
        m._X[j, j].BranchPriority = 1

def add_shir_constraints(m):
    DG = m._DG

    # F[j,u,v] tells how much flow (from source j) is sent across arc (u,v)
    F = m.addVars(DG.nodes, DG.edges, vtype = GRB.CONTINUOUS)

    # compute big-M
    M = most_possible_nodes_in_one_district(m._p, m._U) - 1

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
        D[i] =  nx.shortest_path_length(G, source = i)
    m.setObjective(gp.quicksum(gp.quicksum(m._p[i]*D[i][j]*m._X[i, j] for j in G.nodes) for i in G.nodes))

def Hess_model(m):
    ############################
    # Build base model
    ############################
    G = m._G
    DG = nx.DiGraph(G)
    m._DG = DG
    m._X = m.addVars(DG.nodes, DG.nodes, vtype = GRB.BINARY)
    add_base_constraints(m, m._k)
    add_objective(m, G)
    add_shir_constraints(m)
    
    ####################################   
    # Inject heuristic warm start
    ####################################    
    if m._heuristic:
        for district in m._hdistricts:    
            H = G.subgraph(district)
            min_score = nx.diameter(H) * max(m._p) * len(district)
            min_root = -1
            for vertex in H.nodes:
                length, path = nx.single_source_dijkstra(H, vertex)
                score = sum(length[node]*m._p[node] for node in H.nodes)
                if score < min_score:
                    min_score = score
                    min_root = vertex
            for i in district:
                m._X[i, min_root].start = 1
    m.update()
    m.optimize()
    run_time = m.Runtime
    node_count = 0
    # Print the solution if optimality if a feasible solution has been found
    if m.SolCount > 0:
        labels = [ j for j in DG.nodes if m._X[j, j].x > 0.5 ]
        
        districts = [ [ i for i in DG.nodes if m._X[i, j].x > 0.5 ] for j in labels]             
        node_count = m.NodeCount
        obj_bound = m.ObjBound
        obj_val = m.objVal
    else:
        node_count = 0
        districts = "N/A"
        obj_val = "N/A"
        obj_bound = m.ObjBound
    return [run_time, node_count, districts, obj_val, obj_bound]