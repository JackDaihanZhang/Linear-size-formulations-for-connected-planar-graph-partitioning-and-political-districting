import gurobipy as gp
import networkx as nx
from gurobipy import GRB

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
    m.setObjective(gp.quicksum(gp.quicksum(m._population[i]*D[i][j]*m._X[i,j] for j in G.nodes) for i in G.nodes), GRB.MINIMIZE)


def Hess_model(state, G, k):
    ############################
    # Build base model
    ############################
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