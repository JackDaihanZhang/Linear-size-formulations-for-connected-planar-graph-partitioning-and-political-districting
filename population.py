import gurobipy as gp
from gurobipy import GRB
import read
import networkx as nx

# The function to be called in William's file
def add_population_constraints(m, r, x, p, primal_nodes, primal_graph, primal_edges, add_objective, d_dict):

    # L is the lower bound of each node's population, and U is the upper bound
    L = 3
    U = 3
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
    g = m.addVars(primal_nodes, vtype = GRB.INTEGER, name = 'g')
    f = m.addVars(primal_graph.edges, vtype = GRB.INTEGER, name = 'f')

    # Have the option to add an objective function
    if add_objective:
        m.setObjective(gp.quicksum(f[edge]*d_dict[edge] for edge in primal_graph.edges))

    # add constraints
    m.addConstrs(g[node] - r[node]*L >= 0 for node in primal_nodes)
    m.addConstrs(g[node] - r[node]*U <= 0 for node in primal_nodes)
    m.addConstrs(g[node] + gp.quicksum(f[predecessor, node] for predecessor in primal_graph.predecessors(node)) -
                  gp.quicksum(f[out_edge] for out_edge in out_edges[node]) - p[node] == 0 for node in primal_nodes)
    for head_node in out_edges.keys():
        m.addConstrs(f[edge] <= x[edge]*(U - p[head_node]) for edge in out_edges[head_node])

    m.update()
    m.display()