import gurobipy as gp
from gurobipy import GRB
import read
import networkx as nx
import matplotlib.pyplot as plt
import population

# Do we deal with a forest?
is_forest = True

# Is population balance considered?
is_population_considered = True

# Add an objective function to measure compactness?
add_objective = True

primal = "grid.txt"
dual = "dual_graph.txt"

distance_file = "grid_distances.csv"

k = 3


# Read the primal and dual graph from the txt files
[primal_file, dual_file, primal_graph, dual_graph, primal_nodes, dual_nodes, primal_roots, dual_roots] = \
    read.read(primal, dual)

[primal_draw, primal_edges, tree_nodes] = read.read_draw(primal)

# draw the input primal graph
print("Here is the input graph: ")
nx.draw(primal_draw, with_labels= True)
plt.show()

# Model
m = gp.Model("Williams")

# Create decision variables x
w = m.addVars(primal_graph.edges, name="w")

# Create decision variables y
y = m.addVars(dual_graph.edges, name="y")

#set objective function sense
m.modelSense = GRB.MINIMIZE

# Pick the roots
dual_root = dual_roots[0]
primal_root = primal_roots[0]

# Constraints 1 & 2
for primal_node in primal_nodes:
    if primal_node == primal_root:
        m.addConstr(gp.quicksum(w[neighbor, primal_node] for neighbor in primal_graph.predecessors(primal_node)) == 0)
    else:
        m.addConstr(gp.quicksum(w[neighbor, primal_node] for neighbor in primal_graph.predecessors(primal_node)) == 1)

# Constraints 3 & 4
for dual_node in dual_nodes:
    if dual_node == dual_root:
        m.addConstr(gp.quicksum(y[neighbor, dual_node] for neighbor in dual_graph.predecessors(dual_node)) == 0)
    else:
        m.addConstr(gp.quicksum(y[neighbor, dual_node] for neighbor in dual_graph.predecessors(dual_node)) == 1)

# Constraint 5
for i in range(len(primal_file)):
    wy_sum = 0
    w_nodes = list(primal_graph.neighbors(i))
    wy_sum += w[i, w_nodes[0]] + w[i, w_nodes[1]]
    y_nodes = list(dual_graph.neighbors(i))
    wy_sum += y[i, y_nodes[0]]
    if len(y_nodes) != 1:
        wy_sum += y[i, y_nodes[1]]
    m.addConstr(wy_sum == 1)


# Let Gurobi know that the model has changed
m.update()
m.display()
# Write out the lp in a lp-file
m.write("msp.lp")


# Subgraph division
if is_forest == True:
    # Create variables for the selected edges in forest
    m._x = m.addVars(primal_graph.edges, vtype = GRB.BINARY, name="x")
    # Create root variables
    m._r = m.addVars(primal_nodes, name="r")
    # Constraint 1
    m.addConstrs(gp.quicksum(m._x[out_edge] for out_edge in primal_graph.out_edges(i))  <= gp.quicksum(w[out_edge]
                                            for out_edge in primal_graph.out_edges(i)) for i in range(len(primal_file)))
    # Constraint 2
    m.addConstr(gp.quicksum(m._r[node] for node in primal_nodes) == k)
    # Constraint 3
    m.addConstrs(m._r[node] + gp.quicksum(m._x[predecessor, node] for predecessor in primal_graph.predecessors(node)) == 1
                 for node in primal_nodes)

    # Let Gurobi know that the model has changed
    m.update()
    m.display()

# add population constraints here
if is_population_considered and is_forest:
    p = read.read_population("grid_population.txt")
    d_dict = read.read_distance(distance_file, primal_graph, primal_edges)
    population.add_population_constraints(m, p, primal_nodes, primal_graph, primal_edges, add_objective, d_dict)



# Optimize model
m.optimize()
# Print the solution if optimality is achieved
if m.status == GRB.OPTIMAL:
    spanning_tree_edges = []
    forest_edges = []
    for primal_edge in primal_graph.edges:
        if w[primal_edge].X > 0.5:
            spanning_tree_edges.append(tuple(primal_edges[primal_edge[0]]))
    spanning_tree = primal_draw.edge_subgraph(spanning_tree_edges)
    nx.draw(spanning_tree, with_labels = True)
    plt.show()
    for dual_edge in dual_graph.edges:
        if y[dual_edge].X > 0.5:
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
        plt.show()