import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import math

# create a function to separate the rounded capacity inequalities (or subtour elimination)
def rounded_capacity_ineq(m, where):   
    # check if LP relaxation at this branch-and-bound node has an integer solution
    if where == GRB.Callback.MIPSOL: 
        # retrieve the LP solution
        xval = m.cbGetSolution(m._x)
        sval = m.cbGetSolution(m._s)
        roots = [i for i in m._primalnodes if sval[i] > 0.5]
        # which edges in the primal graph are selected?
        forest_edges = [ e for e in m._primalgraph.edges if xval[e] > 0.5 ]
        # which edges in the actual graph (primal_draw) are selected
        real_forest_edges = []
        for e in forest_edges:
            if e[1] == m._primaledges[e[0]][1]:
                real_forest_edge = tuple(m._primaledges[e[0]])
            else:
                real_forest_edge = tuple([m._primaledges[e[0]][1], m._primaledges[e[0]][0]])
            real_forest_edges.append(real_forest_edge)
        subgraph = nx.Graph(m._primaldraw.edge_subgraph( real_forest_edges ))
        subgraph.add_nodes_from(roots)
        
        # Add a lazy constraint for components that exceed the populaiton upper bound
        for component in nx.connected_components( subgraph ):           
            component_population = sum( m._p[i] for i in list(component) )
            if component_population > m._U:
                cut_edges = []
                for i in component:
                    for predecessor in  m._primalgraph.predecessors(i):
                        real_edge = m._primaledges[predecessor]
                        if real_edge[0] == i:
                            other_node = real_edge[1]
                        else:
                            other_node = real_edge[0]
                        if other_node not in component:
                            cut_edges.append((predecessor,i))
                m.cbLazy(gp.quicksum(m._x[e] for e in cut_edges) + gp.quicksum(m._s[i] for i in component) >= math.ceil(component_population/m._U))
                cut_edges = [ (i,j) for (i,j) in m._G.edges if ( i in component) ^ ( j in component ) ]