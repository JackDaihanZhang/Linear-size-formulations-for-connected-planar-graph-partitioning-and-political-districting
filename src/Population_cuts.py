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
        rval = m.cbGetSolution(m._r)
        roots = [i for i in m._primalnodes if rval[i] > 0.5]
        #print("Roots:", roots)
        # which edges in the primal graph are selected?
        forest_edges = [ e for e in m._primalgraph.edges if xval[e] > 0.5 ]
        #print("Number of edges in the forest",len(forest_edges))
        #print("Edges in the forest:", forest_edges)
        # which edges in the actual graph (primal_draw) are selected
        real_forest_edges = []
        for e in forest_edges:
            if int(e[1]) == m._primaledges[e[0]][1]:
                real_forest_edge = tuple(m._primaledges[e[0]])
            else:
                real_forest_edge = tuple([m._primaledges[e[0]][1], m._primaledges[e[0]][0]])
            real_forest_edges.append(real_forest_edge)
        #print("Number of edges in the real forest",len(real_forest_edges))
        
        #print("number_connected_components(G): ", nx.number_connected_components(m._primaldraw.edge_subgraph( real_forest_edges ))) 
        subgraph = nx.Graph(m._primaldraw.edge_subgraph( real_forest_edges ))
        subgraph.add_nodes_from(roots)
        #print("number_connected_components(G): ", nx.number_connected_components(subgraph) )
        #print("Edges in the real forest:", real_forest_edges)
        #print(m._primaledges)
        for component in nx.connected_components( subgraph ):
            
            component_population = sum( m._p[i] for i in component )
            
            #print("Component: ", component)
            #print("Component population is ", component_population, ", upper bound is ", m._U, " and lower bound is ", m._L)
            if component_population > m._U:
                cut_edges = []
                for i in component:
                    #cut_edges = []
                    for predecessor in  m._primalgraph.predecessors(i):
                        real_edge = m._primaledges[predecessor]
                        if real_edge[0] == i:
                            other_node = real_edge[1]
                        else:
                            other_node = real_edge[0]
                        #print("The other node is:", other_node)
                        #print(not other_node in component)
                        if not other_node in component:
                            cut_edges.append((predecessor,i))
                            #print("Cut edges updated:", cut_edges)
                m.cbLazy(gp.quicksum(m._x[e] for e in cut_edges) + gp.quicksum(m._r[i] for i in component) >= math.ceil(component_population/m._U))
                #print(gp.quicksum(m._x[e] for e in cut_edges) + gp.quicksum(m._r[i] for i in component) >= math.ceil(component_population/m._U))
                #cut_edges = [ (i,j) for (i,j) in m._G.edges if ( i in component) ^ ( j in component ) ]
                #m.cbLazy( gp.quicksum( m._x[e] for e in cut_edges ) >= 2 * math.ceil( component_demand / m._Q ) )
                #print("Here is the over-populated component: ", component)
                #print("An overcut inequality is added for the following cut edges: ", cut_edges)
                
            '''    
            elif component_population < m._L:
                cut_edges = []
                for i in component:
                    for predecessor in  m._primalgraph.predecessors(i):
                            real_edge = m._primaledges[predecessor]
                            if real_edge[0] == i:
                                other_node = real_edge[1]
                            else:
                                other_node = real_edge[0]
                            if not other_node in component:
                                cut_edges.append((predecessor,i))
                    for out_edge in m._incident[str(i)]:
                        if not out_edge[1] in component:
                            cut_edges.append(out_edge)
                            
                m.cbLazy(gp.quicksum(m._x[e] for e in cut_edges) >= 1)
                
            '''    
                #print(cut_edges)
                #print("Here is the under-populated component: ", component)
                #print("An undercut inequality is added for the following cut edges: ", cut_edges)
                
        #m.update()
                