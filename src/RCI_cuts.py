import gurobipy as gp
from gurobipy import GRB
import math
import time

def RCI_inequalities(m, where):
    m._numCallBack += 1
    
    if where != GRB.callback.MIPNODE:
        return
    if m.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
        return
    
    start = time.time()
    G = m._G
    xval = m.cbGetNodeRel(m._x)
    sval = m.cbGetNodeRel(m._s)
    m_rci = gp.Model()   
    m_rci.Params.OutputFlag = 0
    # Set a time limit
    t = m_rci.addVars(G.nodes, vtype = GRB.BINARY, name = 't')
    h = m_rci.addVars(m._primalgraph.edges, vtype = GRB.CONTINUOUS, name = 'h')   
    m_rci.setObjective(gp.quicksum(xval[e]*h[e] for e in m._primalgraph.edges) + gp.quicksum(sval[i]*t[i] for i in G.nodes) - gp.quicksum(m._p[i]*t[i] for i in G.nodes)/m._U, GRB.MINIMIZE)
    
    # Add RCI constraints
    for index in range(len(m._primaledges)):
        real_edge = m._primaledges[index]
        i = real_edge[0]
        j = real_edge[1]
        m_rci.addConstr(t[i] - t[j] <= h[index ,i])
        m_rci.addConstr(t[j] - t[i] <= h[index, j])
        
    m_rci.addConstr(gp.quicksum(m._p[i]*t[i] for i in G.nodes) >= m._U + 1)
    
    m_rci.update()   
    m_rci.optimize()
    
    if m_rci.Status == GRB.INFEASIBLE or m_rci.Status == GRB.INF_OR_UNBD:
        m._callBackTime += time.time() - start
        return
   
    # Retrieve optimization results for lazy cuts
    R = [i for i in G.nodes if t[i].X > 0.5]   
    H_edges = [e for e in m._primalgraph.edges if h[e].X > 0.5]
    LHS = sum(xval[edge] for edge in H_edges) + sum(sval[i] for i in R) 
    RHS = math.ceil(sum(m._p[i] for i in R)/m._U)
    
    # Tolerance parameter
    epsilon = -0.0001
                    
    # Add the cut to Williams's model is the constraint is violated
    if LHS < RHS + epsilon:
        m.cbLazy(gp.quicksum(m._x[edge] for edge in H_edges) + gp.quicksum(m._s[i] for i in R) >= RHS)
        m._numLazy += 1
    m._callBackTime += time.time() - start