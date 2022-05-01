#Most code due to Lorenzo Najt
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from gerrychain import Graph
import geopandas as gpd
import math

g = Graph.from_json("C:/Users/hamid/Downloads/majority-minority-master/majority-minority-master/samuel_code/county/json/LA_counties.json")
df = gpd.read_file("C:/Users/hamid/Downloads/majority-minority-master/majority-minority-master/samuel_code/county/shape/LA_counties.shp")

centroids = df.centroid
c_x = centroids.x
c_y = centroids.y
#shape = True

nlist = list(g.nodes())
n = len(nlist)


#pos = nx.kamada_kawai_layout(g)
#if shape:
pos = {node:(c_x[node],c_y[node]) for node in g.nodes}

#print("positions: ", pos)

#pos = nx.planar_layout(g)

for node in g.nodes():
    g.nodes[node]["pos"] = np.array(pos[node])
    if g.nodes[node]["NAME20"] == "Camas":
        g.nodes[node]["pos"] = [-114.80577687,  43.25]
    if g.nodes[node]["NAME20"] == "Minidoka":
        g.nodes[node]["pos"] = [-113.9374618, 42.85425972] 
    if g.nodes[node]["NAME20"] == "Lewis":
        g.nodes[node]["pos"] = [-116.02632612, 46.33699339]     
        #print("County is ", g.nodes[node]["NAME20"], " with position ", g.nodes[node]["pos"])


def compute_rotation_system(graph):
    #Graph nodes must have "pos"
    #The rotation system is  clockwise (0,2) -> (1,1) -> (0,0) around (0,1)
    for v in graph.nodes():
        #if v == artificial_martin: continue
        graph.nodes[v]["pos"] = np.array(graph.nodes[v]["pos"])
    
    for v in graph.nodes():
        #if v == artificial_martin: continue
        locations = []
        neighbor_list = list(graph.neighbors(v))
        for w in neighbor_list:
            #if w == artificial_martin: continue
            locations.append(graph.nodes[w]["pos"] - graph.nodes[v]["pos"])
        angles = [float(np.arctan2(x[0], x[1])) for x in locations]
        neighbor_list.sort(key=dict(zip(neighbor_list, angles)).get)
        #sorted_neighbors = [x for _,x in sorted(zip(angles, neighbor_list))]
        rotation_system = {}
        for i in range(len(neighbor_list)):
            rotation_system[neighbor_list[i]] = neighbor_list[(i + 1) % len(neighbor_list)]
        graph.nodes[v]["rotation"] = rotation_system
    return graph

def transform(x):
    #takes x from [-pi, pi] and puts it in [0,pi]
    if x >= 0:
        return x
    if x < 0:
        return 2 * np.pi + x
    


def is_clockwise(graph,face, average):
    #given a face (with respect to the rotation system computed), determine if it belongs to a the orientation assigned to bounded faces
    angles = [transform(float(np.arctan2(graph.nodes[x]["pos"][0] - average[0], graph.nodes[x]["pos"][1] - average[1])))  for x in face]
    first = min(angles)
    rotated = [x - first for x in angles]
    next_smallest = min([x for x in rotated if x != 0])
    ind = rotated.index(0)
    if rotated[(ind + 1)% len(rotated)] == next_smallest:
        return False
    else:
        return True

def cycle_around_face(graph, e):
    face = list([e[0], e[1]])
    last_point = e[1]
    current_point = graph.nodes[e[1]]["rotation"][e[0]]
    next_point = current_point
    while next_point != e[0]:
        face.append(current_point)
        next_point = graph.nodes[current_point]["rotation"][last_point]
        last_point = current_point
        current_point = next_point
    return face


def compute_face_data(graph):
    #graph must already have a rotation_system
    faces = []
    #faces will stored as sets of vertices

    for e in graph.edges():
        #need to make sure you get both possible directions for each edge..
        
        face = cycle_around_face(graph, e)
        faces.append(tuple(face))
        face = cycle_around_face(graph, [ e[1], e[0]])
        faces.append(tuple(face))
    #detect the unbounded face based on orientation
    bounded_faces = []
    for face in faces:
        run_sum = np.array([0,0]).astype('float64')
        for x in face:
            run_sum += np.array(graph.nodes[x]["pos"]).astype('float64')
        average = run_sum / len(face)
        if is_clockwise(graph,face, average):
            bounded_faces.append(face)    
    faces_set = [frozenset(face) for face in bounded_faces]
    graph.graph["faces"] = set(faces_set)
    return graph

def compute_all_faces(graph):
        #graph must already have a rotation_system
    faces = []
    #faces will stored as sets of vertices

    for e in graph.edges():
        #need to make sure you get both possible directions for each edge..
        
        face = cycle_around_face(graph, e)
        faces.append(tuple(face))
        face = cycle_around_face(graph, [ e[1], e[0]])
        faces.append(tuple(face))
    
    #This overcounts, have to delete cyclic repeats now:
        
    sorted_faces = list(set([tuple(canonical_order(graph,x)) for x in faces]))
    cleaned_faces = [ tuple([ y for y in F]) for F in sorted_faces]
    graph.graph["faces"] = cleaned_faces
    return graph

def canonical_order(graph, face):
    '''
    Outputs the coordinates of the nodes of the face in a canonical order
    in particular, the first one is the lex-min. 
    
    You need to use the graph structure to make this work
    '''
    
    lex_sorted_nodes = sorted(face)
    first_node = lex_sorted_nodes[0]
    cycle_sorted_nodes = [first_node]
    local_cycle = nx.subgraph( graph, face)

    #Compute the second node locally based on angle orientation
    
    v = first_node
    locations = []
    neighbor_list = list(local_cycle.neighbors(v))
    for w in neighbor_list:
        locations.append(graph.nodes[w]["pos"] - graph.nodes[v]["pos"])
    angles = [float(np.arctan2(x[1], x[0])) for x in locations]
    neighbor_list.sort(key=dict(zip(neighbor_list, angles)).get)
    
    second_node = neighbor_list[0]
    cycle_sorted_nodes.append(second_node)
    ##Now compute a canonical ordering of local_cycle, clockwise, starting
    ##from first_node
    
  
    while len(cycle_sorted_nodes) < len(lex_sorted_nodes):
        
        v = cycle_sorted_nodes[-1]
        neighbor_list = list(local_cycle.neighbors(v))
        neighbor_list.remove(cycle_sorted_nodes[-2])
        cycle_sorted_nodes.append(neighbor_list[0])
    
    return cycle_sorted_nodes


def delete_copies_up_to_permutation(array):
    '''
    Given an array of tuples, return an array consisting of one representative
    for each element in the orbit of the reordering action.
    '''
    
    cleaned_array = list(set([tuple(canonical_order(x)) for x in array]))
    
    return cleaned_array

def face_refine(graph):
    #graph must already have the face data computed
    #this adds a vetex in the middle of each face, and connects that vertex to the edges of that face...
    
    for face in graph.graph["faces"]:
        graph.add_node(face)
        location = np.array([0,0]).astype("float64")
        for v in face:
            graph.add_edge(face, v)
            location += graph.nodes[v]["pos"].astype("float64")
        graph.nodes[face]["pos"] = location / len(face)
    return graph

def edge_refine(graph):
    edge_list = list(graph.edges())
    for e in edge_list:
        graph.remove_edge(e[0],e[1])
        graph.add_node(str(e))
        location = np.array([0,0]).astype("float64")
        for v in e:
            graph.add_edge(str(e), v)
            location += graph.nodes[v]["pos"].astype("float64")
        graph.nodes[str(e)]["pos"] = location / 2
    return graph

def refine(graph):
    graph = compute_rotation_system(graph)
    graph = compute_face_data(graph)
    graph = face_refine(graph)
    return graph

def depth_k_refine(graph,k):
    graph.name = graph.name + str("refined_depth") + str(k)
    for i in range(k):
        graph = refine(graph)
    return graph

def depth_k_barycentric(graph, k):
    graph.name = graph.name + str("refined_depth") + str(k)
    for i in range(k):
        graph = barycentric_subdivision(graph)
    return graph

def barycentric_subdivision(graph):
    #graph must already have the face data computed
    #this adds a vetex in the middle of each face, and connects that vertex to the edges of that face...
    graph = edge_refine(graph)
    graph = refine(graph)
    return graph
    

def restricted_planar_dual(graph):
    #computes dual without unbounded face
    graph = compute_rotation_system(graph)
    graph = compute_face_data(graph)
    dual_graph = nx.Graph()
    counter = 0
    for face in graph.graph["faces"]:
        if face == frozenset({0, 2, 3, 4, 6, 7, 8, 11, 12, 13, 14, 16, 18, 21, 23, 25, 27, 30, 31, 32, 36, 37, 39, 42}) or face == frozenset({0, 2, 3, 4, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 21, 23, 25, 27, 30, 31, 32, 36, 37, 39, 42}) or face == frozenset({34, 35, 15, 19, 20}) or face == frozenset({41, 34, 2, 28}) or face == frozenset({41, 2, 19, 34}) or face == frozenset({19, 34, 35, 20}) or face == frozenset({33, 2, 34, 41}) or face == frozenset({34, 35, 20, 29}) or face == frozenset({34, 35, 20, 15}) or face == frozenset({2, 34, 41, 19, 28}): 
            print("We found a bad face!")
            continue
        print("face: ", face)
        dual_graph.add_node(face)
        location = np.array([0,0]).astype("float64")
        for v in face:
            location += graph.nodes[v]["pos"].astype("float64")
        dual_graph.nodes[face]["pos"] = location / len(face)
        dual_graph.nodes[face]["label"] = counter
        counter += 1
    ##handle edges
    for e in graph.edges():
        for face in dual_graph.nodes():
            for face2 in dual_graph.nodes():
                if face != face2 and (dual_graph.nodes[face]["label"] < dual_graph.nodes[face2]["label"]):
                    if (e[0] in face) and (e[1] in face) and (e[0] in face2) and (e[1] in face2):
                        dual_graph.add_edge(face, face2)
                        print(e[0], e[1], " and ", dual_graph.nodes[face]["label"], dual_graph.nodes[face2]["label"])
    return dual_graph



def draw_with_location(graph,c='k',ns=100,w=3,ec='b'):
#    for x in graph.nodes():
#        graph.node[x]["pos"] = [graph.node[x]["X"], graph.node[x]["Y"]]

    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size = ns, width = w, node_color=c,edge_color=ec)


graph = compute_rotation_system(g)
graph = compute_face_data(graph) 

dual = restricted_planar_dual(graph)

#bad_vertex = -1
#for vertex in dual.nodes():
 #   print("vertex ", vertex, " has degree ", dual.degree[vertex])
  #  if dual.degree[vertex] == 21: bad_vertex = vertex
        
        


#print("start print edges")



#print("finish print edges")

# label of outer face
outer = len(dual.nodes)

# add edges from the outer face
counter = 0
for edge in graph.edges:
    if g.nodes[edge[0]]["boundary_node"] and g.nodes[edge[1]]["boundary_node"]:
        face_counter = 0
        for face0 in dual.nodes():
            if edge[0] in face0 and edge[1] in face0:
                face_counter += 1
        if  face_counter > 1: continue 
        if face_counter == 1:      
            for face in dual.nodes():
                if (edge[0] in face) and (edge[1] in face):
                #if dual_graph.nodes[face]["label"] <= dual_graph.nodes[face2]["label"]:
                    print(edge[0], edge[1], " and ", dual.nodes[face]["label"], outer)
                    counter += 1
        if face_counter == 0:
            print(edge[0], edge[1], " and ", outer, outer)
            counter += 1            
#print("# of dual edges added: ", counter)  

# remove extra edges for LA
for (u,v) in g.edges:
    if (g.nodes[u]["NAME20"] == "St. Martin" and g.nodes[v]["NAME20"] == "Assumption") or (g.nodes[v]["NAME20"] == "St. Martin" and g.nodes[u]["NAME20"] == "Assumption"): 
        g.remove_edge(u, v)
    if (g.nodes[u]["NAME20"] == "St. Martin" and g.nodes[v]["NAME20"] == "St. Mary") or (g.nodes[v]["NAME20"] == "St. Martin" and g.nodes[u]["NAME20"] == "St. Mary"):   
        g.remove_edge(u, v)
        
# add new st martin
#artificial_martin = len(g.nodes)

#g.add_node(artificial_martin)

#for vertex in g.nodes():
 #   if vertex == artificial_martin: continue
    #print(g.nodes[vertex]["NAME20"])
  #  if g.nodes[vertex]["NAME20"] == "St. Mary" or g.nodes[vertex]["NAME20"] == "Assumption":        
   #     g.add_edge(vertex, artificial_martin)

        
print("# of primal vertices is: ", len(g.nodes))
print("# of primal edges is: ", len(g.edges))      

print("# of dual vertices is: ", len(dual.nodes) + 1)
print("# of dual edges is: ", len(dual.edges)+counter)

print("Euler's check? ", len(g.nodes) + len(dual.nodes) - 1 == len(g.edges))

print("Is the primal graph planar? ", nx.check_planarity(g, counterexample=False)[0])

print("Here are faces: ")

#for u, v in g.edges:
 #   print("Adjacent counties ", g.nodes[u]["NAME20"], ", ", g.nodes[v]["NAME20"])

'''
k = 7       
population = [g.nodes[i]['P0010001'] for i in g.nodes()]    
deviation = 0.01
L = math.ceil((1-deviation/2)*sum(population)/k)
U = math.floor((1+deviation/2)*sum(population)/k)
print("L =",L,", U =",U,", k =",k)

for vertex in dual.nodes():
    if sum(g.nodes[i]['P0010001'] for i in list(vertex)) > U:
        print(list(vertex))
'''

#dual.remove_node(bad_vertex)

plt.figure()
draw_with_location(graph,'b',50,1,'b')
draw_with_location(dual,'r',50,1,'r')
plt.show()

print("Is primal planar? ", list(nx.check_planarity(g, counterexample=True)[1]))
## 
#m= 3
#graph = nx.grid_graph([m,m])
#graph.name = "grid_size:" + str(m)
#for x in graph.nodes():
#    
#    graph.node[x]["pos"] = np.array([x[0], x[1]])
#
###graph = depth_k_refine(graph,0)
##graph = depth_k_barycentric(graph, 4)
#draw_with_location(graph)
#graph = compute_rotation_system(graph)
#graph = compute_face_data(graph) 
##print(len(graph.graph["faces"]))
##
#dual = restricted_planar_dual(graph)
#draw_with_location(dual)