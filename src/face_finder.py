#Most code due to Lorenzo Najt
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def compute_rotation_system(graph, pos):
    #The rotation system is clockwise (0,2) -> (1,1) -> (0,0) around (0,1)
    for v in graph.nodes():
        graph.nodes[v]["pos"] = np.array(graph.nodes[v]["pos"])
    
    for v in graph.nodes():
        locations = []
        neighbor_list = list(graph.neighbors(v))
        for w in neighbor_list:
            locations.append(graph.nodes[w]["pos"] - graph.nodes[v]["pos"])
        angles = [float(np.arctan2(x[0], x[1])) for x in locations]
        neighbor_list.sort(key=dict(zip(neighbor_list, angles)).get)
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
    cleaned_faces = [ tuple([y for y in F]) for F in sorted_faces]
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
    local_cycle = nx.subgraph(graph, face)

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

def refine(graph,pos):
    graph = compute_rotation_system(graph,pos)
    graph = compute_face_data(graph)
    graph = face_refine(graph)
    return graph

def depth_k_refine(graph,k,pos):
    graph.name = graph.name + str("refined_depth") + str(k)
    for i in range(k):
        graph = refine(graph,pos)
    return graph

def depth_k_barycentric(graph, k, pos):
    graph.name = graph.name + str("refined_depth") + str(k)
    for i in range(k):
        graph = barycentric_subdivision(graph,pos)
    return graph

def barycentric_subdivision(graph,pos):
    #graph must already have the face data computed
    #this adds a vetex in the middle of each face, and connects that vertex to the edges of that face...
    graph = edge_refine(graph)
    graph = refine(graph,pos)
    return graph
    

def restricted_planar_dual(graph,df,state):
    centroids = df.centroid
    c_x = centroids.x
    c_y = centroids.y
    pos = {node:(c_x[node],c_y[node]) for node in graph.nodes}
    
    for node in graph.nodes():
        graph.nodes[node]["pos"] = np.array(pos[node])
        if state == "ID":
            # Three nodes in ID's graph is poorly places and results in malformed dual graph, need to manually adjust their position
            if graph.nodes[node]["NAME20"] == "Camas":
                graph.nodes[node]["pos"] = np.array([-114.80577687,  43.25])
            if graph.nodes[node]["NAME20"] == "Minidoka":
                graph.nodes[node]["pos"] = np.array([-113.9374618, 42.85425972])
            if graph.nodes[node]["NAME20"] == "Lewis":
                graph.nodes[node]["pos"] = np.array([-116.32632612, 46.33699339])
        if state == "MT":
            if graph.nodes[node]["NAME20"] == "Lake":
                graph.nodes[node]["pos"] = [-114.24938,  47.6459043]
            if graph.nodes[node]["NAME20"] == "Treasure":
                graph.nodes[node]["pos"] = [-107.4,  45.9]

    #computes dual without unbounded face
    graph = compute_rotation_system(graph, pos)
    graph = compute_face_data(graph)
    dual_graph = nx.Graph()
    counter = 0
    for face in graph.graph["faces"]:
        if face == frozenset({3, 4, 7, 8, 12, 13, 19, 21, 24, 28, 29, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 51, 54}) or face == frozenset({0, 2, 3, 4, 6, 7, 8, 11, 12, 13, 14, 16, 18, 21, 23, 25, 27, 30, 31, 32, 36, 37, 39, 42}) or face == frozenset({0, 2, 3, 4, 6, 7, 8, 11, 12, 13, 14, 16, 17, 18, 21, 23, 25, 27, 30, 31, 32, 36, 37, 39, 42}) or face == frozenset({34, 35, 15, 19, 20}) or face == frozenset({41, 34, 2, 28}) or face == frozenset({41, 2, 19, 34}) or face == frozenset({19, 34, 35, 20}) or face == frozenset({33, 2, 34, 41}) or face == frozenset({34, 35, 20, 29}) or face == frozenset({34, 35, 20, 15}) or face == frozenset({2, 34, 41, 19, 28}): 
            continue
        dual_graph.add_node(face)
        location = np.array([0,0]).astype("float64")
        for v in face:
            location += graph.nodes[v]["pos"].astype("float64")
        dual_graph.nodes[face]["pos"] = location / len(face)
        dual_graph.nodes[face]["label"] = counter
        counter += 1
    ##handle edges
    primal_dual_pair = []
    for e in graph.edges():
        for face in dual_graph.nodes():
            for face2 in dual_graph.nodes():
                if face != face2 and (dual_graph.nodes[face]["label"] < dual_graph.nodes[face2]["label"]):
                    if (e[0] in face) and (e[1] in face) and (e[0] in face2) and (e[1] in face2):
                        dual_graph.add_edge(face, face2)
                        primal_dual_pair.append([[e[0], e[1]], [dual_graph.nodes[face]["label"], dual_graph.nodes[face2]["label"]]])

    draw_with_location(graph, df,'b',50,1,'b')
    draw_with_location(dual_graph, df,'r',50,1,'r')
    # label of outer face
    outer = len(dual_graph.nodes)
    # add edges from the outer face
    counter = 0
    for edge in graph.edges:
        if graph.nodes[edge[0]]["boundary_node"] and graph.nodes[edge[1]]["boundary_node"]:
            face_counter = 0
            for face0 in dual_graph.nodes():
                if edge[0] in face0 and edge[1] in face0:
                    face_counter += 1
            if  face_counter > 1: continue 
            if face_counter == 1:      
                for face in dual_graph.nodes():
                    if (edge[0] in face) and (edge[1] in face):
                        counter += 1
                        primal_dual_pair.append([[edge[0], edge[1]], [dual_graph.nodes[face]["label"], outer]])
            if face_counter == 0:
                counter += 1   
                primal_dual_pair.append([[edge[0], edge[1]], [outer, outer]])
    plt.figure()    
    return dual_graph, primal_dual_pair

def draw_with_location(graph, df,c='k',ns=100,w=3,ec='r'):

    nx.draw(graph, pos=nx.get_node_attributes(graph, 'pos'), node_size = ns, width = w, node_color=c,edge_color=ec)