###########################
# Imports
###########################
import csv
import networkx as nx
import Hess
import Williams_test
import matplotlib.pyplot as plt
from gerrychain import Graph
import geopandas as gpd
import face_finder

################################################
# Summarize computational results to csv file
################################################
def write_to_csv(state_rows, state_results, filename, fields, model):
    rows = []
    # Create an index to keep track of the
    result_index = 0
    for state in state_rows:
        if model == "Williams":
            [run_time, node_count, _, _, val, bound] = state_results[result_index]
        else:
            [run_time, node_count, _, val, bound] = state_results[result_index]
        result_index += 1
        row = states_rows[state]
        row.insert(0, state)
        row.append(run_time)
        row.append(node_count)
        row.append(val)
        row.append(bound)
        rows.append(row)

    with open(filename, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        # writing the data rows
        csvwriter.writerows(rows)

"""
def write_to_txt(forest, num_district, file_location, model):
    output_list = []
    if model == "Williams":
        components = sorted(list(nx.connected_components(forest)))
        for node in sorted(list(forest.nodes)):
            for index in range(num_district):
                if node in components[index]:
                    real_index = index + 1
                    output_list.append([str(node) + " " + str(real_index) + "\n"])
    else:
        for i in range(1,num_district+1):
            for node in forest[i-1]:
                output_list.append([str(node) + " " + str(i) + "\n"])
    with open(file_location, 'w') as f:
        for line in output_list:
            f.writelines(line)
"""

###########################
# Hard-coded inputs
###########################

#states_rows = {"AL": [67, 106, 171, 7], "AR": [75, 119, 192, 4], "IA": [99, 125, 222, 4], "KS": [105, 160, 263, 4],
#                "ME": [16, 20, 34, 2], "MS": [82, 122, 202, 4], "NE": [93, 140, 231, 3], "NM": [33, 47, 78, 3],
#               "WV": [55, 72, 125, 2], "ID":[44, 60, 102, 2]}
states_rows = {"WV": [55, 72, 125, 2]}
fields = ["State", "Primal Vertices", "Dual Vertices", "Edges", "Districts", "Run Time (Seconds)", "Branch and Bound Nodes", "Objective Value", "Objective Bound"]


###########################
# Run An Instance using Williams' model
###########################
def run_williams(state,df,p,G):
    num_district = states_rows[state][3]
    return Williams_test.Williams_model(num_district, state,df,p,G)

###########################
# Run An Instance using Hess' model
###########################
def run_Hess(state,G):
    num_district = states_rows[state][3]
    return Hess.Hess_model(state,G,num_district)


################################################
# Draws districts and saves to png file
################################################ 

def export_to_png(G, df, assignment, filename):
    new_assignment = [ -1 for u in G.nodes ]
    
    #for col in df.columns:
    #    print(col)
    
    #for vertex in G.nodes():
     #   print(G.nodes[vertex]["GEOCODE"], df['GEOID10'][vertex])
    
    #for j in range(len(districts)):
    for vertex in G.nodes():
        geoID = G.nodes[vertex]["GEOCODE"]
        #print(geoID)
        for u in G.nodes:
            #print("df: ", df['GEOID10'][u])
            if geoID == df['GEOID20'][u]:
                new_assignment[u] = assignment[vertex]
    
    #if min(assignment[v] for v in G.nodes) < 0:
     #   print("Error: did not assign all nodes in district map png.")
    #else:
    df['assignment'] = new_assignment
    my_fig = df.plot(column='assignment').get_figure()
    RESIZE_FACTOR = 3
    my_fig.set_size_inches(my_fig.get_size_inches()*RESIZE_FACTOR)
    plt.axis('off')
    my_fig.savefig(filename)


###########################
# Run the complete experiment
###########################
state_results = []
# Specify the model
model = "Williams"
for state in states_rows:
    # C:\Users\hamid\Downloads\Political-Districting-to-Minimize-Cut-Edges-master (3).zip\Political-Districting-to-Minimize-Cut-Edges-master\data\county\shape_files
    G = Graph.from_json("C:/Users/hamid/Downloads/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/data/county/dual_graphs/" + state + "_counties.json")
    p = [G.nodes[i]['P0010001'] for i in G.nodes()]
    #df = gpd.read_file("C:/Users/hamid/Downloads/Political-Districting-to-Minimize-Cut-Edges-master (3)/Political-Districting-to-Minimize-Cut-Edges-master/data/county/shape_files/"+state+"_county.shp")
    df = gpd.read_file("C:/Users/hamid/Downloads/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/data/county/shape_files/"+state+"_counties.shp")
    #face_finder.check_planarity(G, df)
    if model == "Hess":
        result = run_Hess(state,G)
    else:
        result = run_williams(state,df,p,G)
    state_results.append(result)
    forest = result[2]
    directed_forest = result[3]
    num_district = states_rows[state][3]
    if model == "Williams":
        assignment = []
        components = sorted(list(nx.connected_components(forest)))
        population = [0]*len(components)
        for node in sorted(list(forest.nodes)):
            for index in range(num_district):
                if node in components[index]:
                    population[index] += p[node]
                    real_index = index + 1
                    assignment.append(real_index)
    else:
        assignment = [0]*len(G.nodes)
        for i in range(1,num_district+1):
            for node in forest[i-1]:
                assignment[node] = i
    
    filename = "C:/Users/hamid/Downloads/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/" + state + '_map'
    export_to_png(G, df, assignment, filename)
    if model == "Williams":
        directed_forest = result[3]
        face_finder.draw_with_location(directed_forest,df,'k',100,3,'r')
    #write_to_txt(result[2],states_rows[state][3], "C:/Users/hamid/Downloads/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/" + state + "_solution.txt", model)
write_to_csv(states_rows, state_results, "C:/Users/hamid/Downloads/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/" + state  + "result.csv", fields, model)