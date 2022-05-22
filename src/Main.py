###########################
# Imports
###########################
import gurobipy as gp
from gurobipy import GRB
import csv
import networkx as nx
import Hess
import Williams_test
import matplotlib.pyplot as plt
from gerrychain import Graph
import geopandas as gpd
import face_finder
import sys
import json
import os
from datetime import date
import read
import Population_cuts
import math

###############################################
# Read configs/inputs and set parameters
############################################### 
# Credit to https://github.com/hamidrezavalidi/Political-Districting-to-Minimize-Cut-Edges/blob/master/src/main.py
if len(sys.argv)>1:
    # name your own config file in command line, like this: 
    #       python main.py usethisconfig.json
    # to keep logs of the experiments, redirect to file, like this:
    #       python main.py usethisconfig.json 1>>log_file.txt 2>>error_file.txt
    config_filename = sys.argv[1] 
else:
    config_filename = 'config_tract.json' # default
    
print("Here is the config name: ", config_filename)    
    
print("Reading config from",config_filename)    
config_filename_wo_extension = config_filename.rsplit('.',1)[0]
configs_file = open(config_filename,'r')
batch_configs = json.load(configs_file)
configs_file.close()


# create directory for results
path = os.path.join("..", "results_for_" + config_filename_wo_extension) 
os.mkdir(path) 

# print results to csv file
today = date.today()
today_string = today.strftime("%Y_%b_%d") # Year_Month_Day, like 2019_Sept_16
results_filename = "../results_for_" + config_filename_wo_extension + "/results_" + config_filename_wo_extension + "_" + today_string + ".csv"
fields = ["State", "level", "Model", "Primal Vertices", "Edges", "Districts", "Run Time (Seconds)", "Branch and Bound Nodes", "Objective Value", "Objective Bound"]

################################################
# Summarize computational results to csv file
################################################
def write_to_csv(state_results, filename, fields, models, levels):
    rows = []
    for i in range(len(state_results)):
        model = models[i]
        level = levels[i]
        if model == "Williams_cuts" or model == "Williams_flow":
            [state, num_primal_nodes, num_edges, num_districts, run_time, node_count, _, _, val, bound] = state_results[i]
        else:
            [state, num_primal_nodes, num_edges, num_districts,run_time, node_count, _, val, bound] = state_results[i]
        row = [state, level, model, num_primal_nodes, num_edges, num_districts,run_time, node_count, val, bound]
        rows.append(row)

    with open(filename, 'w', newline='') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        # writing the data rows
        csvwriter.writerows(rows)


###########################
# Hard-coded inputs
###########################
number_of_congressional_district = {"AL": 7, "AR": 4, "IA": 4, "KS": 4, "ME": 2, "MS": 4, "NE": 3, "NM": 3, "WV": 2, "ID": 2}
#states_rows = {"AL": [67, 106, 171, 7], "AR": [75, 119, 192, 4], "IA": [99, 125, 222, 4], "KS": [105, 160, 263, 4],
#                "ME": [16, 20, 34, 2], "MS": [82, 122, 202, 4], "NE": [93, 140, 231, 3], "NM": [33, 47, 78, 3],
#               "WV": [55, 72, 125, 2], "ID":[44, 60, 102, 2]}

    

###########################
# Run An Instance using Williams' model
###########################
def run_williams(state,num_district,G,level,df,primal_dual_pairs,m):
    return Williams_test.Williams_model(num_district, state,G,level,df,primal_dual_pairs,m)

###########################
# Run An Instance using Hess' model
###########################
def run_Hess(state,num_district,G,m):
    return Hess.Hess_model(state,G,num_district,m)


################################################
# Draws districts and saves to png file
################################################ 

def export_to_png(G, df, assignment, filename,level):
    new_assignment = [ -1 for u in G.nodes ]
    
    #for col in df.columns:
    #    print(col)
    
    #for vertex in G.nodes():
     #   print(G.nodes[vertex]["GEOCODE"], df['GEOID10'][vertex])
    
    #for j in range(len(districts)):
    if level == "County":    
        for vertex in G.nodes():
            
            geoID = G.nodes[vertex]["GEOCODE"]
            #print(geoID)
            for u in G.nodes():
                #print("df: ", df['GEOID10'][u])
                if geoID == df['GEOID20'][u]:
                    new_assignment[u] = assignment[vertex]
    #elif level == "Tract":
     #   for u in G.nodes():
      #      new_assignment[u] = assignment[u] 
         
                
    
    #if min(assignment[v] for v in G.nodes) < 0:
     #   print("Error: did not assign all nodes in district map png.")
    #else:
    #print("assignment: ", assignment)
    #print("new assignment: ", new_assignment)
    if level == "County": 
        df['assignment'] = new_assignment
    elif level == "Tract": 
        df['assignment'] = assignment
    
    my_fig = df.plot(column='assignment').get_figure()
    RESIZE_FACTOR = 3
    my_fig.set_size_inches(my_fig.get_size_inches()*RESIZE_FACTOR)
    plt.axis('off')
    my_fig.savefig(filename)


############################################################
# Run experiments for each config in batch_config file
############################################################
# Credit to https://github.com/hamidrezavalidi/Political-Districting-to-Minimize-Cut-Edges/blob/master/src/main.py
results = []
models = []
levels = []
for key in batch_configs.keys():
    config = batch_configs[key]
    model = config['model']
    models.append(model)
    state = config['state']
    num_district = config['num_district']
    level = config['level']
    levels.append(level)
    if level == "County":
        G = Graph.from_json("C:/Users/hamid/Downloads/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/data/county/dual_graphs/" + state + "_counties.json")
        p = [G.nodes[i]['P0010001'] for i in G.nodes()]
        df = gpd.read_file("C:/Users/hamid/Downloads/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/data/county/shape_files/"+state+"_counties.shp")
        primal_dual_pairs=[]
    else:
        primal_path = "C:/Users/hamid/Downloads/dualization/"+state+"_primalGraph.txt"
        dual_path = "C:/Users/hamid/Downloads/dualization/"+state+"_dualGraph.txt"
        population_path = "C:/Users/hamid/Downloads/dualization/"+state+"_population.population"
        [G, primal_dual_pairs, p] = read.read_county_txt(primal_path,dual_path,population_path)
        df = gpd.read_file("C:/Users/hamid/Downloads/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/data/tract/shape_files/"+state+"_tracts.shp")
    num_primal_node = len(G.nodes)
    num_edge = len(G.edges)
    
    m = gp.Model()
    # Set a time limit
    m.setParam('TimeLimit', 3600)
    # Set gap to zer
    m.setParam('MIPGap', 0)
    m.modelSense = GRB.MINIMIZE
    total_pop = sum(p)
    L = math.ceil((total_pop/num_district)*(0.995))
    U = math.floor((total_pop/num_district)*(1.005))
    print("Solving " + state + " with L = " + str(L) + " and U = " + str(U) + " on " + level + " level under " + model + " model")
    m._U = U
    m._L = L
    m._p = p
    m._G = G
    m._k = num_district
    if model == "Hess":
        result = run_Hess(state,num_district,G,m)
    elif model == "Williams_cuts" or model == "Williams_flow":
        m._populationparam = model[9:]
        m._callback = None
        result = run_williams(state,num_district,G,level,df,primal_dual_pairs,m)
        directed_forest = result[3]
    forest = result[2]
    node_count = result[1]
    result.insert(0,num_district)
    result.insert(0,num_edge)
    result.insert(0,num_primal_node)
    result.insert(0,state)
    results.append(result)
    if node_count > 0:
        if model == "Williams_cuts" or model =="Williams_flow":
            assignment = []
            components = sorted(list(nx.connected_components(forest)))
            population = [0]*len(components)
            for node in sorted(list(forest.nodes)):
                for index in range(num_district):
                    if node in components[index]:
                        population[index] += p[node]
                        real_index = index + 1
                        assignment.append(real_index)
            print("Components populations:",population)
        else:
            assignment = [0]*len(G.nodes)
            for i in range(1,num_district+1):
                for node in forest[i-1]:
                    assignment[node] = i
        png_filename = path + "/" + state + "_" + model + '_map.png'
        export_to_png(G, df, assignment, png_filename, level)
        if level == "County" and ( model == "Williams_cuts" or model == "Williams_flow"):
            face_finder.draw_with_location(directed_forest,df,'k',100,3,'r')
            arrow_graph_path = path + "/" + state + "_arrows.png"
            plt.savefig(arrow_graph_path)
write_to_csv(results, results_filename, fields, models, levels)