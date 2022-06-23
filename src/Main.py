###########################
# Imports
###########################
import gurobipy as gp
from gurobipy import GRB
import csv
from csv import DictWriter
import networkx as nx
import Hess
import Williams
import matplotlib.pyplot as plt
from gerrychain import Graph
import geopandas as gpd
import face_finder
import sys
import json
import os
from datetime import date
import read
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
print("Reading config from", config_filename)
  
config_filename_wo_extension = config_filename.rsplit('.', 1)[0]
configs_file = open(config_filename, 'r')
batch_configs = json.load(configs_file)
configs_file.close()

# create directory for results
path = os.path.join("..", "results_for_" + config_filename_wo_extension) 
os.mkdir(path) 

# print results to csv file
today = date.today()
today_string = today.strftime("%Y_%b_%d") # Year_Month_Day, like 2019_Sept_16
results_filename = "../results_for_" + config_filename_wo_extension + "/results_" + config_filename_wo_extension + "_" + today_string + ".csv"
fields = ["State", "Level", "Model", "Primal Vertices", "Edges", "Districts", "Number of Max Clique",
          "Run Time (Seconds)", "Callback Time (Seconds)", 'Time without Callbacks (Seconds)',
          "Branch and Bound Nodes", "Number of Callbacks", "Number of Lazy Constraints Added",
          "Objective Value", "Objective Bound", "heur_obj", "heur_time", "heur_iter"]

################################################
# Summarize computational results to csv file
################################################ 
def append_dict_as_row(file_name, dict_of_elem, field_names):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        # Add dictionary as wor in the csv
        dict_writer.writerow(dict_of_elem)

###########################
# Hard-coded inputs
###########################
number_of_congressional_district = {"AL": 7, "AR": 4, "IA": 4, "KS": 4, "ME": 2, "MS": 4, "NE": 3, "NM": 3, "WV": 2, "ID": 2}

#########################################
# Run An Instance using Williams' model
#########################################
def run_williams(m):
    return Williams.Williams_model(m)

####################################
# Run An Instance using Hess' model
####################################
def run_Hess(m):
    return Hess.Hess_model(m)

################################################
# Draws districts and saves to png file
################################################ 
def export_to_png(m, assignment, filename):
    G = m._G
    df = m._df
    state = m._state
    level = m._level
    new_assignment = [ -1 for u in G.nodes ]
    
    if level == "tract" and ( model == "Williams_cuts" or model =="Williams_flow"):
        G_fig = Graph.from_json("data/tract/dual_graphs/" + state + "_tracts.json")
    else:
        G_fig = G

    for vertex in G_fig.nodes():
        geoID = G_fig.nodes[vertex]["GEOCODE"]
        for u in G_fig.nodes():
            if geoID == df['GEOID20'][u]:
                new_assignment[u] = assignment[vertex]
     
    df['assignment'] = new_assignment  
    my_fig = df.plot(column = 'assignment').get_figure()
    RESIZE_FACTOR = 3
    my_fig.set_size_inches(my_fig.get_size_inches() * RESIZE_FACTOR)
    plt.axis('off')
    my_fig.savefig(filename)

# prepare csv file by writing column headers
with open(results_filename,'w', newline = '') as csvfile:   
    writer = csv.DictWriter(csvfile, fieldnames = fields)
    writer.writeheader()
    
############################################################
# Run experiments for each config in batch_config file
############################################################
# Credit to https://github.com/hamidrezavalidi/Political-Districting-to-Minimize-Cut-Edges/blob/master/src/main.py
# Read through the config file to run all instances
for key in batch_configs.keys():
    config = batch_configs[key]
    model = config['model']
    state = config['state']
    num_district = config['num_district']
    level = config['level']
    RCI = config['RCI']
    max_clique = config["max clique"]
    heuristic = config['heuristic']
    heuristic_iter = config['heuristic_iter']
    
    # Read the primal and dual graph from the text files when running Williams model on tract level
    if level == "tract": 
        suffix = "tracts"
    elif level == "county":
        suffix = "counties"    
    
    # Read input files depending on the model and experiment level
    if level == "tract" and ( model == "Williams_cuts" or model =="Williams_flow"):
        primal_path = "C:/Users/hamid/Downloads/dualization/" + state + "_primalGraph.txt"
        dual_path = "C:/Users/hamid/Downloads/dualization/" + state + "_dualGraph.txt"
        population_path = "C:/Users/hamid/Downloads/dualization/" + state + "_population.population"
        [G, primal_dual_pairs, p] = read.read_tract_txt(primal_path, dual_path, population_path)
    else:
        G =  Graph.from_json("data/" + level + "/dual_graphs/" + state + "_" + suffix + ".json")
        p = [G.nodes[i]['P0010001'] for i in G.nodes()]
        primal_dual_pairs = 'n/a'
    df = gpd.read_file("data/" + level + "/shape_files/" + state + "_" + suffix + ".shp")
    
    # Build the model
    m = gp.Model()
    # Set a time limit
    m.setParam('TimeLimit', 3600)
    # Set gap to zero
    m.setParam('MIPGap', 0)
    m.modelSense = GRB.MINIMIZE
    total_pop = sum(p)
    L = math.ceil((total_pop/num_district)*(0.995))
    U = math.floor((total_pop/num_district)*(1.005))
    
    print("Solving " + state + " with L = " + str(L) + " and U = " + str(U) + " at " + level + " level under " + model + " model")
    
    # Attach parameters to the model
    m._state = state
    m._level = level
    m._model = model
    m._total_pop = total_pop 
    m._U = U
    m._L = L
    m._p = p
    m._G = G
    m._df = df
    m._pdp = primal_dual_pairs
    m._k = num_district
    m._RCI = RCI
    m._maxclique = max_clique
    if max_clique:
        m._numMaxClique = 0
    else:
        m._numMaxClique = 'n/a'
    m._numCallBack = 0
    m._numLazy = 0
    m._callBackTime = 0
    # read heuristic solution from an external file
    m._heuristic = heuristic
    m._hiter = heuristic_iter
    
    # Initialize the result dictionary used for reporting experiment results
    result = {}
    
    # Read and construct heuristic warm-start solutions if heuristic is turned on
    if heuristic:
        # How to deplace the 100 in the file location with an input iteration
        heuristic_file = open('../heuristic-results/' + str(heuristic_iter) + '-iterations/heur_' + state + "_" + level + ".json", 'r')
        heuristic_dict = json.load(heuristic_file)     
        heuristic_districts = [ [node['index'] for node in heuristic_dict['nodes'] if node['district'] == j ] for j in range(m._k) ]
        m._hdistricts = heuristic_districts
        heuristic_cut_edges = heuristic_dict['cut edges']
        m._cuts = heuristic_cut_edges
        result['heur_obj'] = heuristic_dict['obj']
        result['heur_time'] = heuristic_dict['time']
        result['heur_iter'] = heuristic_iter
        heuristic_label = "w_heuristic"
    else:
        heuristic_districts = None
        result['heur_obj'] = 'n/a'
        result['heur_time'] = 'n/a'
        result['heur_iter'] = 'n/a'
        heuristic_label = "wo_heuristic"
    
    # Run the instance
    if model == "Hess":
        [run_time, node_count, forest, obj_val, obj_bound] = run_Hess(m)
    elif model == "Williams_cuts" or model == "Williams_flow":
        m._populationparam = model[9:]
        m._callback = None
        [run_time, node_count, forest, directed_forest, obj_val, obj_bound] = run_williams(m)
        
    # Compile the results    
    result['State'] = state
    result['Level'] = level
    result['Model'] = model
    result['Primal Vertices'] = len(G.nodes)
    result['Edges'] = len(G.edges)
    result['Districts'] = m._k
    result['Number of Max Clique'] = m._numMaxClique
    result['Run Time (Seconds)'] = run_time
    result['Callback Time (Seconds)'] = m._callBackTime
    result['Time without Callbacks (Seconds)'] = run_time - m._callBackTime
    result['Branch and Bound Nodes'] = node_count
    result['Number of Callbacks'] = m._numCallBack
    result['Number of Lazy Constraints Added'] = m._numLazy
    result['Objective Value'] = obj_val
    result['Objective Bound'] = obj_bound
    append_dict_as_row(results_filename, result, fields)
    
    # Output districting figures if feasible solutions are found within the givn time limit
    if node_count > 0:
        assignment = [ -1 for u in G.nodes ]
        if model == "Williams_cuts" or model =="Williams_flow":
            components = sorted(list(nx.connected_components(forest)))
            for node in sorted(list(forest.nodes)):
                for index in range(num_district):
                    if node in components[index]:
                        assignment[node] = index
        else:
            for i in range(num_district):
                for node in forest[i]:
                    assignment[node] = i
        
        png_filename = path + "/" + state + "_" + model + "_" + level + "_" + heuristic_label + '.png'
        export_to_png(m, assignment, png_filename)
        
        # Output an additional arowed map when districted using William's model and on county level
        if level == "county" and ( model == "Williams_cuts" or model == "Williams_flow"):
            face_finder.draw_with_location(directed_forest, df, 'k', 100, 3,'r')
            arrow_graph_path =  path + "/" + state + "_" + model + "_" + level + "_" + heuristic_label + '_arrows.png'
            plt.savefig(arrow_graph_path)