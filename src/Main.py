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
    config_filename = 'config.json' # default
    
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
fields = ["State", "Model", "Primal Vertices", "Edges", "Districts", "Run Time (Seconds)",
          "Branch and Bound Nodes", "Objective Value", "Objective Bound"]

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
    new_assignment = [ -1 for u in G.nodes ]

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
    warm_start = config['warm_start']
     
    G =  Graph.from_json("../data/" + "dual_graphs/" + state  + "_counties.json")
    p = [G.nodes[i]['P0010001'] for i in G.nodes()]
    df = gpd.read_file("../data/" + "shape_files/" + state + "_counties.shp")
    
    # Build the model
    m = gp.Model()
    # Set gap to zero
    m.setParam('MIPGap', 0)
    # Make tolerance tight
    m.Params.IntFeasTol = 1.e-9
    m.Params.FeasibilityTol = 1.e-9
    
    m.modelSense = GRB.MINIMIZE
    total_pop = sum(p)
    L = math.ceil((total_pop/num_district)*(0.995))
    U = math.floor((total_pop/num_district)*(1.005))
    
    print("Solving " + state + " with L = " + str(L) + " and U = " + str(U) + " under " + model + " model")
    
    # Attach parameters to the model
    m._state = state
    m._model = model
    m._total_pop = total_pop 
    m._U = U
    m._L = L
    m._p = p
    m._G = G
    m._df = df
    m._k = num_district
    # read heuristic solution from an external file
    m._ws = warm_start
    
    # Initialize the result dictionary used for reporting experiment results
    result = {}
    
    # Read and construct heuristic warm-start solutions if heuristic is turned on
    if warm_start:
        # How to deplace the 100 in the file location with an input iteration
        ws_file = open('../warm_start/' + model + "_" + state + ".json", 'r')
        ws_dict = json.load(ws_file)
        if model == "Hess":
            m._partitions = ws_dict['partitions']
        elif model =="Williams":
            m._forest = ws_dict['forest']
        ws_label = "w_warmstart"
    else:
        ws_districts = None
        ws_label = "wo_warmstart"
    
    # Run the instance
    if model == "Hess":
        [run_time, node_count, partitions_dict, obj_val, obj_bound] = run_Hess(m)
    elif model == "Williams":
        m._populationparam = model[9:]
        m._callback = None
        [run_time, node_count, forest, directed_forest, obj_val, obj_bound] = run_williams(m)
        
    # Compile the results    
    result['State'] = state
    result['Model'] = model
    result['Primal Vertices'] = len(G.nodes)
    result['Edges'] = len(G.edges)
    result['Districts'] = m._k
    result['Run Time (Seconds)'] = '{0:.2f}'.format(run_time)
    result['Branch and Bound Nodes'] = '{0:.0f}'.format(node_count)
    result['Objective Value'] = '{0:.0f}'.format(obj_val) 
    result['Objective Bound'] = '{0:.0f}'.format(obj_bound) 
    append_dict_as_row(results_filename, result, fields)
    
    # Create the json file for optimal solutions
    if model == "Hess":
        warm_start = partitions_dict
        key = "partitions"
    elif model == "Williams":
        warm_start = m._forestedges
        key = "forest"
    
    # filename for outputs
    fn = "../warm_start/" + model + "_" + state
    
    # dump the solution info to json file
    json_fn = fn + ".json"
    with open(json_fn, 'w') as outfile:
        data = {}
        data[key] = warm_start
        json.dump(data, outfile)
    
    # Output districting figures if feasible solutions are found within the givn time limit
    if m.solCount > 0:
        assignment = [ -1 for u in G.nodes ]
        if model =="Williams":
            components = sorted(list(nx.connected_components(forest)))
            for node in sorted(list(forest.nodes)):
                for index in range(num_district):
                    if node in components[index]:
                        assignment[node] = index
        elif model == "Hess":
            color_index = 0
            for j in partitions_dict.keys():
                for node in partitions_dict[j]:
                    assignment[node] = color_index
                color_index += 1
        
        png_filename = path + "/" + state + "_" + model + "_" + ws_label + '.png'
        export_to_png(m, assignment, png_filename)
        
        # Output an additional arowed map when districted using William's model and on county level
        if model == "Williams":
            face_finder.draw_with_location(directed_forest, df, 'k', 100, 3,'r')
            arrow_graph_path =  path + "/" + state + "_" + model + "_" + "_" + ws_label + '_arrows.png'
            plt.savefig(arrow_graph_path)