###########################
# Imports
###########################
import gurobipy as gp
from gurobipy import GRB
import csv
from csv import DictWriter
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

###########################
# Run An Instance using Williams' model
###########################
def run_williams(df, primal_dual_pairs, m):
    return Williams_test.Williams_model(df, primal_dual_pairs, m)

###########################
# Run An Instance using Hess' model
###########################
def run_Hess(state,G,m):
    return Hess.Hess_model(state,G,m)

################################################
# Draws districts and saves to png file
################################################ 

def export_to_png(G, df, assignment, filename,level):
    new_assignment = [ -1 for u in G.nodes ]
    if level == "County":    
        for vertex in G.nodes():
            geoID = G.nodes[vertex]["GEOCODE"]
            for u in G.nodes():
                if geoID == df['GEOID20'][u]:
                    new_assignment[u] = assignment[vertex]
    if level == "County": 
        df['assignment'] = new_assignment
    elif level == "Tract": 
        df['assignment'] = assignment  
    my_fig = df.plot(column='assignment').get_figure()
    RESIZE_FACTOR = 3
    my_fig.set_size_inches(my_fig.get_size_inches()*RESIZE_FACTOR)
    plt.axis('off')
    my_fig.savefig(filename)

# prepare csv file by writing column headers
with open(results_filename,'w',newline='') as csvfile:   
    writer = csv.DictWriter(csvfile, fieldnames = fields)
    writer.writeheader()
    
############################################################
# Run experiments for each config in batch_config file
############################################################
# Credit to https://github.com/hamidrezavalidi/Political-Districting-to-Minimize-Cut-Edges/blob/master/src/main.py
for key in batch_configs.keys():
    config = batch_configs[key]
    model = config['model']
    state = config['state']
    num_district = config['num_district']
    level = config['level']
    RCI = config['RCI']
    max_clique = config["max clique"]
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
    m = gp.Model()
    # Set a time limit
    m.setParam('TimeLimit', 3600)
    # MIP focus
    #m.setParam('MIPFocus', 2)
    # Set gap to zero
    m.setParam('MIPGap', 0)
    m.modelSense = GRB.MINIMIZE
    total_pop = sum(p)
    m._total_pop = total_pop 
    L = math.ceil((total_pop/num_district)*(0.995))
    U = math.floor((total_pop/num_district)*(1.005))
    print("Solving " + state + " with L = " + str(L) + " and U = " + str(U) + " at " + level + " level under " + model + " model")
    m._state = state
    m._level = level
    m._model = model
    m._U = U
    m._L = L
    m._p = p
    m._G = G
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
    m._heuristic = config['heuristic']
    result = {}
    if m._heuristic:
        heuristic_file = open('../heuristic-results/10000-iterations/heur_'+state+"_"+level+".json", 'r')
        heuristic_dict = json.load(heuristic_file)     
        heuristic_districts = [ [node['index'] for node in heuristic_dict['nodes'] if node['district']==j ] for j in range(m._k) ]
        m._hdistricts = heuristic_districts
        heuristic_cut_edges = heuristic_dict['cut edges']
        m._cuts = heuristic_cut_edges
        result['heur_obj'] = heuristic_dict['obj']
        result['heur_time'] = heuristic_dict['time']
        result['heur_iter'] = heuristic_dict['iterations']
        heuristic_label ="w_heuristic"
    else:
        heuristic_districts = None
        result['heur_obj'] = 'n/a'
        result['heur_time'] = 'n/a'
        result['heur_iter'] = 'n/a'
        heuristic_label = "wo_heuristic"
    if model == "Hess":
        [run_time, node_count, forest, obj_val, obj_bound] = run_Hess(state,G,m)
    elif model == "Williams_cuts" or model == "Williams_flow":
        m._populationparam = model[9:]
        m._callback = None
        [run_time, node_count, forest, directed_forest, obj_val, obj_bound] = run_williams(df,primal_dual_pairs,m)
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
    append_dict_as_row(results_filename,result,fields)
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
        else:
            assignment = [0]*len(G.nodes)
            for i in range(1,num_district+1):
                for node in forest[i-1]:
                    assignment[node] = i
        png_filename = path + "/" + state + "_" + model + "_" + level + "_" + heuristic_label + '.png'
        export_to_png(G, df, assignment, png_filename, level)
        if level == "County" and ( model == "Williams_cuts" or model == "Williams_flow"):
            face_finder.draw_with_location(directed_forest,df,'k',100,3,'r')
            arrow_graph_path =  path + "/" + state + "_" + model + "_" + level + "_" + heuristic_label + '_arrows.png'
            plt.savefig(arrow_graph_path)