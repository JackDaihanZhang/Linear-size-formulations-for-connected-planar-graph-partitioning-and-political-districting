###########################
# Run options
###########################  

levels = { 'county' }
iteration_options = { 100 }

###########################
# Imports
########################### 
import networkx as nx 

import time
import json
import os

from gerrychain import (GeographicPartition, Graph, MarkovChain, updaters, constraints, accept)
from gerrychain.tree import recursive_tree_part
from gerrychain.proposals import recom
from functools import partial

import geopandas as gpd
import matplotlib.pyplot as plt

import read

###########################
# Hard-coded inputs
###########################  

state_codes = {
    'WA': '53', 'DE': '10', 'WI': '55', 'WV': '54', 'HI': '15',
    'FL': '12', 'WY': '56', 'NJ': '34', 'NM': '35', 'TX': '48',
    'LA': '22', 'NC': '37', 'ND': '38', 'NE': '31', 'TN': '47', 'NY': '36',
    'PA': '42', 'AK': '02', 'NV': '32', 'NH': '33', 'VA': '51', 'CO': '08',
    'CA': '06', 'AL': '01', 'AR': '05', 'VT': '50', 'IL': '17', 'GA': '13',
    'IN': '18', 'IA': '19', 'MA': '25', 'AZ': '04', 'ID': '16', 'CT': '09',
    'ME': '23', 'MD': '24', 'OK': '40', 'OH': '39', 'UT': '49', 'MO': '29',
    'MN': '27', 'MI': '26', 'RI': '44', 'KS': '20', 'MT': '30', 'MS': '28',
    'SC': '45', 'KY': '21', 'OR': '41', 'SD': '46'
}

congressional_districts = {
    'WA': 10, 'DE': 1, 'WI': 8, 'WV': 2, 'HI': 2,
    'FL': 28, 'WY': 1, 'NJ': 12, 'NM': 3, 'TX': 38,
    'LA': 6, 'NC': 14, 'ND': 1, 'NE': 3, 'TN': 9, 'NY': 26,
    'PA': 17, 'AK': 1, 'NV': 4, 'NH': 2, 'VA': 11, 'CO': 8,
    'CA': 52, 'AL': 7, 'AR': 4, 'VT': 1, 'IL': 17, 'GA': 14,
    'IN': 9, 'IA': 4, 'MA': 9, 'AZ': 9, 'ID': 2, 'CT': 5,
    'ME': 2, 'MD': 8, 'OK': 5, 'OH': 15, 'UT': 4, 'MO': 8,
    'MN': 8, 'MI': 13, 'RI': 2, 'KS': 4, 'MT': 2, 'MS': 4,
    'SC': 7, 'KY': 6, 'OR': 6, 'SD': 1
}

skips = {
    ('WA','tract'), ('WA','county'), ('DE','tract'), ('DE','county'), ('WI','tract'), 
    ('WI','county'), ('HI','tract'), ('HI','county'), ('FL','tract'), ('FL','county'), 
    ('WY','tract'), ('WY','county'), ('NJ','tract'), ('NJ','county'), ('TX','tract'), 
    ('TX','county'), ('LA','tract'), ('LA','county'), ('NC','tract'), ('NC','county'), 
    ('ND','tract'), ('ND','county'), ('TN','tract'), ('TN','county'), ('NY','tract'), 
    ('NY','county'), ('PA','tract'), ('PA','county'), ('AK','tract'), ('AK','county'), 
    ('NV','county'), ('NH','county'), ('VA','tract'), ('VA','county'), ('CO','tract'), 
    ('CO','county'), ('CA','tract'), ('CA','county'),  ('VT','tract'), 
    ('VT','county'), ('IL','tract'), ('IL','county'), ('GA','tract'), ('GA','county'),
    ('IN','tract'), ('IN','county'), ('IA','tract'), ('MA','tract'), ('MA','county'), 
    ('AZ','tract'), ('AZ','county'), ('CT','tract'), ('CT','county'), ('MD','tract'),
    ('MD','county'), ('OK','tract'),('OK','county'), ('OH','tract'), ('OH','county'), 
    ('UT','county'), ('MO','tract'), ('MO','county'), ('MN','tract'), ('MN','county'), 
    ('MI','tract'), ('MI','county'), ('RI','tract'), ('RI','county'), ('KS','tract'), 
    ('MT','tract'), ('MT','county'), ('SC','tract'), ('SC','county'), ('KY','tract'), 
    ('KY','county'), ('OR','tract'), ('OR','county'), ('SD','tract'), ('SD','county')
}

################################################
# Draws districts and saves to png file
################################################ 

def export_to_png(G, df, districts, filename):
    
    assignment = [ -1 for u in G.nodes ]
    
    for j in range(len(districts)):
        for i in districts[j]:
            geoID = G.nodes[i]["GEOID20"]
            for u in G.nodes:
                if geoID == df['GEOID20'][u]:
                    assignment[u] = j
    
    if min(assignment[v] for v in G.nodes) < 0:
        print("Error: did not assign all nodes in district map png.")
    else:
        df['assignment'] = assignment
        my_fig = df.plot(column='assignment').get_figure()
        RESIZE_FACTOR = 3
        my_fig.set_size_inches(my_fig.get_size_inches()*RESIZE_FACTOR)
        plt.axis('off')
        my_fig.savefig(filename)
        

####################################
# Function for GerryChain call
####################################                       

def run_GerryChain_heuristic(G,population_deviation,k,iterations,p):
    
    my_updaters = {"population": updaters.Tally('P0010001', alias="population")}
    start = recursive_tree_part(G,range(k),sum(G.nodes[i]['P0010001'] for i in G.nodes())/k,'P0010001', population_deviation/2,1)
    initial_partition = GeographicPartition(G, start, updaters = my_updaters)
    
    proposal = partial(recom,
                       pop_col="P0010001",
                       pop_target=sum(G.nodes[i]["P0010001"] for i in G.nodes())/k,
                       epsilon=population_deviation/2,
                       node_repeats= 2
                      )
    
    compactness_bound = constraints.UpperBound(
        lambda p: len(p["cut_edges"]),
        1.5*len(initial_partition["cut_edges"])
    )
    
    pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, population_deviation/2)
    
    my_chain = MarkovChain(
        proposal=proposal,
        constraints=[
            pop_constraint,
            compactness_bound
        ],
        accept=accept.always_accept,
        initial_state=initial_partition,
        total_steps=iterations
    )
    
    min_map_score = nx.diameter(G)*max(p)*len(G.nodes)
    print("In GerryChain heuristic, current min score is: ",end='')
    print(min_map_score,",",sep='',end=' ')
    for partition in my_chain:
        #current_cut_edges = sum(G[i][j]['edge_length'] for i,j in partition["cut_edges"])
        districts = [[i for i in G.nodes if partition.assignment[i]==j] for j in range(k)]
        map_score_sum = 0
        for district in districts:
            H = G.subgraph(district)
            #scores = { 0 : vertex for vertex in H.nodes }
            #scores = [0 for vertex in H.nodes]
            min_score = nx.diameter(H) * max(p) * len(H.nodes)
            #min_root = -1
            #min_path = []
            for vertex in H.nodes:
                length, path = nx.single_source_dijkstra(H, vertex)
                score = sum(length[node]*p[node] for node in H.nodes)
                if score < min_score:
                    min_score = score
                    #min_root = vertex
                    #min_path = path
                    
            map_score_sum += min_score        
                    
            #district_subgraph = G.subgraph(district)
            #district_radius = nx.radius(district_subgraph)
            #districts_radius_sum += district_radius

        
        print(map_score_sum,",",sep='',end=' ')
        
        if map_score_sum < min_map_score:
            min_map_score = map_score_sum
            best_partition = partition
            best_cut_edges = [(i,j) for (i,j) in partition["cut_edges"]]
        
    
    print("Best heuristic solution has the minimum map score of =", min_map_score)
    return ([[i for i in G.nodes if best_partition.assignment[i]==j] for j in range(k)], best_cut_edges, min_map_score)


###########################
# Main part of the code
###########################  

# create directories for results
os.mkdir("../heuristic-results")
for iterations in iteration_options:
    os.mkdir("../heuristic-results/"+str(iterations)+"-iterations") 

# run all settings
for state in state_codes.keys():
    #if state not in ["NH", "ID", "ME"]: continue 
    #if state == "AL" or state == "NM" or state == "NE" or state == "WV" or state == "KS": continue 
    if state == "AL": continue
    # parameters            
    k = congressional_districts[state]
    deviation = 0.01
    code = state_codes[state]
    
    for level in levels:
        
        # skip certain (state,level) pairs that we know:
        #   1. are infeasible, 
        #   2. are outside our scope (because of size), or
        #   3. gerrychain gets stuck on (infinite loop).
        
        if (state,level) in skips:
            continue
        else:
            print("Running GerryChain for " + state + " at " + level + " level.")
        
        # read input graph and shapefile df
        if level == "county":
            G = Graph.from_json("C:/Users/hamid/Downloads/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/data/county/dual_graphs/" + state + "_counties.json")
            p = [G.nodes[i]['P0010001'] for i in G.nodes()]
            df = gpd.read_file("C:/Users/hamid/Downloads/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/data/county/shape_files/"+state+"_counties.shp")
        elif level == "tract":
            G = Graph.from_json("C:/Users/hamid/Downloads/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/data/tract/dual_graphs/" + state + "_tracts.json")
            p = [G.nodes[i]['P0010001'] for i in G.nodes()]
            df = gpd.read_file("C:/Users/hamid/Downloads/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/data/tract/shape_files/"+state+"_tracts.shp")
            '''
            primal_path = "C:/Users/hamid/Downloads/dualization/"+state+"_primalGraph.txt"
            dual_path = "C:/Users/hamid/Downloads/dualization/"+state+"_dualGraph.txt"
            population_path = "C:/Users/hamid/Downloads/dualization/"+state+"_population.population"
            [G,_,p] = read.read_county_txt(primal_path,dual_path,population_path)
            df = gpd.read_file("C:/Users/hamid/Downloads/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/A-Compact-and-Integral-Model-for-Partitioning-Planar-Graphs-main/data/tract/shape_files/"+state+"_tracts.shp")
            '''
         
        # give each edge a "length" of one
        for i,j in G.edges:
            G[i][j]['edge_length'] = 1
        
        for iterations in iteration_options:
        
            # run GerryChain 
            start = time.time()
            (districts, cut_edges, heur_obj) = run_GerryChain_heuristic(G,deviation,k,iterations,p)
            stop = time.time()
            
            # filename for outputs
            fn = "../heuristic-results/"+str(iterations)+"-iterations/heur_"+state+"_"+level
            
            # draw the solution on a map
            png_fn = fn + ".png"
            export_to_png(G, df, districts, png_fn)
            
            # dump the solution info to json file
            json_fn = fn + ".json"
            with open(json_fn, 'w') as outfile:
                data = {}
                data['obj'] = heur_obj
                data['time'] = '{0:.2f}'.format(stop-start)
                data['iterations'] = iterations
                data['cut edges'] = cut_edges
                data['nodes'] = list()
        
                for j in range(k):
                    for i in districts[j]:
                        data['nodes'].append({
                                'name': G.nodes[i]["NAME20"],
                                'index': i,
                                'district': j
                                })
                json.dump(data, outfile)