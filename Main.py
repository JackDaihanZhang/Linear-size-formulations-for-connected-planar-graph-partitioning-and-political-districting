###########################
# Imports
###########################
import csv
import networkx as nx
import Hess
import Williams_test

################################################
# Summarize computational results to csv file
################################################
def write_to_csv(state_rows, state_results, filename, fields):
    rows = []
    # Create an index to keep track of the
    result_index = 0
    for state in state_rows:
        [run_time, node_count, _, val, bound] = state_results[result_index]
        result_index += 1
        row = states_rows[state]
        row.insert(0, state)
        row.append(run_time)
        row.append(node_count)
        row.append(val)
        row.append(bound)
        rows.append(row)

    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        # writing the data rows
        csvwriter.writerows(rows)

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

###########################
# Hard-coded inputs
###########################

#states_rows = {"AL": [67, 106, 171, 7], "AR": [75, 119, 192, 4], "IA": [99, 125, 222, 4], "KS": [105, 160, 263, 4],
#                "ME": [16, 20, 34, 2], "MS": [82, 122, 202, 4], "NE": [93, 140, 231, 3], "NM": [33, 47, 78, 3],
#               "WV": [55, 72, 125, 2], "ID":[44, 60, 102, 2]}
states_rows = {"ME": [16, 20, 34, 2]}
fields = ["State", "Primal Vertices", "Dual Vertices", "Edges", "Districts", "Run Time (Seconds)", "Branch and Bound Nodes", "Objective Value", "Objective Bound"]


###########################
# Run An Instance using Williams' model
###########################
def run_williams(state):
    num_district = states_rows[state][3]
    return Williams_test.Williams_model(num_district, state)

###########################
# Run An Instance using Hess' model
###########################
def run_Hess(state):
    num_district = states_rows[state][3]
    return Hess.Hess_model(state,num_district)


###########################
# Run the complete experiment
###########################
state_results = []
# Specify the model
model = "Hess"
for state in states_rows:
    if model == "Hess":
        result = run_Hess(state)
    else:
        result = run_williams(state)
    state_results.append(result)
    write_to_txt(result[2],states_rows[state][3], "Experiment_" + model + "/" + state + "_solution.txt", model)
write_to_csv(states_rows, state_results, "Experiment_" + model + "/" + "result.csv", fields)