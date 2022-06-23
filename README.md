# A linear-size and integral model for partitioning planar graphs with application in political redistricting

Code and data for the "A compact and integral model for partitioning planar graphs with application in political redistricting" by Jack Zhang, Hamidreza Validi, Austin Buchanan, and Illya V. Hicks.

Motivated by applications in political redistricting, we propose a linear size and compact model for partitioning a planar graph into k partitions. 


  
![Figure 1](readme_images/trees.png?raw=true "Input graph")


For New Mexico at the county level, districts obtained by the Williams-based model and the Hess-based model are shown below. 

![Figure 1](readme_images/NM_Will_Hess.jpg?raw=true "NM")

## Require
To run the code, you will need installations of [Gurobi](https://www.gurobi.com/) and [GerryChain](https://gerrychain.readthedocs.io/en/latest/).

The input data is provided by [Daryl DeFord](https://www.math.wsu.edu/faculty/ddeford/).

## Run
You can run the code from command line, like this:

```
C:\A-Linear-Size-and-Integral-Model-for-Partitioning-Planar-Graphs-main\src>python Main.py config.json 1>>log-file.txt 2>>error-file.txt
```

## config.json
The config file can specify a batch of runs. A particular run might look like this:
* state: AL
* model: Williams_flow
* num_district: 7
* level: county
* heuristic: false
* heuristic_iter: 100
* RCI: false
* max clique: false

The config.json file might look like this:
```
{
    "run1": {"state": "AL", "model": "Williams_flow", "num_district": 7, "level": "county", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run2": {"state": "AR", "model": "Williams_flow", "num_district": 4, "level": "county", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run3": {"state": "IA", "model": "Williams_flow", "num_district": 4, "level": "county", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run4": {"state": "KS", "model": "Williams_flow", "num_district": 4, "level": "county", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run5": {"state": "ME", "model": "Williams_flow", "num_district": 2, "level": "county", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run6": {"state": "MS", "model": "Williams_flow", "num_district": 4, "level": "county", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run7": {"state": "NE", "model": "Williams_flow", "num_district": 3, "level": "county", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run8": {"state": "NM", "model": "Williams_flow", "num_district": 3, "level": "county", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run9": {"state": "WV", "model": "Williams_flow", "num_district": 2, "level": "county", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run10": {"state": "ID", "model": "Williams_flow", "num_district": 2, "level": "county", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run11": {"state": "ME", "model": "Williams_flow", "num_district": 2, "level": "tract", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run12": {"state": "ID", "model": "Williams_flow", "num_district": 2, "level": "tract", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run13": {"state": "NH", "model": "Williams_flow", "num_district": 2, "level": "tract", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run14": {"state": "AL", "model": "Hess", "num_district": 7, "level": "county", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run15": {"state": "AR", "model": "Hess", "num_district": 4, "level": "county", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run16": {"state": "IA", "model": "Hess", "num_district": 4, "level": "county", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run17": {"state": "KS", "model": "Hess", "num_district": 4, "level": "county", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run18": {"state": "ME", "model": "Hess", "num_district": 2, "level": "county", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run19": {"state": "MS", "model": "Hess", "num_district": 4, "level": "county", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run20": {"state": "NE", "model": "Hess", "num_district": 3, "level": "county", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run21": {"state": "NM", "model": "Hess", "num_district": 3, "level": "county", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run22": {"state": "WV", "model": "Hess", "num_district": 2, "level": "county", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run23": {"state": "ID", "model": "Hess", "num_district": 2, "level": "county", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run24": {"state": "ME", "model": "Hess", "num_district": 2, "level": "tract", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run25": {"state": "ID", "model": "Hess", "num_district": 2, "level": "tract", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false},
    "run26": {"state": "NH", "model": "Hess", "num_district": 2, "level": "tract", "heuristic": false, "heuristic_iter": 100, "RCI": false, "max clique": false}
}
```

## Config options
Generally, each run should pick from the following options:
* state : {AL, AK, AZ, AR, CA, ... } 
  * [See list of 2-letter codes](https://en.wikipedia.org/wiki/List_of_U.S._state_and_territory_abbreviations)
* model : {Hess, Williams_flow, Williams_tract} 
  * Hess model uses binary variables x_ij that equal one when vertex i is assigned to the district rooted at vertex j
  * Williams_flow
* num_district:
* level : {county, tract}
  * Either treat counties or census tracts as indivisible land units
* heuristic : {true, false}
  * If true, will use a heuristic MIP warm start obtained from [GerryChain](https://gerrychain.readthedocs.io/en/latest/)
* heuristic_iter :
* RCI : {true, false}
* max clique : {true, false}
