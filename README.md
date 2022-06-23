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
* state: OK
* level: county
* base: hess
* fixing: true
* contiguity: scf
* symmetry: default
* extended: true
* order: B_decreasing
* heuristic: true
* lp: true

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
* level : {county, tract}
  * Either treat counties or census tracts as indivisible land units
* base : {hess, labeling} 
  * Hess model uses binary variables x_ij that equal one when vertex i is assigned to the district rooted at vertex j
  * Labeling model uses binary variables x_ij that equal one when vertex i is assigned to district number j, where j in {1, 2, ..., k }
* fixing : {True, False}
  * If true, will apply procedures to (safely) fix some variables to zero or one
* contiguity : {none, lcut, scf, shir}
  * none means that contiguity is not imposed
  * LCUT imposes contiguity with length-U a,b-separator inequalities (in branch-and-cut fashion)
  * SCF imposes contiguity with a single-commodity flow model. See [Hojny et al](https://link.springer.com/article/10.1007/s12532-020-00186-3)
  * SHIR imposes contiguity with a multi-commodity flow model. See [Shirabe2005](https://onlinelibrary.wiley.com/doi/full/10.1111/j.1538-4632.2005.00605.x) and [Shirabe2009](https://journals.sagepub.com/doi/abs/10.1068/b34104) and [Oehrlein and Haunert](http://www.josis.org/index.php/josis/article/viewArticle/379) and [Validi et al.](http://www.optimization-online.org/DB_HTML/2020/01/7582.html)
* symmetry : {default, aggressive, orbitope}
  * Default uses whatever the Gurobi MIP solver does by default
  * Aggressive is a Gurobi setting that seeks to exploit symmetry
  * Orbitope is the extended formulation for partitioning orbitopes due to [Faenza and Kaibel](https://pubsonline.informs.org/doi/abs/10.1287/moor.1090.0392)
* extended : {True, False}
  * If true, use an extended formluation to better capture the cut edges objective function. See [Ferreira et al](https://link.springer.com/article/10.1007/BF02592198)
* order : {none, decreasing, B_decreasing}
  * If none, the given vertex ordering will be used
  * If decreasing, the vertices will be sorted in terms of decreasing population
  * If B_decreasing, a vertex subset B in which all components of G[B] have population less than L will be placed at back, others placed at front by decreasing population
* heuristic : {True, False}
  * If true, will use a heuristic MIP warm start obtained from [GerryChain](https://gerrychain.readthedocs.io/en/latest/)
* lp : {True, False} 
  * If true, will create a (separate) model for the LP relaxation and solve it to evaluate LP strength

