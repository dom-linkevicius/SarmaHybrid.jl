# SarmaHybrid

## General description
This repository aims to implement a scientific machine learning model in a scenario where only parts of the system (in terms of variables and variable interactions) are known. The test case chosen fr thsi approach is a MAPK cascade model by [Sarma and Ghosh (2012)](http://www.biomedcentral.com/1756-0500/5/287). The repository contains an implementation of their model (currently only S2, S1 to be implemented later) and a hybrid version of the same model where some intermediate species are taken out, see Figure 1 below, which is a slide from a presentation I had given. 

![Figure 1, true and hybrid model structures.](model_comparison.PNG?raw=true "blabla")

The hybrid model contains [augmented neural ODEs](https://dl.acm.org/doi/10.5555/3454287.3454569) which both link the disjoint parts of the model and provide additional dimensionality to represent the missing variables.


## Brief overview of current package functionality

The package currently implements $v_1$ - $v_{10}$ for model S2 from [Sarma and Ghosh (2012)](http://www.biomedcentral.com/1756-0500/5/287), various functions that wrap them so that for use with `DifferentialEquations.jl`, a data generation function `gen_gauss_input_data` which generates data with narrow gaussian pulses of $Signal$ molecule, a hybrid model function `hybrid_model_s2` and functions that take a parameter array `P` and produce predictions (`predict_node_gaussian`), $L_2$ loss over the true species (`loss_node`), as well as two convenience functions `do_plot` which plots comparisons between the true data and the hybrid model data, and `callback` which stores data and calls `do_plot`. 