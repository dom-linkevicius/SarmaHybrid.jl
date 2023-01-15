# SarmaHybrid

## Overview
This repository aims to implement a scientific machine learning model in a scenario where only parts of the system (in terms of variables and variable interactions) are known. The test case chosen fr thsi approach is a MAPK cascade model by [Sarma and Ghosh (2012)](http://www.biomedcentral.com/1756-0500/5/287). The repository contains an implementation of their model (currently only S2, S1 to be implemented later) and a hybrid version of the same model where some intermediate species are taken out, see Figure 1 below, which is a slide from a presentation I had given. 

![Figure 1, true and hybrid model structures.](model_comparison.PNG?raw=true "blabla")

The hybrid model contains [augmented neural ODEs](https://dl.acm.org/doi/10.5555/3454287.3454569) which both link the disjoint parts of the model and provide additional dimensionality to represent the missing variables.

## Package installation

Package installation should be as simple as:
```import Pkg
Pkg.add("https://github.com/dom-linkevicius/SarmaHybrid.jl/tree/first_updates")
```
If there are any problems with this, please raise an issue.




## Brief overview of current package functionality

The package currently implements $v_1$ - $v_{10}$ for model S2 from [Sarma and Ghosh (2012)](http://www.biomedcentral.com/1756-0500/5/287), various functions that wrap them so that for use with `DifferentialEquations.jl` and `Flux.jl`, a data generation function `gen_gauss_input_data` which generates data with narrow gaussian pulses of $Signal$ molecule, a hybrid model function `hybrid_model_s2` and functions that take a parameter array `P` and produce predictions (`predict_node_gaussian`), $L_2$ loss over the true species (`loss_node`), as well as two convenience functions `do_plot` which plots comparisons between the true data and the hybrid model data, and `callback` which stores data and calls `do_plot`. 


## Example usage

```
### Importing packages
import SarmaHybrid
import Distributions
import Flux
import Optimization
import Optimisers
import OptimizationOptimisers
import IterTools


dt = 0.1  ### time-step for data saving

### data generation using the model S2
N_TRAIN 	= 10  ### Number of simulations to run
TSPAN_TRAIN 	= (0, 500)  ### Time span of the simulations
SAVE_T_TRAIN	= TSPAN_TRAIN[1]:dt:TSPAN_TRAIN[2] ### Array of times for which data will be saved
INIT_TRAIN 	= SarmaHybrid.init_concs ### Initial concentration, taken from Sarma, Ghosh (2012)
max_u0 = maximum(SarmaHybrid.init_concs()) ### Used for scaling the concentrations to a range of 0 to 1
INP_DISTS_TRAIN = [2:6, Distributions.Uniform(TSPAN_TRAIN[1], TSPAN_TRAIN[2]-TSPAN_TRAIN[2]/5.), Distributions.Uniform(5/max_u0, 500/max_u0)]  ### Array of objects used to sample the parameters for input functions: # of inputs, timing of inputs, amplitude of input 

TRAINING_DATA, TRAINING_INPUTS_T, TRAINING_INPUTS_A = SarmaHybrid.gen_gauss_input_data(N_TRAIN, TSPAN_TRAIN, SAVE_T_TRAIN, INIT_TRAIN, INP_DISTS_TRAIN) ### Data generation proper


n_spc   = 9	### number of real species
aug_dim = 1 ### augmented dimensionality in aNODEs
h_dim = 5	### number of NN hidden units
o_dim = 2   ### number of variables needed to link the real species


net = Flux.Chain(Flux.Dense(n_spc + aug_dim, h_dim, Flux.celu),  ### neural network definition
 				 Flux.Dense(h_dim, h_dim, Flux.celu),
		 		 Flux.Dense(h_dim, o_dim + aug_dim))


net_p, net_re = Flux.destructure(net)


TRAINING_DATA     = cat(TRAINING_DATA, zeros(aug_dim, size(TRAINING_DATA)[2], size(TRAINING_DATA)[3]), dims=1) ### augmenting data for aNODEs


train_loss = [] ### array to save training loss
L2_reg_LAM = 1e-6 ### L2 regularization size


loss(p, X_T, X_A, Y) = SarmaHybrid.loss_node(p, X_T, X_A, Y, net_re, TSPAN_TRAIN, SAVE_T_TRAIN, max_u0, L2_reg_LAM_LO) ### closure for the loss function


pred_func(p) = SarmaHybrid.predict_node_gaussian(TRAINING_DATA[:, 1, end], net_re, p, TRAINING_INPUTS_T[:,end], TRAINING_INPUTS_A[:,end], TSPAN_TRAIN, SAVE_T_TRAIN, max_u0) ### closure to generate predictions using the hybrid model S2

cb(param, l) = SarmaHybrid.callback(param, l, pred_func, TRAINING_DATA[:,:,end], "Plots/S2_", train_loss, SAVE_T_TRAIN, "show_save") ### callback for convenience, saves data in Plots/ by default, to avoid spamming directory with plot files

adtype = Optimization.AutoForwardDiff()  ### ad type used to calculate gradients

opt_func = Optimization.OptimizationFunction((x, p, X_T, X_A, Y) -> loss(x, X_T, X_A, Y), adtype)
opt_prob = Optimization.OptimizationProblem(opt_func, net_p)

### Training parameters
BATCH_SIZE 	= 2  
MAX_ITERS 	= 20

train_loader = Flux.Data.DataLoader((TRAINING_INPUTS_T, TRAINING_INPUTS_A, TRAINING_DATA); batchsize=BATCH_SIZE, shuffle=true) ### data loader for batching during training

res = Optimization.solve(opt_prob, Optimisers.Adam(), IterTools.ncycle(train_loader, MAX_ITERS), callback=cb) ### training proper ``` 