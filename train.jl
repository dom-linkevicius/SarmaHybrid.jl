import JLD2
import Flux


import DifferentialEquations as DE
import DiffEqFlux as DEF
import Optimization
### import NLopt
### import OptimizationNLopt
### import Optimisers
### import OptimizationOptimisers
###import Optim
###import OptimizationOptimJL
###import LineSearches as LS
###### import Enzyme
### import SciMLSensitivity
### import Statistics as S
### import FFTW

import Dates
import PyPlot as plt
import IterTools


### using InteractiveUtils
### import Random
### Random.seed!(1)


include("codebase.jl")


dtype_func = Float64


STUFF_TRAIN  	= JLD2.load("Data/training_data.jld2")["data"]
### STUFF_TRAIN  	= JLD2.load("training_data_unnorm.jld2")["data"]


DATA_TRAIN   	= dtype_func.(STUFF_TRAIN["TRAINING_DATA"])
INPUTS_TRAIN_T 	= dtype_func.(STUFF_TRAIN["TRAINING_INPUTS_T"])
INPUTS_TRAIN_A 	= dtype_func.(STUFF_TRAIN["TRAINING_INPUTS_A"])
TSPAN_TRAIN  	= dtype_func.(STUFF_TRAIN["TSPAN"])
SAVE_T_TRAIN 	= dtype_func.(STUFF_TRAIN["SAVE_T"])


n_spc   = 9	## number of species
aug_dim = 1
h_dim = 15#0	## number of NN hidden units
o_dim = 2


net = Flux.Chain(Flux.Dense(n_spc + aug_dim, h_dim, Flux.celu),
 				 Flux.Dense(h_dim, h_dim, Flux.celu),
		 		 Flux.Dense(h_dim, o_dim + aug_dim))


net_p, net_re = Flux.destructure(net)

	
###lim = Inf
###num_0 = sum(net_p .== 0)
###net_p[net_p .== 0] .=  (rand(num_0) .* 2) .- 1 


net_p = dtype_func.(net_p)
println("NUMBER OF PARAMETERS IS " * string(size(net_p)[1]))


##########################
###### AUGMENT DATA ######
##########################


DATA_TRAIN     = cat(DATA_TRAIN, zeros(aug_dim, size(DATA_TRAIN)[2], size(DATA_TRAIN)[3]), dims=1)
###
###
###DATA_TEST     = DATA_TRAIN[:, :, end:end] 
###INPUTS_TEST_T = INPUTS_TRAIN_T[:, end:end]
###INPUTS_TEST_A = INPUTS_TRAIN_A[:, end:end] 
###
###use_dat=1#20 #100
###
###DATA_TRAIN     = DATA_TRAIN[:, :, 1:use_dat] 
###INPUTS_TRAIN_T = INPUTS_TRAIN_T[:, 1:use_dat]
###INPUTS_TRAIN_A = INPUTS_TRAIN_A[:, 1:use_dat] 
###
###### net_p = dtype_func.(JLD2.load("Model_results/icsb.jld2")["data"]["u"])
###

#################################### 
##### TEST THAT PREDICTION WORKS ###
#################################### 

###using BenchmarkTools
###### using InteractiveUtils
###### import Random
###### Random.seed!(1)
### 
######## loaded_p = JLD2.load("Model_results/local_test.jld2")["data"]["u"]
###norm = dtype_func.(4000)
### 
###
###### @time pred  = predict_node_gaussian(DATA_TRAIN[:,1,1], net_re, net_p, INPUTS_TRAIN_T[:,1:1], INPUTS_TRAIN_A[:,1:1], TSPAN_TRAIN, SAVE_T_TRAIN, norm)
###@btime pred = predict_node_gaussian(DATA_TRAIN[:,1,1], net_re, net_p, INPUTS_TRAIN_T[:,1:1], INPUTS_TRAIN_A[:,1:1], TSPAN_TRAIN, SAVE_T_TRAIN, norm)
###### pred = predict_node_gaussian(DATA_TRAIN[:,2,2], net_re, net_p, INPUTS_TRAIN_T[:,2:2], INPUTS_TRAIN_A[:,2:2], TSPAN_TRAIN, SAVE_T_TRAIN, norm)
### 
### plt.subplot(2, 1, 1)
### 
### plt.plot(SAVE_T_TRAIN, pred[2, : ], 	label="MODEL M3K")
### plt.plot(SAVE_T_TRAIN, DATA_TRAIN[2, :, 1], 	label="TRUE M3K")
### plt.plot(SAVE_T_TRAIN, pred[3, : ], 	label="MODEL M3K*")
### plt.plot(SAVE_T_TRAIN, DATA_TRAIN[3, :, 1], 	label="TRUE M3K*")
### plt.legend()
### 
### 
### plt.subplot(2, 1, 2)
### plt.plot(SAVE_T_TRAIN, pred[4, : ], 	label="MODEL MK")
### plt.plot(SAVE_T_TRAIN, DATA_TRAIN[7, :, 1], 	label="TRUE MK")
### plt.plot(SAVE_T_TRAIN, pred[5, : ], 	label="MODEL MK*")
### plt.plot(SAVE_T_TRAIN, DATA_TRAIN[8, :, 1], 	label="TRUE MK*")
### plt.plot(SAVE_T_TRAIN, pred[6, : ], 	label="MODEL MK**")
### plt.plot(SAVE_T_TRAIN, DATA_TRAIN[9, :, 1], 	label="TRUE MK**")
### 
### plt.legend()
### 
### plt.savefig("test_hybrid_plot.jpg")
###
#######################################
##### TEST THAT PREDICTION WORKS ###
####################################
 
 
##############################
##### TEST THAT LOSS WORKS ###
##############################
###### 
### using BenchmarkTools
##### using InteractiveUtils
##### import Random
##### Random.seed!(1)
##### 
##### 
##### 
##### ### @time pred = predict_node_gaussian(DATA_TRAIN[:,1,1], net_re, net_p, INPUTS_TRAIN[:,1:1], TSPAN_TRAIN, SAVE_T_TRAIN, norm)
##### 
##### 
### @btime loss_node(net_p, INPUTS_TRAIN_T, INPUTS_TRAIN_A, DATA_TRAIN, net_re, TSPAN_TRAIN, SAVE_T_TRAIN, dtype_func(4000))
###
###
###
###### STUFF_RAMP  	= JLD2.load("test_data_ramp.jld2")["data"]
###### 
###### DATA_RAMP   	= dtype_func.(STUFF_RAMP["TRAINING_DATA"])
###### TSPAN_RAMP  	= dtype_func.(STUFF_RAMP["TSPAN"])
###### SAVE_T_RAMP 	= dtype_func.(STUFF_RAMP["SAVE_T"])
###### 
###### 
##### println(loss_node_std_gaussian(net_p, INPUTS_TRAIN_T, INPUTS_TRAIN_A, DATA_TRAIN, net_re, TSPAN_TRAIN, SAVE_T_TRAIN, dtype_func(4000)))
###### 
###### println(loss_node_std_ramp(net_p, [5.; 10.; 50; 500.] ./ 4000, DATA_RAMP, net_re, TSPAN_RAMP, SAVE_T_RAMP, dtype_func(4000)))
###
###
######  
###### 
##############################
##### TEST THAT LOSS WORKS ###
##############################
###
###### println("SETING UP LBFGS TRAINING")
###### 
###### train_loss = []
###### norm = dtype_func.(4000)
###### L1_reg_LAM_LO = dtype_func.(1e-3)
###### 
###### loss(p) = loss_node(p, INPUTS_TRAIN_T, INPUTS_TRAIN_A, DATA_TRAIN, net_re, TSPAN_TRAIN, SAVE_T_TRAIN, norm, L1_reg_LAM_LO)
###### pred_func(p) = predict_node_gaussian(DATA_TRAIN[:, 1, end], net_re, p, INPUTS_TRAIN_T[:,end], INPUTS_TRAIN_A[:,end], TSPAN_TRAIN, SAVE_T_TRAIN, norm)
###### 
###### cb(param, l) = callback(param, l, pred_func, DATA_TRAIN[:,:,end], "Plots/S2_", train_loss, SAVE_T_TRAIN)
###### 
###### adtype = Optimization.AutoForwardDiff()
###### opt_func = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
###### 
###### down_lim = ones(size(net_p)[1]) .* -Inf
###### up_lim 	 = ones(size(net_p)[1]) .*  Inf
###### 
###### opt_prob = Optimization.OptimizationProblem(opt_func, net_p, lb=down_lim, ub=up_lim)
######  
###### res = Optimization.solve(opt_prob, Optim.LBFGS(m=50, linesearch=LS.HagerZhang()), callback=cb, allow_f_increases=true)
######  
###


 
println("SETING UP ADAM TRAINING")


train_loss = []
norm = dtype_func.(4000)
### norm = dtype_func.(1)
L2_reg_LAM_LO = dtype_func.(1e-6)


loss(p, X_T, X_A, Y) = loss_node(p, X_T, X_A, Y, net_re, TSPAN_TRAIN, SAVE_T_TRAIN, norm, L2_reg_LAM_LO)


pred_func(p) = predict_node_gaussian(DATA_TRAIN[:, 1, end], net_re, p, INPUTS_TRAIN_T[:,end], INPUTS_TRAIN_A[:,end], TSPAN_TRAIN, SAVE_T_TRAIN, norm)

cb(param, l) = callback(param, l, pred_func, DATA_TRAIN[:,:,end], "Plots/S2_", train_loss, SAVE_T_TRAIN)


adtype = Optimization.AutoForwardDiff()
###adtype = Optimization.AutoZygote()


opt_func = Optimization.OptimizationFunction((x, p, X_T, X_A, Y) -> loss(x, X_T, X_A, Y), adtype)
opt_prob = Optimization.OptimizationProblem(opt_func, net_p)


BATCH_SIZE 	= 2#5 
MAX_ITERS 	= 2000
L_RATE 		= 1e-3
L_DECAY		= 0.3
DEC_STEP	= 500
DEC_CLIP	= 1e-6


train_loader = Flux.Data.DataLoader((INPUTS_TRAIN_T, INPUTS_TRAIN_A, DATA_TRAIN); batchsize=BATCH_SIZE, shuffle=true)


opt = Flux.Optimiser(Flux.ExpDecay(L_RATE, L_DECAY, DEC_STEP, DEC_CLIP), Flux.ADAM())
res = Optimization.solve(opt_prob, opt, IterTools.ncycle(train_loader, MAX_ITERS), callback=cb)

###### @btime res = Optimization.solve(opt_prob, opt, IterTools.ncycle(train_loader, MAX_ITERS), callback=cb)
###
###
###JLD2.save("./Model_results/LOCAL_OPT_mapk_s2_" * string(Dates.now()) * ".jld2", "data", Dict("u"=>res.u, "loss"=>train_loss))
###