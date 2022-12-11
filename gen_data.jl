import Dates
import Distributions
import JLD2

import PyPlot as plt
import DifferentialEquations as DE

include("codebase.jl")

dt = 0.1

N_TRAIN 	= 2#00
TSPAN_TRAIN 	= (0, 500) 
SAVE_T_TRAIN	= TSPAN_TRAIN[1]:dt:TSPAN_TRAIN[2]
INIT_TRAIN 	= init_concs
max_u0 = maximum(init_concs())
INP_DISTS_TRAIN = [2:6, Distributions.Uniform(TSPAN_TRAIN[1], TSPAN_TRAIN[2]-TSPAN_TRAIN[2]/5.), Distributions.Uniform(5/max_u0, 500/max_u0)]

TRAINING_DATA, TRAINING_INPUTS_T, TRAINING_INPUTS_A = gen_gauss_input_data(N_TRAIN, TSPAN_TRAIN, SAVE_T_TRAIN, INIT_TRAIN, INP_DISTS_TRAIN)

plt.clf()
plt.subplot(2, 2, 1)
plt.title("Signal")
plt.plot(SAVE_T_TRAIN, TRAINING_DATA[1,:,1])

plt.subplot(2, 2, 2)
plt.title("M3K*")
plt.plot(SAVE_T_TRAIN, TRAINING_DATA[3,:,1])


plt.subplot(2, 2, 3)
plt.title("M2K**")
plt.plot(SAVE_T_TRAIN, TRAINING_DATA[6,:,1])


plt.subplot(2, 2, 4)
plt.title("MK**")
plt.plot(SAVE_T_TRAIN, TRAINING_DATA[9,:,1])
plt.savefig("sample_plot.png", dpi=500)


JLD2.save("Data/training_data.jld2", "data", Dict("TRAINING_DATA"=>TRAINING_DATA, "TRAINING_INPUTS_A"=>TRAINING_INPUTS_A, "TRAINING_INPUTS_T"=>TRAINING_INPUTS_T, "TSPAN"=>TSPAN_TRAIN, "SAVE_T"=>SAVE_T_TRAIN, "INIT"=>INIT_TRAIN))