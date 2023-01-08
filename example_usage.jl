import SarmaHybrid
import Distributions
import Flux
import Optimization
import Optimisers
import OptimizationOptimisers
import IterTools

dt = 0.1

N_TRAIN 	= 10#00
TSPAN_TRAIN 	= (0, 500) 
SAVE_T_TRAIN	= TSPAN_TRAIN[1]:dt:TSPAN_TRAIN[2]
INIT_TRAIN 	= SarmaHybrid.init_concs
max_u0 = maximum(SarmaHybrid.init_concs())
INP_DISTS_TRAIN = [2:6, Distributions.Uniform(TSPAN_TRAIN[1], TSPAN_TRAIN[2]-TSPAN_TRAIN[2]/5.), Distributions.Uniform(5/max_u0, 500/max_u0)]

TRAINING_DATA, TRAINING_INPUTS_T, TRAINING_INPUTS_A = SarmaHybrid.gen_gauss_input_data(N_TRAIN, TSPAN_TRAIN, SAVE_T_TRAIN, INIT_TRAIN, INP_DISTS_TRAIN)


n_spc   = 9	## number of species
aug_dim = 1
h_dim = 5	## number of NN hidden units
o_dim = 2


net = Flux.Chain(Flux.Dense(n_spc + aug_dim, h_dim, Flux.celu),
 				 Flux.Dense(h_dim, h_dim, Flux.celu),
		 		 Flux.Dense(h_dim, o_dim + aug_dim))


net_p, net_re = Flux.destructure(net)


TRAINING_DATA     = cat(TRAINING_DATA, zeros(aug_dim, size(TRAINING_DATA)[2], size(TRAINING_DATA)[3]), dims=1)


train_loss = []
L2_reg_LAM_LO = 1e-6


loss(p, X_T, X_A, Y) = SarmaHybrid.loss_node(p, X_T, X_A, Y, net_re, TSPAN_TRAIN, SAVE_T_TRAIN, max_u0, L2_reg_LAM_LO)


pred_func(p) = SarmaHybrid.predict_node_gaussian(TRAINING_DATA[:, 1, end], net_re, p, TRAINING_INPUTS_T[:,end], TRAINING_INPUTS_A[:,end], TSPAN_TRAIN, SAVE_T_TRAIN, max_u0)

cb(param, l) = SarmaHybrid.callback(param, l, pred_func, TRAINING_DATA[:,:,end], "Plots/S2_", train_loss, SAVE_T_TRAIN, "show_save")

adtype = Optimization.AutoForwardDiff()

opt_func = Optimization.OptimizationFunction((x, p, X_T, X_A, Y) -> loss(x, X_T, X_A, Y), adtype)
opt_prob = Optimization.OptimizationProblem(opt_func, net_p)


BATCH_SIZE 	= 2#5 
MAX_ITERS 	= 20
L_RATE 		= 1e-3
L_DECAY		= 0.3
DEC_STEP	= 500
DEC_CLIP	= 1e-6


train_loader = Flux.Data.DataLoader((TRAINING_INPUTS_T, TRAINING_INPUTS_A, TRAINING_DATA); batchsize=BATCH_SIZE, shuffle=true)

res = Optimization.solve(opt_prob, Optimisers.Adam(), IterTools.ncycle(train_loader, MAX_ITERS), callback=cb)