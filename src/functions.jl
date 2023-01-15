function v1_f(sig, m3k, mk_ss, L)
	return (sig * L * m3k * L / 15) * (1 + (100 * mk_ss * L / 500)) / ((1 + m3k * L / 15) * (1 + mk_ss * L / 500))
end

function v2_f(p1, m3k_s, L)
	return 0.1 * p1 * L * (m3k_s * L / 100) / (1 + m3k_s * L / 100)
end

function v3n_f(m3k_s, m2k, m2k_s, mk_ss, L)
	return (0.1 * m3k_s * L * m2k * L / 20) / ((1 + m2k_s * L / 20 + m2k * L / 20) * (1 + mk_ss * L / 9))
end

function v4n_f(m3k_s, m2k, m2k_s, mk_ss, L)
	return (0.1 * m3k_s * L * m2k_s * L / 20) / ((1 + m2k_s * L / 20 + m2k * L / 20) * (1 + mk_ss * L / 9))
end

function v5_f(p2, m2k_s, m2k_ss, L)
	return (0.02 * p2 * m2k_ss * L * L) / 20 / (1 + m2k_ss * L / 20 + m2k_s * L / 20)
end

function v6_f(p2, m2k_s, m2k_ss, L)
	return (0.02 * p2 * m2k_s * L * L) / 20 / (1 + m2k_s * L / 20 + m2k_ss * L / 20)
end

function v7_f(m2k_ss, mk, mk_s, L)
	return (0.1 * m2k_ss * mk * L * L) / 20 / (1 + mk * L / 20 + mk_s * L / 20)
end

function v8_f(m2k_ss, mk_s, mk, L)
	return (0.1 * m2k_ss * L * mk_s * L) / 20 / (1 + mk_s * L / 20 + mk * L / 20)
end

function v9_f(p3, mk_ss, mk_s, L)
	return (0.02 * p3 * mk_ss * L * L) / 20 / (1 + mk_ss * L / 20 + mk_s * L / 20)
end

function v10_f(p3, mk_s, mk_ss, L)
	return (0.02 * p3 * mk_s * L * L) / 20 / (1 + mk_ss * L / 20 + mk_s * L / 20)
end



function gaussian(x, mu, sigma)
	val = 1.0 / (sigma * (2*pi)^(0.5)) * exp(-0.5 * (x-mu)^2 / sigma^2)
end


function gauss_input!(du, t, t_par, a_par, n_stim)
	du[1] += sum([a_par[i] * gaussian(t, t_par[i], 0.001) for i in 1:n_stim])
end


function trapz_int(x, y)

	neighbour_sums = (@view(y[1:end-1]) .+ @view(y[2:end])) ./ 2
	d_xk = @view(x[2:end]) .- @view(x[1:end-1])
	integral = sum(neighbour_sums .* d_xk)

end



"""
model_s2(u, L)
MAPK model S2 from Sarma and Ghost (2012). 

Parameters:

u - arary of current state of the system
L - float scaling factor
u[1]  - Sig
u[2]  - M3K
u[3]  - M3K*
u[4]  - M2K
u[5]  - M2K*
u[6]  - M2K**
u[7]  - MK
u[8]  - MK* 
u[9]  - MK**
u[10] - P1
u[11] - P2
u[12] - P3
"""
function model_s2(u, L)
	SIG    = u[1]
	M3K    = u[2]
	M3K_S  = u[3]
	M2K    = u[4]
	M2K_S  = u[5]
	M2K_SS = u[6]
	MK     = u[7]
	MK_S   = u[8]
	MK_SS  = u[9]
	P1     = u[10]
	P2     = u[11]
	P3     = u[12]

	v1_v  = v1_f(SIG, M3K, MK_SS, L)
	v2_v  = v2_f(P1, M3K_S, L)
	v3_v  = v3n_f(M3K_S, M2K, M2K_S, MK_SS, L)
	v4_v  = v4n_f(M3K_S, M2K, M2K_S, MK_SS, L)
	v5_v  = v5_f(P2, M2K_S, M2K_SS, L)
	v6_v  = v6_f(P2, M2K_S, M2K_SS, L)
	v7_v  = v7_f(M2K_SS, MK, MK_S, L)
	v8_v  = v8_f(M2K_SS, MK_S, MK, L)
	v9_v  = v9_f(P3, MK_SS, MK_S, L)
	v10_v = v10_f(P3, MK_S, MK_SS, L)

	out = similar(u)

	out[1] = -u[1]
	out[2] = 1/L * (v2_v - v1_v)
	out[3] = 1/L * (v1_v - v2_v)
	out[4] = 1/L * (v6_v - v3_v)
	out[5] = 1/L * (v3_v + v5_v - v4_v - v6_v)
	out[6] = 1/L * (v4_v - v5_v)
	out[7] = 1/L * (v10_v - v7_v)
	out[8] = 1/L * (v7_v - v8_v + v9_v - v10_v)
	out[9] = 1/L * (v8_v - v9_v)
	out[10]= 0
	out[11]= 0
	out[12]= 0

	return out

end



"""
prob_s2(du, u, p, t, I, L)
model_s2 wrapper function to make it consistent with DifferentialEquations.jl

Parameters:

du, u, p, t - usual meaning in the context of DifferentialEquations.j
I - input function that mutates u in-place
L - float scaling of concentrations
"""
function prob_s2(du, u, p, t, I, L)

	du[:] .= model_s2(u, L)
	I(t)

	return du
end



function init_concs()
	vals = [0; 1000; 0; 4000; 0; 0; 1000; 0; 0; 100; 500; 500]
end



"""
gen_gauss_input_data(N, tspan, save_t, init_func, inp_dists; setmax=true)
Function to generated data using the true model S2.

Parameters:

N - Int of simulations to run
tspan - (t_start, t_end) tuple passed to DifferentialEquations.ODEProblem
save_t - array of times at which the data is saved, passed to DifferentialEquations.solve
init_func - function used to provide initial concentration for the simulations
inp_dists - array with three objects (such that one could call rand(...) on them) used to sample parameters for inputs
setmax - boolean to check whether to scale concentrations by maximal initial value
"""
function gen_gauss_input_data(N, tspan, save_t, init_func, inp_dists; setmax=true)

	all_data = []
	all_inp_t  = []
	all_inp_a  = []

	for i in 1:N

		println("Run " * string(i))

		u0 = init_func()
		if setmax
			max_u0 = maximum(u0)
		else
			max_u0 = 1
		end
		u0 = u0 ./ max_u0
		n_inp = rand(inp_dists[1], 1)[1]
		t_inp = [rand(inp_dists[2], n_inp); zeros(maximum(inp_dists[1]) - n_inp)]
		a_inp = [rand(inp_dists[3], n_inp); zeros(maximum(inp_dists[1]) - n_inp)]

		_base_dudt(du, u, p, t) = prob_s2(du, u, p, t, t -> gauss_input!(du, t, t_inp, a_inp, size(t_inp)[1]), max_u0)

		prob 	 = DE.ODEProblem(_base_dudt, u0, tspan)
		prob_sol = Array(DE.solve(prob, DE.Rosenbrock23(), saveat=save_t, tstops=t_inp))

		if all_data == []
			all_data = reshape(prob_sol, (size(prob_sol)[1], size(prob_sol)[2], 1))
			all_inp_t = t_inp
			all_inp_a = a_inp
		else
			all_data = cat(all_data, reshape(prob_sol, (size(prob_sol)[1], size(prob_sol)[2], 1)), dims=3)
			all_inp_t = cat(all_inp_t, t_inp, dims=2)
			all_inp_a = cat(all_inp_a, a_inp, dims=2)
		end


	end

	return all_data, all_inp_t, all_inp_a
end



"""
hybrid_model_s2(du, u, p, t, I, net_obj, L)
Hybrid model S2 function used in training/prediction.

Parameters:

du, u, p, t - usual meaning in the DifferentialEquations.jl context.
I - input function that mutates u in-place
net_obj - Flux reconstructor object obtained from Flux.destructure
L - scaling factor for concentrations

HYBRID MODEL SPECIES
u[1]  - Sig
u[2]  - M3K
u[3]  - M3K*
u[4]  - MK
u[5]  - MK* 
u[6]  - MK**
u[7] - P1
u[8] - P2
u[9] - P3
u[10...] - augmented dimensions
"""
function hybrid_model_s2(du, u, p, t, I, net_obj, L)

	v1_v  = v1_f(u[1], u[2], u[6], L)
	v2_v  = v2_f(u[7], u[3], L)

	v9_v  =  v9_f(u[9], u[6], u[5], L)
	v10_v = v10_f(u[9], u[5], u[6], L)

	y = net_obj(p)(u)

 	du[1] = -u[1]
	du[2] = 1/L * (v2_v - v1_v)
	du[3] = 1/L * (v1_v - v2_v)
 	du[4] = 1/L * (v10_v)			- Flux.softplus(y[1])*u[4]
 	du[5] = 1/L * (v9_v - v10_v) 	+ Flux.softplus(y[1])*u[4] 	- Flux.softplus(y[2])*u[5]
 	du[6] = 1/L * (-v9_v)			    		   				+ Flux.softplus(y[2])*u[5]
	du[7]= 0.0
	du[8]= 0.0
	du[9]= 0.0

	du[10:end] .= Flux.tanh.(y[3:end])

	I(t)

	return du
end



""" 
predict_node_gaussian(INIT, NET_OBJ, P, T_INPUT, A_INPUT, TSPAN, SAVE_T, L)
Function used to produce data from a hybrid model S2.

Parameters:

INIT - function tha returns initial concentrations
NET_OBJ - reconstructor object obtained from Flux.destructure
P - array of parameters for neural network
T_INPUT - array of parameters with timing of inputs
A_Input - array of parameters with amplitude of inputs
TSPAN - tuple (t_start, t_end) with start and end times, eventually passed to DifferentialEquations.ODEProblem
SAVE_T - array of times at which the data is saved, eventually passed to DifferentialEquations.solve
L - float scaling factor to scale concentrations
"""
function predict_node_gaussian(INIT, NET_OBJ, P, T_INPUT, A_INPUT, TSPAN, SAVE_T, L)


	_model(du, u, p, t) = hybrid_model_s2(du, u, p, t, t -> gauss_input!(du, t, T_INPUT, A_INPUT, size(T_INPUT)[1]), NET_OBJ, L)
	model_prob = DE.ODEProblem(_model, INIT[[1:3;7:size(INIT)[1]]], TSPAN)


	out_data = Array(DE.solve(model_prob, DE.AutoTsit5(DE.Rosenbrock23(autodiff=false)), p=P, saveat=SAVE_T, tstops=T_INPUT, abstol=1e-7, reltol=1e-4)) 
end



"""
loss_node(P, X_T, X_A, Y, NET_RE, TSPAN, SAVE_T, L, LAM=0)
Loss (L2) function that makes predictions given some parameters and calculates error between real species present in both the true and the hybrid models.

Parameters:

P - parameters for the neural network in hybrid_model_s2
X_T - array of paramteres with timing of inputs
X_A - array of parameters with timing of inputs
Y - true data
NET_RE - reconstructor object obtained from Flux.destructure
TSPAN - tuple (t_start, t_end) with start and end times, eventually passed to DifferentialEquations.ODEProblem
SAVE_T - array of times at which the data is saved, eventually passed to DifferentialEquations.solve
L - float scaling factor to scale concentrations
LAM - float L1 regularization constant, defaults to 0
"""
function loss_node(P, X_T, X_A, Y, NET_RE, TSPAN, SAVE_T, L, LAM=0)

    full_loss   = 0.0
	pred 		= nothing

    for i in 1:size(Y)[3]

        pred = predict_node_gaussian(view(Y, :, 1, i), NET_RE, P, view(X_T,:,i), view(X_A,:,i), TSPAN, SAVE_T, L)
    	full_loss += sum(abs2.(view(Y, [2, 3, 7, 8, 9], :, i) .- view(pred, 2:6, :))) + sum(abs2.(view(pred, 10:size(pred)[1], :)))
    end

	full_loss = full_loss / size(Y)[3]
	full_loss += LAM * sum(abs.(P))

    return full_loss
end



""" 
do_plot(pred_dat, true_dat, str_folder, loss_list, param, tt, command)
Convenience function used for plotting used in a callback.

Parameters:

pred_dat - array of data from the hybrid_model_s2
true_dat - array of true data
str_folder - string for the folder where to save the plots 
loss_list - array with training loss values
param - array of parameters
tt - array of times at which data was saved
command - what to do with the plots- "save" (only saves), "show" (only shows), "show_save" (shows and saves), "pass" (do nothing)
"""
function do_plot(pred_dat, true_dat, str_folder, loss_list, param, tt, command)


	p1 = Plots.histogram(param, label="weights", color="black", bins=40, xlabel="Weights", ylabel="#")
	p2 = Plots.plot(tt, hcat(pred_dat[3,:], true_dat[3,:]), label=["Pred M3K*" "True M3K*"], linewidth=3, ylabel="Norm. conc (a.u.)", xlabel="Time (s)")
	p3 = Plots.plot(tt, hcat(pred_dat[5,:], true_dat[8,:]), label=["Pred MK*" "True MK*"], linewidth=3, ylabel="Norm. conc (a.u.)", xlabel="Time (s)")
	p4 = Plots.plot(tt, hcat(pred_dat[6,:], true_dat[9,:]), label=["Pred MK**" "True MK**"], linewidth=3, ylabel="Norm. conc (a.u.)", xlabel="Time (s)")
	p5 = Plots.plot(log10.(loss_list), color="black", linewidth=3, ylabel="Log10(MAE)", xlabel="Iteration #")
	p6 = Plots.plot(tt, pred_dat[10:end,:]', linewidth=3, title="Augmented dims")

	full_p = Plots.plot(p1, p2, p3, p4, p5, p6, layout=(3,2), size=(1200, 1200))
	Plots.display(full_p)

	if command == "save"
		Plots.png(full_p, str_folder * replace(string(Dates.now()), ':'=>'_') * ".png")
	elseif command == "show"
		Plots.display(full_p)
	elseif command == "show_save"
		Plots.display(full_p)
		Plots.png(full_p, str_folder * replace(string(Dates.now()), ':'=>'_') * ".png")
	elseif command == "pass"
	
	else
		println("Erroneous callback command given")
	end

end



"""
callback(param, l, predictor, last_data, str_folder, loss_list, tt, command)
Callback function for reporting things and plotting during training

Parameters:

param - array of training parameters
predictor - function f(p) that is used to predict values given some parameter p, currently only predict_node_gaussian is used
last_data - array for the last true simulation that is used for comparison (can't really plot all simulations), can be non-last too
str_folder - string for the folder where to save the plots
loss_list - array where the training loss should be !push'ed 
tt -  array of times where data was saved during simulations
command - what to do with the plots- "save" (only saves), "show" (only shows), "show_save" (shows and saves), "pass" (do nothing)
"""
function callback(param, l, predictor, last_data, str_folder, loss_list, tt, command)

	println("=================================")
	println("Loss ", l)
	println("Net L1 norm ", sum(abs.(param)))

	push!(loss_list, l)


	do_plot(predictor(param), last_data, str_folder, loss_list, param, tt, command)


	return false
	
end