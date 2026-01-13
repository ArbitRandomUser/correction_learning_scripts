"""
    exp_folder:: a string, the name of the expeirment folder
"""
function get_pulses_from_folder(exp_folder)
    list_of_pulses_unordered = []
    results = JSON.parsefile("experiments/$exp_folder/counts")
    
    no_of_files = length(filter(
        fname -> fname[end-3:end] == ".npy" && occursin("pulse", fname),
        readdir("experiments/$exp_folder/pulses"),
    ))
    list_of_pulses_unordered = []
    for i in 1:no_of_files
        ff = filter(fname -> fname[end-3:end] == ".npy" && startswith(fname, "$(i)_"),
            readdir("experiments/$exp_folder/pulses"))
        @assert length(ff) == 1
        ff = ff[1]
        push!(list_of_pulses_unordered, NPZ.npzread("experiments/$exp_folder/pulses/$ff"))
    
    end
end


"""
    turns result Dict into a result prob dist Array 
"""
function arrayify(sys::System, result)
    ret = zeros(Float64, 2^sys.N)
    for k in keys(result)
        ret[parse(Int, k, base=2)+1] = result[k]
    end
    return ret / sum(ret)
end

function matshow(s::System,x,sym=:M)
    bound = s.dim^2
    if sym==:M
        reshape(x[1:bound],(s.dim,s.dim))
    elseif sym==:D1
        reshape(x[bound+1:2*bound],(s.dim,s.dim))
    elseif sym==:D2
        reshape(x[2*bound+1:3*bound],(s.dim,s.dim))
    elseif sym==:D3 #out of bounds for 2 qubit system
        reshape(x[3*bound+1:4*bound],(s.dim,s.dim))
    end
end

"""
    loss funciton for quantum state u w.r.t some probability distribution result
"""
function loss(sys::System, u, res)
    #s = u[sys.repart].^2 .+ u[sys.impart].^2
    #snew = view(s,sys.twoindices)
    # #return sum(abs.(u[sys.repart] .^ 2 .+ u[sys.impart] .^ 2 .- res))
    #return sum(abs.(snew .- res))
    ss = 0.0
    imstart = sys.impart.start
    for (i,j) in enumerate(sys.twoindices)
        ss += abs(u[j]^2 + u[imstart-1+j]^2 - res[i])
    end
    return ss
end

function make_dloss_du(sys::System, loss::Function, res)
    ret = zeros(2*sys.dim)
    dres = make_zero(res)
    function dloss_du(out, u, p, t, i)
        #ret = zero(u)
        #dres = make_zero(res)
        ret .= 0
        dres .= 0
        autodiff(Reverse, loss, Active, Const(sys), Duplicated(u, ret), Duplicated(res, dres))
        out .= ret
        nothing
    end
end

function get_derivative(S::System, dl_du, MDparams, pulse_seq, u0 ; abstol=1e-5, reltol=1e-6, method=Tsit5(), train_M=false,train_Ds=(2,))
    prob, MDparams = make_MDprob(S, pulse_seq; MDparams=MDparams,u0=u0, train_M,train_Ds)
    sol = solve(prob, Tsit5();abstol,reltol)
    adj_sense = adjoint_sensitivities(
        sol,
        Tsit5(),
        t=[pulse_tspan(S, pulse_seq)[2],],
        dgdu_discrete=dl_du,
        sensealg=InterpolatingAdjoint(autodiff=true, autojacvec=EnzymeVJP());
        abstol,
        reltol,
    )
    return sol, adj_sense
end


"""
    get_derivatives:
    S :: System
    dloss_dus :: Array of functions , a loss for every result
    MDparams :: M and D params
    pulses :: list of Pulses
"""
function get_derivatives(S, dloss_dus, MDparams, pulses, init_states ; train_M=false,train_Ds=(2,), abstol=1e-9, reltol=1e-9)
    sols = Any[nothing for _ in eachindex(pulses)]
    derivatives = Any[nothing for _ in eachindex(pulses)]
    @floop FLoops.ThreadedEx(basesize=1) for i in eachindex(pulses)
        pulse_seq = pulses[i]
        u0 = init_states[i]
        sol, deriv = get_derivative(S, dloss_dus[i], MDparams, pulse_seq,u0; abstol, reltol, train_M, train_Ds)
        sols[i] = sol
        derivatives[i] = deriv
    end
    return sols, derivatives
end

get_derivatives(ds::Dict,MDparams;abstol=1e-6,reltol=1e-6) = begin 
    get_derivatives(ds[:sys],ds[:dloss_dus],MDparams,ds[:pulses],ds[:init_states];abstol=abstol,reltol=reltol)
end

function get_sols(sys, MDparams, pulses, u0; abstol=1e-9, reltol=1e-9)
    sols = Any[nothing for _ in eachindex(pulses)]
    for i in eachindex(pulses)
        prob, MDparams = make_MDprob(sys, pulses[i]; MDparams=MDparams,u0=u0)
        sols[i] = solve(prob,Tsit5())
    end
    return sols
end

function get_init_state(s::Array,sys::System)
    ret = zeros(2*sys.dim)
    num = *(string.(s)...) #convert to string and concat
    i = parse(Int,num,base=sys.lvls)
    ret[i+1] = 1.0
    ret
end


function getdatasetloss(ds,sections = [1:length(ds[:pulses]),])
    ret = 0.0
    for section in sections
        for i in section 
            init_state = TMSimulator.cstate(ds[:init_states][i],ds[:sys])
            sol = evolve_oncomplex(ds[:sys],ds[:pulses][i],init_state)
            #@show loss(ds[:sys],vcat(real(sol[end]),imag(sol[end]) ),ds[:dloss_dus][i].res)
            ret+= loss(ds[:sys],vcat(real(sol[end]),imag(sol[end]) ),ds[:dloss_dus][i].res)
        end
    end
    #return ret/length(ds[:pulses])
    return ret/sum(length(section) for section in sections) 
end

get_dataset_loss_params(ds::Dict,MDparams) = begin
    #sols1,_ = get_derivatives(ds,MDparams)
    sols = Any[nothing for _ in 1:length(ds[:pulses])]
    @floop for i in 1:length(ds[:pulses])
        prob,_ = make_MDprob(ds[:sys],ds[:pulses][i];MDparams,u0=ds[:init_states][i])
        sol = solve(prob,Tsit5(),abstol=1e-6,reltol=1e-6)
        sols[i] = sol
    end
    sum( loss(ds[:sys],sols[j][end],ds[:dloss_dus][j].res) for j in eachindex(sols) )/length(sols)
end


function train_loop(n::Int, S::System, MDparams, dloss_dus, pulses, init_state, lr=0.0001; abstol=1e-6, reltol=1e-6)
    @assert length(dloss_dus) == length(pulses)
    ret = []
    for i in 1:n
        @time sols, derivs = get_derivatives(S, dloss_dus, MDparams, pulses ,init_state; abstol, reltol);
        lossval = sum(loss(S,sols[j][end],dloss_dus[j].res) for j in eachindex(sols))/length(dloss_dus)
        println("loss :",lossval, "current lr: ",lr)
        if !isempty(ret)
            if lossval > ret[end]
                lr=lr*0.9
            elseif abs(lossval - ret[end]) < 5e-4
                lr=lr*1.1
            end
        end
        push!(ret,lossval)
        for j in eachindex(dloss_dus)
            MDparams .= MDparams .- (1 / length(dloss_dus)) .* lr .* derivs[j][2]'
        end
        #display(matshow(S,derivs[1][2],:M))
        println("-------------------")
    end
    sols, derivs = get_derivatives(S, dloss_dus, MDparams, pulses ,init_state; abstol, reltol);
    lossval = sum(loss(S,sols[j][end],dloss_dus[j].res) for j in eachindex(sols))/length(dloss_dus)
    push!(ret,lossval)
    return ret
end


"""
    run n iterations of optimisation
    S :: the system
    MDparams :: params to optimize over
    pulses :: pulse , refer to TMSimulator for format
    init_state :: init state of the system
    opt_state  :: optimizer state

    showmat :: matrix to show while optimizing
"""
function train_loop_optim(n::Int,S::System,MDparams,dloss_dus,pulses,init_state,opt_state;abstol=1e-6,reltol=1e-6,train_M=false,train_Ds=(2,),showmat=:M)
    @assert length(dloss_dus) == length(pulses)
    lossvals = []
    for i in 1:n
        display("iteration $i")
        display(matshow(S,MDparams,showmat))
        @time sols,derivs = get_derivatives(S,dloss_dus,MDparams,pulses,init_state;abstol,reltol,train_M=train_M,train_Ds=train_Ds)
        lossval = sum(loss(S,sols[j][end],dloss_dus[j].res) for j in eachindex(sols))/length(dloss_dus)
        push!(lossvals,lossval)
        println("loss :",lossval)
        totderivs = zero(derivs[1][2]')
        for j in eachindex(derivs)
            totderivs += derivs[j][2]'
        end
        totderivs = totderivs ./ length(dloss_dus)
        #display(matshow(S,totderivs))
        opt_state,_ = Optimisers.update!(opt_state,MDparams,totderivs)
        println("___________________________________________")
    end
    return (opt_state,lossvals)
end

"""
    sorts eigen vectors, to make it look like I
    use for near I matrices with permuted columns only
"""
function sorteigen(eigen)
    M = eigen.vectors
    E = eigen.values
    retM = zero(M)
    retE = zero(E)
    for i in 1:(size(M)[1])
        ind = findfirst(x->x==maximum(abs.(M[:,i])),abs.(M[:,i]))
        retM[:,ind] .= M[:,i]
        retE[ind] = E[i]
    end
    return Eigen(retE,retM)
end
