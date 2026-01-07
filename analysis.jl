using Revise
using Plots
using FLoops
using TMSimulator

using Enzyme
using SciMLSensitivity
using DifferentialEquations
using JSON
using NPZ
using Optimisers

lvls = 3
errormat(p1,p2) = [1.0-p1     p2;
                   p1        1.0-p2;]
errormat(p) = errormat(p,p)
include("util.jl")

exp_folders = filter(dirname->isdir("experiments/findataset/2qubit/$dirname"),readdir("experiments/findataset/2qubit"))

foldersdata = Dict() 
for folder in exp_folders
    full_folder_name = "experiments/findataset/2qubit/$folder"
    metadata = JSON.parsefile(full_folder_name*"/metadata.json")
    errors = metadata["errors"]
    p_m0_p1 = metadata["prob_meas0_prep1"]
    p_m1_p0 = metadata["prob_meas1_prep0"]
    selected_qubits = metadata["qubits_used"]   
    dt = "dt" in keys(metadata) ? metadata["dt"]*1e9 : 0.5 
    error_matrices = Any[nothing for _ in 1:length(selected_qubits)]
    for k in keys(errors)
        error_matrices[parse(Int,k)+1] = errormat(p_m1_p0[k],p_m0_p1[k])
    end
    full_error_mat = kron(reverse(error_matrices)...)
    hamiltonian = JSON.parsefile(full_folder_name * "/h_$folder.json",dicttype=Dict{Any,Any})
    sys1 = make_system(hamiltonian, selected_qubits; lvls=3, dt=dt)
    list_of_pulses_unordered = []
    results = JSON.parsefile(full_folder_name * "/counts")
    no_of_files = length(filter(
        fname -> fname[end-3:end] == ".npy" && occursin("pulse", fname),
        readdir(full_folder_name * "/pulses"),
    ))
    list_of_pulses_unordered = []
    for i in 1:no_of_files
        ff = filter(fname -> fname[end-3:end] == ".npy" && startswith(fname, "$(i)_"),
            readdir(full_folder_name * "/pulses"))
        @assert length(ff) == 1
        ff = ff[1]
        push!(list_of_pulses_unordered, NPZ.npzread(full_folder_name * "/pulses/$ff"))

    end
    list_of_pulses = []
    for pulses in list_of_pulses_unordered
        @assert length(pulses)%(3*length(selected_qubits)) == 0
        pcount = length(pulses) ÷ (3 * length(selected_qubits))
        pulse = zeros(3 * sys1.N * pcount)
        for q in 1:length(selected_qubits)
            set_pulse!(sys1, pulse, q, :inphase, pulses[3*pcount*(q-1)+1:3*pcount*(q-1)+pcount])
            set_pulse!(sys1, pulse, q, :quad, pulses[3*pcount*(q-1)+pcount+1:3*pcount*(q-1)+2pcount])
            set_pulse!(sys1, pulse, q, :freq, pulses[3*pcount*(q-1)+2*pcount+1:3*pcount*(q-1)+3*pcount] .* 2pi ./ 1e9)
        end
        push!(list_of_pulses, pulse)
    end
    pulse_metadata = []
    for i in 1:no_of_files
        pulse_metadir = full_folder_name * "/pulse_metadata"
        if isdir(pulse_metadir)
            ff = filter(fname -> fname[end-4:end] == ".json" && startswith(fname,"$(i)_"),
                         readdir(pulse_metadir))
            @assert length(ff) == 1
            ff= ff[1] 
            push!(pulse_metadata,JSON.parsefile(pulse_metadir*"/$ff"))
        else 
            val = Dict(["init_state"=>[0,0],"pulse_vals"=>[]])
            push!(pulse_metadata,val)
        end
    end
    dloss_dus = [make_dloss_du(sys1,loss,full_error_mat\arrayify(sys1,res)) for res in results]
    init_states = [get_init_state(pmdat["init_state"],sys1) for pmdat in pulse_metadata]
    foldersdata[folder]=Dict(:sys=>sys1,
                             :pulses=>list_of_pulses,
                             :pulse_metadata=>pulse_metadata,
                             :dloss_dus => dloss_dus,
                             :results=>results,
                             :errormat=>full_error_mat,
                             :init_states=>init_states
                            )
end

function fetchregion(d::Dict,r)
    Dict(:sys=>d[:sys] ,
         :pulses=>d[:pulses][r],
         :results=>d[:results][r],
         :pulse_metadata=>d[:pulse_metadata][r],
         :dloss_dus => d[:dloss_dus][r],
         :errormat=>d[:errormat],
         :init_states=>d[:init_states][r])
end

function checkintegrity(dat,amp1,amp2,err=true)
    retflag = true
    len = length(dat[:pulses])
    for i in 1:len 
        maxval = maximum(TMSimulator.get_pulse(dat[:sys],dat[:pulses][i],1,:inphase)) 
        if  maxval!=amp1
            err ? error("maximum of amp1 is not $amp1 but $maxval at index $i") :
            println("maximum of amp1 is not $amp1 but $maxval at index $i")
            println("dataset with amps $(amp1)_$(amp2) is not correct")
            println("_________________________")
            retflag = false
            break
        end
        maxval= maximum(TMSimulator.get_pulse(dat[:sys],dat[:pulses][i],2,:inphase))
        if maxval!=amp2
            err ? error("maximum of amp2 is not $amp2 but is $maxval at index $i") :
            println("maximum of amp2 is not $amp2 but is $maxval at index $i")
            println("dataset with amps $(amp1)_$(amp2) is not correct")
            println("_________________________")
            retflag = false
            break
        end
    end
    retflag
end

train_set(ds,section,offset=20) = begin
    (vcat(ds[:dloss_dus][section],ds[:dloss_dus][section.+offset]),
    vcat(ds[:pulses][section],ds[:pulses][section.+offset]),
    vcat(ds[:init_states][section],ds[:init_states][section.+offset]))
end


dataset = Dict{String,Any}()
amps = Iterators.product([0.0,0.01,0.02],[0.1,0.2,0.4]) 
ranges = vcat([1:60,],[st:st+39 for st in 61:40:380])
for (amp,r) in zip(amps,ranges)
    dataset["$(amp[1])_$(amp[2])_00"] = fetchregion(foldersdata["exp_data_01_Feb_25_19_45_09"],r)
end

amps = Iterators.product([0.0,0.01,0.02],[0.3,0.5,0.6]) 
ranges = [st:st+39 for st in 1:40:360 ]
length(foldersdata["exp_data_02_Feb_25_01_42_04"][:pulses])
for (amp,r) in zip(amps,ranges)
    dataset["$(amp[1])_$(amp[2])_00"] = fetchregion(foldersdata["exp_data_02_Feb_25_01_42_04"],r)
end

amps = Iterators.product([0.03,0.04],[0.1,0.2,0.3,0.4,0.5,0.6]) 
ranges = [st:st+39 for st in 1:40:480 ]
length(foldersdata["exp_data_02_Feb_25_16_21_57"][:pulses])
for (amp,r) in zip(amps,ranges)
    dataset["$(amp[1])_$(amp[2])_00"] = fetchregion(foldersdata["exp_data_02_Feb_25_16_21_57"],r)
end

amps = Iterators.product([0.0,0.01,0.02,0.03,0.04],[0.1,0.2,0.3,0.4,0.5,0.6])
ranges = [st:st+9 for st in 1:10:300] 
length(foldersdata["exp_data_02_Feb_25_16_56_59"][:pulses])
for (amp,r) in zip(amps,ranges)
    dataset["$(amp[1])_$(amp[2])_01"] = fetchregion(foldersdata["exp_data_02_Feb_25_16_56_59"],r)
end

for key in keys(dataset)
    splits = split(key,'_')
    amp1,amp2 = parse.(Float64,(splits[1],splits[2]))
    if (amp2==0.3) && (amp1==0.03 || amp1==0.04)
        checkintegrity(dataset[key],amp1,amp2,false)
    else
        checkintegrity(dataset[key],amp1,amp2,true)
    end
end

allkeys = keys(dataset)
keys00 = filter((key)->key[end-1:end]=="00", allkeys)
keys01 = filter((key)->key[end-1:end]=="01", allkeys)

##cell
allmdparams = Dict() 
for key in collect(keys00)[1:1]
    ds1 = dataset[key]
    Msize = ds1[:sys].dim*ds1[:sys].dim
    Ds = (zeros(ds1[:sys].dim,ds1[:sys].dim) for _ in 1:ds1[:sys].N)
    MDparams = zeros(Msize + ds1[:sys].N*Msize)
    if key in ["0.0_0.1_00"]
        totloss = getdatasetloss(ds1,(1:15,31:45))
        lr = 0.00001 
        rule = Optimisers.Nesterov(lr)
        opt_state = Optimisers.setup(rule,MDparams)
        opt_state,lossvals = train_loop_optim(5,ds1[:sys],MDparams,train_set(ds1,1:15,30)...,opt_state)
    else
        totloss = getdatasetloss(ds1,(1:10,21:30))
        lr = 0.00001
        rule = Optimisers.Nesterov(lr)
        opt_state = Optimisers.setup(rule,MDparams)
        opt_state,lossvals = train_loop_optim(5,ds1[:sys],MDparams,train_set(ds1,1:10,20)...,opt_state)
    end
    allmdparams[key]=(MDparams,lossvals,opt_state)  
end

##cell
init_state = zeros(ComplexF64,ds1[:sys].dim);init_state[1]=1.0
sol = evolve_oncomplex(ds1[:sys],ds1[:pulses][end-8],init_state;)#abstol=1e-9,reltol=1e-6)
probdist(sol[end],ds1[:sys])
plotsol(sol,ds1[:sys])
##cell
for key in filter(k-> begin
                      kk=split(k,'_')
                      kk[1]=="0.03" && kk[2] == "0.6"
                  end
                  ,keys00)
    ds1 = dataset[key]
    
    if key == "0.0_0.1_00"
        #section = 1:30; ind = "00"
        section = 31:60;ind = "10"
    else
        #section = 1:20; ind = "00"
        section = 21:40;ind = "10"
    end
    
    #section = 1:20; ind = "00"
    mdkey = key[1:end-2]*"00"
    
    ind1 = parse(Int,ind,base=2)+1
    ind2 = parse(Int,ind,base=ds1[:sys].lvls)+1
    
    init_state = zeros(ComplexF64,ds1[:sys].dim);init_state[ind2]=1.0
    pltdat = Any[nothing for _ in 1:length(section)]
    pltdat2 = Any[nothing for _ in 1:length(section)]
    hard = Any[nothing for _ in 1:length(section)]
    hard2 = Any[nothing for _ in 1:length(section)]
    tspans = Any[nothing for _ in 1:length(section)]
    @floop for (j,i) in enumerate(section)
        #init_state = ds1[:init_states][i]
        init_statec = TMSimulator.cstate(ds1[:init_states][i],ds1[:sys])
        sol = evolve_oncomplex(ds1[:sys],ds1[:pulses][i],init_state;)#abstol=1e-9,reltol=1e-6)
        prob,_ = make_MDprob!(ds1[:sys],ds1[:pulses][i];MDparams=allmdparams[mdkey][1],u0=ds1[:init_states][i])
        #prob,_ = make_MDprob!(ds1[:sys],ds1[:pulses][i];MDparams,u0=ds1[:init_states][i])
        sol2 = solve(prob,Tsit5(),abstol=1e-6,reltol=1e-6)
        pltdat[j] = probdist(sol[end],ds1[:sys])[ind2]
        pltdat2[j]=probdist(sol2[end],ds1[:sys])[ind2]
        hard[j] = ds1[:dloss_dus][i].res[ind1]
        tspans[j]=pulse_tspan(ds1[:sys],ds1[:pulses][i])[2]
    end
    
    plt = plot(tspans,pltdat,label="sim $ind2",title="$(key)_$(section)",yrange=(0.0,1.1))
    plot!(plt,tspans,pltdat2,label="simc $ind2")
    plot!(plt,tspans,hard,label="hard")
    display(plt)
end

##cell

for key in filter(k-> begin
                      kk=split(k,'_')
                      #kk[1]=="0.03"
                      kk[1]=="0.03" && kk[2] == "0.6"
                  end
                  ,keys01)
    ds1 = dataset[key]
    
    #section = 1:5; ind = "01"
    section = 6:10;ind = "11"
    mdkey = key[1:end-2]*"00"
    
    ind1 = parse(Int,ind,base=2)+1
    ind2 = parse(Int,ind,base=ds1[:sys].lvls)+1
    
    init_state = zeros(ComplexF64,ds1[:sys].dim);init_state[ind2]=1.0
    pltdat = Any[nothing for _ in 1:length(section)]
    pltdat2 = Any[nothing for _ in 1:length(section)]
    hard = Any[nothing for _ in 1:length(section)]
    hard2 = Any[nothing for _ in 1:length(section)]
    tspans = Any[nothing for _ in 1:length(section)]
    @floop for (j,i) in enumerate(section)
        init_statec = TMSimulator.cstate(ds1[:init_states][i],ds1[:sys])
        sol = evolve_oncomplex(ds1[:sys],ds1[:pulses][i],init_state;)#abstol=1e-9,reltol=1e-6)
        prob,_ = make_MDprob!(ds1[:sys],ds1[:pulses][i];MDparams=allmdparams[mdkey][1],u0=ds1[:init_states][i])
        sol2 = solve(prob,Tsit5(),abstol=1e-6,reltol=1e-6)
        pltdat[j] = probdist(sol[end],ds1[:sys])[ind2]
        pltdat2[j]=probdist(sol2[end],ds1[:sys])[ind2]
        hard[j] = ds1[:dloss_dus][i].res[ind1]
        tspans[j]=pulse_tspan(ds1[:sys],ds1[:pulses][i])[2]
    end
    
    plt = plot(tspans,pltdat,label="simulated $ind2",title="$(key)_$(section)",marker=:circle,yrange=(0.0,1.1))
    plot!(plt,tspans,pltdat2,label="corrected simulat $ind2",marker=:circle)
    plot!(plt,tspans,hard,label="hard",marker=:circle)
    display(plt)
end
##cell
