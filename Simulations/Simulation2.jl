using LinearAlgebra, Distributions, Random, Plots, LaTeXStrings, DataFrames, CSV, Distributed, Optim, TracyWidomBeta
include("/src/LanczosSpikeEstimation.jl")
include("/src/AuxiliaryFunctions.jl")
include("/src/PA.jl")
include("/src/BEMA.jl")

f = open("hosts.txt")
nodes = readlines(f)
close(f)
num_procs = 40
addprocs([nodes[2] for j in 1:num_procs],tunnel=true)
addprocs([nodes[3] for j in 1:num_procs],tunnel=true)
addprocs([nodes[4] for j in 1:num_procs],tunnel=true)
addprocs(num_procs-1)
@everywhere begin
    using LinearAlgebra, Distributions, Random, Optim, TracyWidomBeta
    include("/src/LanczosSpikeEstimation.jl")
    include("/src/AuxiliaryFunctions.jl")
    include("/src/PA.jl")
    include("/src/BEMA.jl")
end

logfile = open("progress.log", "w")
N = 8000 
d = 0.5
M = convert(Int64,ceil(N/d))
σ = 1
δvec = vcat(1.5:0.01:3)
σvec = σ^2*ones(N)
sqrtΣ = Diagonal(sqrt.(σvec))
MPquants = quantMP(N,d)
jmp = 5
tol = 3/sqrt(N)
t = TWquant(0.1)
max_iter = convert(Int64,ceil(max(6*log(N)+24,sqrt(N))))
k = convert(Int64,floor(log(N)/2))
vecNbr = [1,50,100,200]
t = TWquant(0.1)
lenδ = length(δvec)
SampleNbr = 100 
Percent = zeros(Float64,lenδ,6)
Avrg = zeros(Float64,lenδ,6)
Time = zeros(Float64,lenδ,6)
EvalTime = zeros(Float64,lenδ)

@everywhere function process_sample(N, M, d, sqrtΣ, SpikeNbr, MPquants, t, tol, k, jmp, max_iter, vecNbr)
    locPercent = zeros(Float64,6)
    locAvrg = zeros(Float64,6)
    locTime = zeros(Float64,6)
    X = randn(Float64,N,M)
    W = sqrtΣ*X*X'*sqrtΣ/M |> Symmetric
    time_evals = @elapsed begin
        evals = eigvals(W)
    end
    time_BEMA0 = @elapsed begin
        BEMA0Out = BEMA0(evals,MPquants,d,0.2,t)
        locPercent[1] = BEMA0Out==SpikeNbr ? 1 : 0
        locAvrg[1] = BEMA0Out
    end
    time_DDPA = @elapsed begin
        DDPAOut = DDPA(evals,N,d)
        locPercent[2] = DDPAOut==SpikeNbr ? 1 : 0
        locAvrg[2] = DDPAOut
    end
    time_CholList1 = @elapsed begin
        TChol,L_list = CholeskyList(W,tol,k,jmp,max_iter,vecNbr[1])
        Nbr,Loc = EstimSpike(TChol,N,c=1.0)
        locPercent[3] = Nbr==SpikeNbr ? 1 : 0
        locAvrg[3] = Nbr
    end
    time_CholList2 = @elapsed begin
        TChol,L_list = CholeskyList(W,tol,k,jmp,max_iter,vecNbr[2])
        Nbr,Loc = EstimSpike(TChol,N,c=1.0)
        locPercent[4] = Nbr==SpikeNbr ? 1 : 0
        locAvrg[4] = Nbr
    end
    time_CholList3 = @elapsed begin
        TChol,L_list = CholeskyList(W,tol,k,jmp,max_iter,vecNbr[3])
        Nbr,Loc = EstimSpike(TChol,N,c=1.0)
        locPercent[5] = Nbr==SpikeNbr ? 1 : 0
        locAvrg[5] = Nbr
    end
    time_CholList4 = @elapsed begin
        TChol,L_list = CholeskyList(W,tol,k,jmp,max_iter,vecNbr[4])
        Nbr,Loc = EstimSpike(TChol,N,c=1.0)
        locPercent[6] = Nbr==SpikeNbr ? 1 : 0
        locAvrg[6] = Nbr
    end
    locTime[1] = time_evals+time_BEMA0
    locTime[2] = time_evals+time_DDPA
    locTime[3] = time_CholList1
    locTime[4] = time_CholList2
    locTime[5] = time_CholList3
    locTime[6] = time_CholList4
    return (locPercent,locAvrg,locTime,time_evals)
end

for i=1:lenδ
    δ = δvec[i]
    sqrtΣ[1,1] = sqrt(6)
    sqrtΣ[2,2] = sqrt(5)
    sqrtΣ[3,3] = sqrt(δ)
    if δ≤1+sqrt(d)
        SpikeNbr = 2
    else
        SpikeNbr = 3
    end
    results = pmap(_-> process_sample(N, M, d, sqrtΣ, SpikeNbr, MPquants, t, tol, k, jmp, max_iter, vecNbr), 1:SampleNbr)
    locPercent = first.(results)
    locAvrg = getindex.(results, 2)
    locTime = getindex.(results, 3)
    loctime_evals = getindex.(results, 4)

    Percent[i,:] = vec(reduce( .+, locPercent))/SampleNbr
    Avrg[i,:] = vec(reduce( .+, locAvrg))/SampleNbr
    Time[i,:] = vec(reduce( .+, locTime))/SampleNbr
    EvalTime[i] = sum(loctime_evals)/SampleNbr
    msg = "δ=$(δ)\n"
    print(msg)
    write(logfile, msg)
    flush(logfile)
    tb = DataFrame(A=δvec[1:i],B=Percent[1:i,1],C=Percent[1:i,2],D=Percent[1:i,3],E=Percent[1:i,4],F=Percent[1:i,5],G=Percent[1:i,6])
    CSV.write("NormPercent.csv",tb)
    tb = DataFrame(A=δvec[1:i],B=Avrg[1:i,1],C=Avrg[1:i,2],D=Avrg[1:i,3],E=Avrg[1:i,4],F=Avrg[1:i,5],G=Avrg[1:i,6])
    CSV.write("NormAvrg.csv",tb)
    tb = DataFrame(A=δvec[1:i],B=Time[1:i,1],C=Time[1:i,2],D=Time[1:i,3],E=Time[1:i,4],F=Time[1:i,5],G=Time[1:i,6])
    CSV.write("NormTime.csv",tb)
    tb = DataFrame(A=δvec[1:i],B=EvalTime[1:i])
    CSV.write("NormEvalTime.csv",tb)
end
close(logfile)