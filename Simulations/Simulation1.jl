using LinearAlgebra, Distributions, Random, Plots, LaTeXStrings, DataFrames, CSV
include("/src/LanczosSpikeDetection.jl")

f = open("hosts.txt")
nodes = readlines(f)
close(f)
num_procs = 40
addprocs([nodes[2] for j in 1:num_procs],tunnel=true)
addprocs([nodes[3] for j in 1:num_procs],tunnel=true)
addprocs([nodes[4] for j in 1:num_procs],tunnel=true)
addprocs(num_procs-1)
@everywhere begin
    using LinearAlgebra, Distributions, Random
    include("/src/LanczosSpikeDetection.jl")
end

# Density and spike estimation for different values of d

N = 5000
σ = sqrt(1.5)
σout = [5,5,4.5]
σvec = σ^2*ones(N)
σvec[1:3] = σout
sqrtΣ = Diagonal(sqrt.(σvec))
k = convert(Int64,floor(log(N)/2))
jmp = 5
tol = 3/sqrt(N)
max_iter = convert(Int64,ceil(max(6*log(N)+24,sqrt(N))))
vecNbr = 100
dvec = [0.1,0.5,0.9]
len_d = length(dvec)
for m=1:len_d
    d = dvec[m]
    M = convert(Int64,ceil(N/d))
    X = randn(N,M)
    W = sqrtΣ*(1/M*X*X')*sqrtΣ'|>Symmetric
    evals = eigvals(W)
    TChol,L_list = CholeskyList(W,tol,k,jmp,max_iter,vecNbr)
    SpikeNbr, SpikeLoc = EstimSpike(TChol,N,c=1.0)
    γmin, γplus = EstimSupp(L_list)
    true_γplus = σ^2*(1+sqrt(d))^2
    true_γmin = σ^2*(1-sqrt(d))^2
    true_spikes = evals[end-2:end]
    x = -0.1+min(γmin,true_γmin):0.001:max(γplus,true_γplus)+0.1
    yAvrg = EstimDensity(x,L_list,N)
    MPdens = x-> true_γmin<x<true_γplus ? sqrt(true_γplus-x)*sqrt(x-true_γmin)/(2*π*d*x*σ^2) : 0
    true_y = map(x->MPdens(x),x)
    p = histogram(evals,bins=evals[1]-0.2:0.1:evals[end]+0.2,normalize=:pdf,label="ESD of "*L"W", legendfontsize=12, framestyle=:box, xtickfontsize=12, ytickfontsize=12)
    p = plot!(x,yAvrg,linecolor=:red,linewidth=3,label="Estimated Density")
    p = scatter!(SpikeLoc,0*SpikeLoc,markersize=5,color=:red,marker=:dot,label="Estimated Spikes")
    p = plot!(x,true_y,linecolor=:blue,linewidth=2,linestyle=:dash,legend=:topright,label="True Density")
    p = scatter!(true_spikes,0*true_spikes,markersize=5,color=:blue,marker=:xcross,label="True Spikes")
    savefig(p,"Density"*string(m)*".pdf")
    tb = DataFrame(A=true_spikes,B=SpikeLoc,C=abs.(true_spikes-SpikeLoc))
    CSV.write("Spikes"*string(m)*".csv",tb)
end

# Comparing efficiency and accuracy

Nvec = 100:100:8000
lenN = length(Nvec)
dvec = [0.1,0.5,0.9]
len_d = length(dvec)
σ = sqrt(1.5)
σout = [5,5,4.5]
jmp = 5
vecNbr = 1
SampleNbr = 200 
Percent = zeros(Float64,len_d,lenN)
SpikeNbr = zeros(Float64,len_d,lenN)
LanTime = zeros(Float64,len_d,lenN)
EigTime = zeros(Float64,len_d,lenN)

@everywhere function Effprocess(N, M, sqrtΣ, tol, k, jmp, max_iter, vecNbr)
    X = randn(N,M)
    W = sqrtΣ*(1/M*X*X')*sqrtΣ'|>Symmetric
    EigDuration = @timed begin
        evals = eigvals(W)
    end
    LanDuration = @timed begin
        TChol,L_list = CholeskyList(W,tol,k,jmp,max_iter,vecNbr)
        Nbr, Loc = EstimSpike(TChol,N,c=1.0)
    end
    return (EigDuration.time, LanDuration.time, Nbr)
end

logfile = open("progress.log", "w")
for m in 1:len_d
    d = dvec[m]
    for ℓ in 1:lenN
        N = Nvec[ℓ]
        M = convert(Int64,ceil(N/d))
        k = convert(Int64,floor(log(N)/2))
        tol = 3/sqrt(N)
        max_iter = convert(Int64,ceil(max(6*log(N)+24,sqrt(N))))
        σvec = σ^2*ones(N)
        σvec[1:3] = σout
        sqrtΣ = Diagonal(sqrt.(σvec))
        results = pmap(_-> Effprocess(N, M, sqrtΣ, tol, k, jmp, max_iter, vecNbr), 1:SampleNbr)
        EigTime[m, ℓ] = sum(first.(results)) / SampleNbr
        LanTime[m, ℓ] = sum(x -> x[2], results) / SampleNbr  
        SpikeNbr[m, ℓ] = sum(x -> x[3], results) / SampleNbr
        Percent[m, ℓ] = sum(Int(x[3] == 3) for x in results) / SampleNbr  
        msg = "d=$(d) and N=$(N)\n"
        print(msg)
        write(logfile, msg)
        flush(logfile)
        tb = DataFrame(A=Nvec[1:ℓ],B=Percent[m,1:ℓ])
        CSV.write("Prct"*string(m)*".csv",tb)
        tb = DataFrame(A=Nvec[1:ℓ],B=SpikeNbr[m,1:ℓ])
        CSV.write("Avrg"*string(m)*".csv",tb)
        tb = DataFrame(A=Nvec[1:ℓ],B=LanTime[m,1:ℓ])
        CSV.write("LanTime"*string(m)*".csv",tb)
        tb = DataFrame(A=Nvec[1:ℓ],B=EigTime[m,1:ℓ])
        CSV.write("EigTime"*string(m)*".csv",tb)
    end
end
close(logfile)

p = plot(Nvec,Percent[1,:],color=:red,linewidth=3,label="",xlabel="N",ylabel="Probability of correct estimation",legend=:bottomright,framestyle=:box, legendfontsize=12, xtickfontsize=12, ytickfontsize=12)
p = scatter!(Nvec, Percent[1,:], markersize=4, color=:red, marker=:diamond, label="d="*string(dvec[1]))
p = plot!(Nvec,Percent[2,:],color=:blue,linewidth=3,label="")
p = scatter!(Nvec, Percent[2,:], markersize=4, color=:blue, marker=:square, label="d="*string(dvec[2]))
p = plot!(Nvec,Percent[3,:],color=:green,linewidth=3,label="")
p = scatter!(Nvec, Percent[3,:], markersize=4, color=:green, marker=:circ, label="d="*string(dvec[3]))
savefig(p,"Ex1Prct.pdf")
p = plot(Nvec,EigTime[2,:],color=:red,linewidth=2,label="",xlabel="N",ylabel="Time (in seconds)",legend=:topleft,framestyle=:box, legendfontsize=12, xtickfontsize=12, ytickfontsize=12)
p = scatter!(Nvec, EigTime[2,:], markersize=4, color=:red, marker=:diamond, label="Eigenvalue Computation")
p = plot!(Nvec,LanTime[2,:],color=:orange,linewidth=2,label="")
p = scatter!(Nvec, LanTime[2,:], markersize=4, color=:orange, marker=:square, label="Lanczos Approach")
savefig(p,"Ex1Time05.pdf")

# Support and density errors

logfile = open("progress.log", "w")
Nvec = vcat(200:200:3000,3500:500:8000)
len_N = length(Nvec)
dvec = [0.1,0.5,0.9]
len_d = length(dvec)
σ = sqrt(1.5)
σ_out = [5,5,4.5]
jmp = 5
vecNbr = 100
SampleNbr = 200
SuppErr = zeros(Float64,len_d,SampleNbr,len_N)
hErr = zeros(Float64,len_d,SampleNbr,len_N)

@everywhere function Errprocess(d, N, σ, σ_out, jmp, vecNbr)
    σvec = σ^2 * ones(N)
    σvec[1:3] = σ_out
    sqrtΣ = Diagonal(sqrt.(σvec))
    tol = 3 / sqrt(N)
    k = convert(Int64, floor(log(N) / 2))
    max_iter = convert(Int64, ceil(max(6 * log(N) + 24, sqrt(N))))
    M = convert(Int64, ceil(N / d))
    X = randn(N, M)
    W = sqrtΣ * (1 / M * X * X') * sqrtΣ' |> Symmetric
    TChol, L_list = CholeskyList(W, tol, k, jmp, max_iter, vecNbr)
    γmin, γplus = EstimSupp(L_list)
    true_γplus = σ^2 * (1 + sqrt(d))^2
    true_γmin = σ^2 * (1 - sqrt(d))^2
    supp_err = max(abs(true_γmin - γmin), abs(true_γplus - γplus))
    x = γmin + 0.2:0.01:γplus - 0.2
    len_x = length(x)
    MP_h = x -> true_γmin < x < true_γplus ? 1 / (2 * π * d * x * σ^2) : 0
    true_h = map(MP_h, x)
    yAvrg = EstimDensity(x, L_list, N)
    hAvrg = yAvrg ./ (sqrt.(γplus .- x) .* sqrt.(x .- γmin))
    h_err = maximum(abs.(hAvrg - true_h))
    return (supp_err, h_err)
end

for m in 1:len_d
    d = dvec[m]
    for ℓ in 1:len_N
        N = Nvec[ℓ]
        results = pmap(_ -> Errprocess(d, N, σ, σ_out, jmp, vecNbr), 1:SampleNbr;batch_size=10)
        SuppErr[m, :, ℓ] .= first.(results)
        hErr[m, :, ℓ] .= last.(results)
        msg = "d=$(d) and N=$(N)\n"
        print(msg)
        write(logfile, msg)
        flush(logfile)
    end
end
close(logfile)

SuppErr1 = vec(sum(SuppErr[1,:,:],dims=1)/SampleNbr)
SuppErr2 = vec(sum(SuppErr[2,:,:],dims=1)/SampleNbr)
SuppErr3 = vec(sum(SuppErr[3,:,:],dims=1)/SampleNbr)
hErr1 = vec(sum(hErr[1,:,:],dims=1)/SampleNbr)
hErr2 = vec(sum(hErr[2,:,:],dims=1)/SampleNbr)
hErr3 = vec(sum(hErr[3,:,:],dims=1)/SampleNbr)
SuppErr1_std = vec(std(SuppErr[1, :, :], dims=1))
SuppErr2_std = vec(std(SuppErr[2, :, :], dims=1))
SuppErr3_std = vec(std(SuppErr[3, :, :], dims=1))
hErr1_std = vec(std(hErr[1, :, :], dims=1))
hErr2_std = vec(std(hErr[2, :, :], dims=1))
hErr3_std = vec(std(hErr[3, :, :], dims=1))
tb = DataFrame(A=Nvec,B=vec(SuppErr1),C=vec(SuppErr1_std),D=vec(SuppErr2),E=vec(SuppErr2_std),F=vec(SuppErr3),G=vec(SuppErr3_std))
CSV.write(joinpath(tableFolder,"SuppErr.csv"),tb)
tb = DataFrame(A=Nvec,B=vec(hErr1),C=vec(hErr1_std),D=vec(hErr2),E=vec(hErr2_std),F=vec(hErr3),G=vec(hErr3_std))
CSV.write(joinpath(tableFolder,"DensErr.csv"),tb)
checkpt = convert(Int64,ceil(len_N/4))
p = plot(Nvec,(SuppErr3[checkpt]*(Nvec[checkpt])^(1/2))*(Nvec).^(-1/2),color=:orange,linewidth=3,label=L"\mathbf{N}^{\mathbf{-1/2}}",xlabel="N",ylabel="Errors in support",legend=:topright,yscale=:log10,linestyle=:dash,framestyle=:box, legendfontsize=12, xtickfontsize=12, ytickfontsize=12)
p = plot!(Nvec,SuppErr1,yerror=SuppErr1_std,color=:red,linewidth=2,label="")
p = scatter!(Nvec, SuppErr1, markersize=4, color=:red, marker=:diamond, label="d="*string(dvec[1]))
p = plot!(Nvec,SuppErr2,yerror=SuppErr2_std,color=:blue,linewidth=2,label="")
p = scatter!(Nvec, SuppErr2, markersize=4, color=:blue, marker=:square, label="d="*string(dvec[2]))
p = plot!(Nvec,SuppErr3,yerror=SuppErr3_std,color=:green,linewidth=2,label="")
p = scatter!(Nvec, SuppErr3, markersize=4, color=:green, marker=:circ, label="d="*string(dvec[3]))
savefig(p, "SuppErr.pdf")
p = plot(Nvec,(hErr1[checkpt]*(Nvec[checkpt])^(1/2))*(Nvec).^(-1/2),color=:orange,linewidth=3,label=L"\mathbf{N}^{\mathbf{-1/2}}",xlabel="N",ylabel="Errors in density",legend=:topright,yscale=:log10,linestyle=:dash,framestyle=:box, legendfontsize=12, xtickfontsize=12, ytickfontsize=12)
p = plot!(Nvec,hErr1,yerror=hErr1_std,color=:red,linewidth=2,label="")
p = scatter!(Nvec, hErr1, markersize=4, color=:red, marker=:diamond, label="d="*string(dvec[1]))
p = plot!(Nvec,hErr2,yerror=hErr2_std,color=:blue,linewidth=2,label="")
p = scatter!(Nvec, hErr2, markersize=4, color=:blue, marker=:square, label="d="*string(dvec[2]))
p = plot!(Nvec,hErr3,yerror=hErr3_std,color=:green,linewidth=2,label="")
p = scatter!(Nvec, hErr3, markersize=4, color=:green, marker=:circ, label="d="*string(dvec[3]))
savefig(p,"DensErr.pdf")