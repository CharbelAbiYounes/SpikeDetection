using Distributed, SharedArrays,LinearAlgebra, Plots, Distributions, LaTeXStrings, Random, Optim, TracyWidomBeta, DataFrames, CSV, Distributed
include("/src/LanczosSpikeDetection.jl")
include("/src/AuxiliaryFunctions.jl")
include("/src/PA.jl")
include("/src/BEMA.jl")
include("/src/PassYao.jl")

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
    include("/src/LanczosSpikeDetection.jl")
    include("/src/AuxiliaryFunctions.jl")
    include("/src/PA.jl")
    include("/src/BEMA.jl")
    include("/src/PassYao.jl")
end

# Demo example

N = 6000
d = 0.1
M = convert(Int64,ceil(N/d))
X = randn(N,M)
K = 200
nodes,weights = Legendre(K)
a = 0.1
b = 4
h = x-> a<x<b ? (x^4+1)*sqrt(x-a)*sqrt(b-x)/x^2 : 0
normCst = QuadInt(h,a,b,nodes,weights)
scaled_h = x->h(x)/normCst
quantiles = zeros(Float64,N+1)
quantiles[1] = a
quantiles[N+1] = b
for i=2:N
    QuantEq = x->QuadInt(scaled_h,a,x,nodes,weights)-(i-1)/N
    quantiles[i] = Bisection(QuantEq,quantiles[1],quantiles[N+1])
end
δ = 6
quantiles[1:3] = [7,δ,δ]
sqrtΣ = Diagonal(sqrt.(quantiles[1:end-1]))
W = sqrtΣ*(1/M*X*X')*sqrtΣ'|>Symmetric
evals = eigvals(W)
true_spikes = evals[end:-1:end-2]
p3 = histogram(quantiles,bins=quantiles[4]-0.2:0.1:quantiles[1]+0.2,normalize=:pdf,label="ESD of Σ",framestyle=:box, legendfontsize=12, xtickfontsize=12, ytickfontsize=12)
vecNbr = 200
k = convert(Int64,floor(log(N)/2))
jmp = 5
tol = 2/sqrt(N)
max_iter = convert(Int64,ceil(max(6*log(N)+24,N/4,sqrt(N))))
TChol,L_list = CholeskyList(W,tol,k,jmp,max_iter,vecNbr)
γmin, γplus = EstimSupp(L_list)
x = -0.1+γmin:0.001:γplus+0.1
yAvrg = EstimDensity(x,L_list,N)
SpikeNbr, SpikeLoc = EstimSpike(TChol,N,c=1.0)
p4 = histogram(evals,bins=γmin-0.2:0.1:γplus+0.2,normalize=:pdf,label="ESD of "*L"W",framestyle=:box, legendfontsize=12, xtickfontsize=12, ytickfontsize=12)
p4 = plot!(x,yAvrg,linecolor=:red,linewidth=3,label="Estimated density")
p4 = scatter!(SpikeLoc,0*SpikeLoc,markersize=5,color=:red,marker=:dot,label="Estimated outliers")
p4 = scatter!(true_spikes,0*true_spikes,markersize=5,color=:blue,marker=:xcross,label="True outliers")
savefig(p3, "DemoSigma.pdf")
savefig(p4, "DemoDensity.pdf")
tb1 = DataFrame(A=true_spikes,B=SpikeLoc,C=abs.(true_spikes-SpikeLoc))
CSV.write("DemoSpikes.csv",tb1)

# Detailed example

N = 2500 
d = 0.1
M = convert(Int64,ceil(N/d))
X = randn(N,M)
K = 200
nodes,weights = Legendre(K)
a = 0.1
b = 4
h = x-> a<x<b ? (2*(3.5-x)^3+x)*(b-x)^(1/2)*(x-a)^(1/2)/(2*(4.5-x)^2) : 0
normCst = QuadInt(h,a,b,nodes,weights)
scaled_h = x->h(x)/normCst
quantiles = zeros(Float64,N+1)
quantiles[1] = a
quantiles[N+1] = b
for i=2:N
    QuantEq = x->QuadInt(scaled_h,a,x,nodes,weights)-(i-1)/N
    quantiles[i] = Bisection(QuantEq,quantiles[1],quantiles[N+1])
end
δ = 6
quantiles[1:2] = [7,δ]
sqrtΣ = Diagonal(sqrt.(quantiles[1:end-1]))
W = sqrtΣ*(1/M*X*X')*sqrtΣ'|>Symmetric
evals = eigvals(W)
true_spikes = evals[end:-1:end-1]
p3 = histogram(quantiles,bins=quantiles[3]-0.2:0.1:quantiles[1]+0.2,normalize=:pdf,label="ESD of Σ",framestyle=:box, legendfontsize=12, xtickfontsize=12, ytickfontsize=12)
vecNbr = 100
k = convert(Int64,floor(log(N)/2))
jmp = 5
tol = 3/sqrt(N)
max_iter = convert(Int64,ceil(max(6*log(N)+24,N/4,sqrt(N))))
TChol,L_list = CholeskyList(W,tol,k,jmp,max_iter,vecNbr)
γmin, γplus = EstimSupp(L_list)
x = -0.1+γmin:0.001:γplus+0.1
yAvrg = EstimDensity(x,L_list,N)
SpikeNbr, SpikeLoc = EstimSpike(TChol,N,c=1.0)
p4 = histogram(evals,bins=γmin-0.2:0.1:γplus+0.2,normalize=:pdf,label="ESD of "*L"W",framestyle=:box, legendfontsize=12, xtickfontsize=12, ytickfontsize=12)
p4 = plot!(x,yAvrg,linecolor=:red,linewidth=3,label="Estimated density")
p4 = scatter!(SpikeLoc,0*SpikeLoc,markersize=5,color=:red,marker=:dot,label="Estimated outliers")
p4 = scatter!(true_spikes,0*true_spikes,markersize=5,color=:blue,marker=:xcross,label="True outliers")
savefig(p3, "Sigma.pdf")
savefig(p4, "Density.pdf")
tb4 = DataFrame(A=true_spikes,B=SpikeLoc,C=abs.(true_spikes-SpikeLoc))
CSV.write("Spikes.csv",tb4)

logfile = open("progress.log", "w")
N = 2500
d = 0.1
M = convert(Int64,ceil(N/d))
K = 200
nodes,weights = Legendre(K)
a = 0.1
b = 4
h = x-> a<x<b ? (2*(3.5-x)^3+x)*(b-x)^(1/2)*(x-a)^(1/2)/(2*(4.5-x)^2) : 0
normCst = QuadInt(h,a,b,nodes,weights)
scaled_h = x->h(x)/normCst
quantiles = zeros(Float64,N+1)
quantiles[1] = a
quantiles[N+1] = b
for i=2:N
    QuantEq = x->QuadInt(scaled_h,a,x,nodes,weights)-(i-1)/N
    quantiles[i] = Bisection(QuantEq,quantiles[1],quantiles[N+1])
end
sqrtΣ = Diagonal(sqrt.(quantiles[1:end-1]))
δvec = [5,7,9,21]
lenδ = length(δvec)
MPquants = quantMP(N,d)
jmp = 5
tol = 3/sqrt(N)
max_iter = convert(Int64,ceil(max(6*log(N)+24,N/4,sqrt(N))))
k = convert(Int64,floor(log(N)/2))
vecNbr = 1
t = TWquant(0.1)
SampleNbr = 100 
SpikeNbr = 2
Percent = zeros(Float64,lenδ,5)
Avrg = zeros(Float64,lenδ,5)
Time = zeros(Float64,lenδ,5)

@everywhere function process_sample(N, M, d, sqrtΣ, SpikeNbr, MPquants, t, tol, k, jmp, max_iter, vecNbr)
    locPercent = zeros(Float64,5)
    locAvrg = zeros(Float64,5)
    locTime = zeros(Float64,5)
    X = randn(N,M)
    W = sqrtΣ*X*X'*sqrtΣ/M|>Symmetric        
    time_evals = @elapsed begin
        evals = eigvals(W)
    end
    time_BEMA0 = @elapsed begin
        BEMA0Out = BEMA0(evals,MPquants,d,0.2,t)
        locPercent[1] = BEMA0Out==SpikeNbr ? 1 : 0
        locAvrg[1] = BEMA0Out
    end
    time_BEMA = @elapsed begin
        BEMAOut = BEMA(evals,0.2,N,d;SampleNbr=75)
        locPercent[2] = BEMAOut==SpikeNbr ? 1 : 0
        locAvrg[2] = BEMAOut
    end
    time_PassYao = @elapsed begin
        PassYaoOut = PassYao(evals,d,N;SampleNbr=50)
        locPercent[3] = PassYaoOut==SpikeNbr ? 1 : 0
        locAvrg[3] = PassYaoOut
    end
    time_DDPA = @elapsed begin
        DDPAOut = DDPA(evals,N,d)
        locPercent[4] = DDPAOut==SpikeNbr ? 1 : 0
        locAvrg[4] = DDPAOut
    end
    time_CholList = @elapsed begin
        TChol,L_list = CholeskyList(W,tol,k,jmp,max_iter,vecNbr)
        Nbr,Loc = EstimSpike(TChol,N,c=1.0)
        locPercent[5] = Nbr==SpikeNbr ? 1 : 0
        locAvrg[5] = Nbr
    end
    locTime[1] = time_evals+time_BEMA0
    locTime[2] = time_evals+time_BEMA
    locTime[3] = time_evals+time_PassYao
    locTime[4] = time_evals+time_DDPA
    locTime[5] = time_CholList
    return (locPercent,locAvrg,locTime)
end

for i=1:lenδ
    δ = δvec[i]
    sqrtΣ[1,1] = sqrt(7)
    sqrtΣ[2,2] = sqrt(δ)
    results = pmap(_-> process_sample(N, M, d, sqrtΣ, SpikeNbr, MPquants, t, tol, k, jmp, max_iter, vecNbr), 1:SampleNbr)
    locPercent = first.(results)
    locAvrg = getindex.(results, 2)
    locTime = getindex.(results, 3)

    Percent[i,:] = vec(reduce( .+, locPercent))/SampleNbr
    Avrg[i,:] = vec(reduce( .+, locAvrg))/SampleNbr
    Time[i,:] = vec(reduce( .+, locTime))/SampleNbr
    msg = "δ=$(δ)\n"
    print(msg)
    write(logfile, msg)
    flush(logfile)
    tb = DataFrame(A=δvec[1:i],B=Percent[1:i,1],C=Percent[1:i,2],D=Percent[1:i,3],E=Percent[1:i,4],F=Percent[1:i,5])
    CSV.write("Percent.csv",tb)
    tb = DataFrame(A=δvec[1:i],B=Avrg[1:i,1],C=Avrg[1:i,2],D=Avrg[1:i,3],E=Avrg[1:i,4],F=Avrg[1:i,5])
    CSV.write("Avrg.csv",tb)
    tb = DataFrame(A=δvec[1:i],B=Time[1:i,1],C=Time[1:i,2],D=Time[1:i,3],E=Time[1:i,4],F=Time[1:i,5])
    CSV.write("Time.csv",tb)
end
close(logfile)