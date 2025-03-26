# BEMA0

function MP(x::Float64,d::Float64)
    dm = (1-sqrt(d))^2
    dp = (1+sqrt(d))^2
    if dm<x<dp
        return (1/(2*π*x*min(d,1)))*sqrt(x-dm)*sqrt(dp-x)
    else
        return 0
    end
end

function quantMP(N::Integer,d::Float64)
    quants = zeros(Float64,N+1)
    dm = (1-sqrt(d))^2
    dp = (1+sqrt(d))^2
    quants[end] = dp
    quants[1] = dm
    K = 200
    nodes,weights = Legendre(K)
    MPdens = x->MP(x,d)
    for i=1:N-1
        QuantEq = x->QuadInt(MPdens,x,dp,nodes,weights)-i/N
        quants[N-i+1] = Bisection(QuantEq,quants[1],quants[N-i+2])
    end
    return quants
end

function TWquant(β::Float64)
    F = TW(1)
    Bis_f = x->F(x)-(1-β)
    t = Bisection(Bis_f,-10.0,13.0)
    return t
end

function BEMA0(evalsr,MPquants,d::Float64,α::Float64,t;ThreshOut::Bool=false)
    p = length(evals)
    lb = convert(Int64,floor(α*p))
    ub = p-lb
    σ2 = sum((@view evals[lb:ub]).*(@view(MPquants[lb+1:ub+1])))/sum(((@view MPquants[lb+1:ub+1])).^2)
    if d<1
        M = convert(Int64,ceil(p/d))
    else
        M = p
    end
    thresh = σ2*((1+sqrt(d))^2+t*M^(-2/3)*d^(-1/6)*(1+sqrt(d))^(4/3))
    count = 0
    i = p
    while i>0 && evals[i]>thresh
        count+=1
        i-=1
    end
    if !ThreshOut
        return count
    else
        return count, thresh
    end
end

#BEMA

function loss(proposal::Float64,evals,α::Float64,d::Float64,N::Integer)
    M = convert(Int64,ceil(N/d))
    L = d<1 ? zeros(Float64,10, N) : zeros(Float64,10, M)
    D = Diagonal(rand(Gamma(proposal,1/proposal),N))
    mv_normal = MvNormal(zeros(N), D)
    p = d<1 ? N : M
    if d<1
        for i in 1:10
            x1 = rand(mv_normal,M)
            W = Symmetric(x1*x1'/M)
            l1 = eigvals(W)
            L[i,:] = l1'
        end
    else
        for i in 1:10
            x1 = rand(mv_normal,M)
            W = Symmetric(x1'*x1/M)
            l1 = eigvals(W)
            L[i,:] = l1'
        end
    end
    l1 = vec(mean(L,dims=1))
    lb = convert(Int64,floor(α*p))
    ub = p-lb
    k = lb:ub
    s1 = (l1[k] \ evals[k])[1]
    l1 .*= s1
    return sum((l1[k] .- evals[k]) .^ 2)
end

function BEMA(evals,α::Float64,N::Integer,d::Float64;SampleNbr::Integer=500,β::Float64=0.1,ThreshOut::Bool=false)
    res = optimize(θ->loss(θ,evals,α,d,N),0.1,50,Brent();iterations=20)
    θ = Optim.minimizer(res)
    M = convert(Int64,ceil(N/d))
    L = d<1 ? zeros(Float64,SampleNbr, N) : zeros(Float64,SampleNbr, M)
    D = Diagonal(rand(Gamma(θ,1/θ),N))
    mv_normal = MvNormal(zeros(N), D)
    p = d<1 ? N : M
    if d<1
        for i in 1:SampleNbr
            x1 = rand(mv_normal,M)
            W = Symmetric(x1*x1'/M)
            l1 = eigvals(W)
            L[i,:] = l1'
        end
    else
        for i in 1:SampleNbr
            x1 = rand(mv_normal,M)
            W = Symmetric(x1'*x1/M)
            l1 = eigvals(W)
            L[i,:] = l1'
        end
    end
    l1 = vec(mean(L,dims=1))
    l2 = vec([quantile(col, β) for col in eachcol(L)])
    lb = convert(Int64,floor(α*p))
    ub = p-lb
    k = lb:ub
    s1 = (l1[k] \ evals[k])[1]
    thresh = maximum(l2.*s1)
    count = 0
    i=p
    while i>0 && evals[i]>thresh
        count+=1
        i-=1
    end
    if !ThreshOut
        return count
    else
        return count, thresh
    end
end