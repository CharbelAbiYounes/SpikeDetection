# BEMA0

@inline function MP(x::Float64,d::Float64,dm::Float64,dp::Float64;σ::Float64=1.0)
    if dm<x<dp
        return (1/(2*π*x*σ^2*min(d,1)))*sqrt(x-dm)*sqrt(dp-x)
    else
        return 0
    end
end

function quantMP(N::Int,d::Float64;σ::Float64=1.0)
    quants = zeros(Float64,N+1)
    dm = (1-sqrt(d))^2
    dp = (1+sqrt(d))^2
    quants[end] = dp
    quants[1] = dm
    K = 400
    nodes,weights = LegQuad(K)
    MPdens = x->MP(x,d,dm,dp,σ=σ)
    for i=1:N-1
        QuantEq = x->LegQuadInt(MPdens,x,dp,nodes,weights)-i/N
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

function BEMA0(evals::AbstractVector{Float64},d::Float64, t::Float64, MPquants::AbstractVector{Float64}; 
    α::Float64=0.2, 
    β::Float64=0.1, 
    ThreshOut::Bool=false
)
    p = length(evals)
    lb = convert(Int64,floor(α*p))
    ub = p-lb
    ev_slice = @view evals[lb:ub]
    mp_slice = @view MPquants[lb+1:ub+1]
    num = sum(ev_slice .* mp_slice)
    denom = sum(mp_slice .^ 2)
    σ2 = num / denom

    σ2 = sum((@view evals[lb:ub]).*(@view(MPquants[lb+1:ub+1])))/sum(((@view MPquants[lb+1:ub+1])).^2)
    M = convert(Int64,ceil(p/d))

    thresh = σ2*((1+sqrt(d))^2+t*M^(-2/3)*d^(-1/6)*(1+sqrt(d))^(4/3))
    count = 0
    @inbounds for i in p:-1:1
        if evals[i] > thresh
            count += 1
        else
            break
        end
    end

    if !ThreshOut
        return count
    else
        return count, thresh
    end
end

#BEMA

function loss(proposal::Float64,evals::AbstractVector{Float64},α::Float64,d::Float64,N::Int)
    M = convert(Int64,ceil(N/d))
    p = d<1 ? N : M
    L = zeros(Float64,10, p)
    D = Diagonal(rand(Gamma(proposal,1/proposal),N))
    mv_normal = MvNormal(zeros(N), D)

    @inbounds for i in 1:10
        x1 = rand(mv_normal, M)
        W = d < 1 ? Symmetric(x1*x1'/M) : Symmetric(x1'*x1/M)
        L[i, :] = eigvals(W)'
    end

    l1 = vec(mean(L,dims=1))
    lb = convert(Int64,floor(α*p))
    ub = p-lb
    k = lb:ub
    s1 = (l1[k] \ evals[k])[1]
    l1 .*= s1
    return sum((l1[k] .- evals[k]) .^ 2)
end

function BEMA(evals::AbstractVector{Float64},d::Float64;
    SampleNbr::Int=100,
    α::Float64=0.2,
    β::Float64=0.1,
    ThreshOut::Bool=false
)
    N = length(evals)
    res = optimize(θ->loss(θ,evals,α,d,N),0.1,50,Brent();iterations=10)
    θ = Optim.minimizer(res)

    M = convert(Int64,ceil(N/d))
    p = d<1 ? N : M
    L = zeros(Float64,SampleNbr, p)
    D = Diagonal(rand(Gamma(θ,1/θ),N))
    mv_normal = MvNormal(zeros(N), D)

    @inbounds for i in 1:SampleNbr
        x1 = rand(mv_normal, M)
        W = d < 1 ? Symmetric(x1*x1'/M) : Symmetric(x1'*x1/M)
        L[i, :] = eigvals(W)'
    end

    l1 = vec(mean(L,dims=1))
    l2 = mapslices(x -> quantile(x, β), L; dims=1) |> vec
    lb = convert(Int64,floor(α*p))
    ub = p-lb
    k = lb:ub
    s1 = (l1[k] \ evals[k])[1]
    thresh = maximum(l2.*s1)
    count = 0
    @inbounds for i in p:-1:1
        if evals[i] > thresh
            count += 1
        else
            break
        end
    end

    if !ThreshOut
        return count
    else
        return count, thresh
    end
end