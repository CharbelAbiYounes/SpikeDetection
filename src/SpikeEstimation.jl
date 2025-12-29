function AsympLanczos(
    W::AbstractMatrix{Float64};
    v::AbstractVector{Float64}=randn(size(W,1)),
    tol::Float64=3*sum(abs,diag(W))/(size(W,1)*sqrt(size(W,1))),
    seq_len::Int=floor(Int, log(size(W,1))÷2),
    jmp::Int=2,
    max_iter::Int=ceil(Int, max(6log(size(W,1))+24, size(W,1)/4, sqrt(size(W,1)))),
    workspace::Dict=Dict()
)
    n = size(W,1)
    z = get!(workspace, :z, zeros(Float64, n))
    d = get!(workspace, :d, zeros(Float64, max_iter))
    od = get!(workspace, :od, zeros(Float64, max_iter))
    proj = get!(workspace, :proj, zeros(Float64, max_iter))
    Q = get!(workspace, :Q, Matrix{Float64}(undef, n, max_iter))

    q = copy(v)
    q ./= norm(q)
    Q[:,1] .= q

    dAvrg_old = 0.0; odAvrg_old = 0.0
    dStd_old = 0.0; odStd_old = 0.0

    idx = seq_len
    i = 1
    Convflag = false

    while i ≤ max_iter && !Convflag
        mul!(z, W, q)
        d[i] = dot(q, z)

        Qi = @view Q[:,1:i]
        pv = @view proj[1:i]
        mul!(pv, Qi', z)
        mul!(z, Qi, pv, -1.0, 1.0)
        mul!(pv, Qi', z)
        mul!(z, Qi, pv, -1.0, 1.0)

        if i < max_iter
            od[i] = norm(z)
            if od[i] == 0
                return true, SymTridiagonal(@view(d[1:i]), @view(od[1:i-1])), i
            end
            q .= z ./ od[i]
            Q[:,i+1] .= q
            if i == idx
                start = max(1, i-seq_len+1)
                len = i - start + 1
                dseg = @view d[start:i]
                odseg = @view od[start:i]
                dAvrg = sum(dseg) / len
                odAvrg = sum(odseg) / len
                dStd = len > 1 ? sqrt(sum((dseg .- dAvrg).^2) / (len - 1)) : 0.0
                odStd = len > 1 ? sqrt(sum((odseg .- odAvrg).^2) / (len - 1)) : 0.0

                if dStd < tol && odStd < tol &&
                   dStd_old < tol && odStd_old < tol &&
                   abs(dAvrg - dAvrg_old) < tol &&
                   abs(odAvrg - odAvrg_old) < tol
                    mul!(z, W, q)
                    d[i+1] = dot(q, z)
                    Convflag = true
                else
                    dAvrg_old = dAvrg
                    odAvrg_old = odAvrg
                    dStd_old = dStd
                    odStd_old = odStd
                    idx += jmp
                end
            end
        end
        i += 1
    end
    used = min(i, max_iter)
    return Convflag, SymTridiagonal(@view(d[1:used]), @view(od[1:used-1])), used
end

function CholeskyList(
    W::AbstractMatrix{Float64},
    vecNbr::Int;
    vecList::AbstractMatrix{Float64}=randn(Float64,size(W,1),vecNbr),
    tol::Float64=3*sum(abs,diag(W))/(size(W,1)*sqrt(size(W,1))),
    seq_len::Int=floor(Int, log(size(W,1))÷2),
    jmp::Int=2,
    max_iter::Int=ceil(Int, max(6log(size(W,1))+24, size(W,1)/4, sqrt(size(W,1)))),
)
    @assert size(vecList,2)==vecNbr "vecNbr and size of vecList not matching!"
    n = size(W,1)
    TChol = Vector{Tridiagonal{Float64}}(undef, vecNbr)
    ModChol = Vector{Tridiagonal{Float64}}(undef, vecNbr)
    sizelist = zeros(Int, vecNbr)

    workspace = Dict{Symbol,Any}()
    
    d = zeros(Float64, max_iter)
    od = zeros(Float64, max_iter)

    @inbounds for j = 1:vecNbr
        v = @view(vecList[:, j])
        Convflag, T, used = AsympLanczos(W, tol=tol, seq_len=seq_len, jmp=jmp, max_iter=max_iter, v=v, workspace=workspace)
        TChol[j] = Cholesky(T)

        d[1:used] .= T.dv
        od[1:used-1] .= T.ev

        if Convflag
            i = used - jmp - seq_len - 1
            dSum  = sum(@view d[i+1 : used])
            odSum = sum(@view od[i+1 : used-1])
            dAsymp = dSum / (used - i)
            odAsymp = odSum / (used - i - 1)

            while i > 1 && abs(d[i] - dAsymp) < tol && abs(od[i] - odAsymp) < tol
                dSum  += d[i]
                odSum += od[i]
                i -= 1
            end
            dAsymp = dSum / (used - i)
            odAsymp = odSum / (used - i - 1)
        else
            i = max_iter - 2
            refd = d[max_iter-1]
            refod = od[max_iter-1]
            dSum = refd
            odSum = refod
            while i > 0 && abs(d[i] - refd) < tol && abs(od[i] - refod) < tol
                dSum += d[i]
                odSum += od[i]
                i -= 1
            end
            dAsymp = dSum / (max_iter - 1 - i)
            odAsymp = odSum / (max_iter - 1 - i)
        end

        d[i+1] = dAsymp
        d[i+2] = dAsymp
        od[i+1] = odAsymp

        ModChol[j] = Cholesky(SymTridiagonal(@view(d[1:i+2]), @view(od[1:i+1])))
        sizelist[j] = i+2
    end

    dAsymp_final = 0.0
    odAsymp_final = 0.0
    @inbounds for j = 1:vecNbr
        m = sizelist[j]
        T = ModChol[j]
        dAsymp_final += T[m,m] + T[m-1,m-1]
        odAsymp_final += T[m, m-1]
    end
    dAsymp_final /= (2*vecNbr)
    odAsymp_final /= vecNbr

    @inbounds for j = 1:vecNbr
        m = sizelist[j]
        T = ModChol[j]
        T[m,m] = dAsymp_final
        T[m-1,m-1] = dAsymp_final
        T[m, m-1] = odAsymp_final
    end

    return TChol, ModChol
end

function EstimSupp(ModChol::Vector{Tridiagonal{Float64}})
    n = size(ModChol[1],1)
    dAsymp = ModChol[1][n,n]
    odAsymp = ModChol[1][n,n-1]
    γmin = (dAsymp - odAsymp)^2
    γplus = (dAsymp + odAsymp)^2
    return γmin, γplus
end

function BiRel(m::Number,z::Number,d::Float64,od::Float64)
    return 1/(-z+d^2-d^2*od^2*(m/(1+od^2*m)))
end
function BiRef(z::Number,d::Float64,od::Float64)
    return (-z+d^2-od^2+sqrt(z-(d+od)^2)*sqrt(z-(d-od)^2))/(2*z*od^2)
end

function mASD(zs::AbstractVector{<:Number},L::Tridiagonal{Float64})
    ms = similar(zs)
    n = size(L,1)
    @. ms = BiRef(zs,L[n-1,n-1],L[n,n-1])
    @inbounds for j = n-2:-1:1
        @. ms = BiRel(ms,zs,L[j,j],L[j+1,j])
    end
    return ms
end

function EstimDensity(x::AbstractVector{<:Real},ModChol::Vector{Tridiagonal{Float64}};eps::Float64=1e-3)
    len_x = length(x)
    dens = zeros(Float64,len_x)
    len = length(ModChol)
    z = x .+ eps*im
    ms = similar(z)
    @inbounds for j=1:len
        ms .= mASD(z,ModChol[j])
        @. dens+=imag(ms)/π
    end
    return dens/len
end

function EstimSpike(TrueChol::Vector{Tridiagonal{Float64}},N::Int;
    δ::Float64=0.25,
    c::Float64=1.0,
    ThreshOut::Bool=false
)
    len = length(TrueChol)
    Vec = zeros(Int,len)
    sizelist = similar(Vec)
    L0 = TrueChol[1]
    γplus = (L0[end,end] + L0[end, end-1])^2
    thresh = γplus + c * N^(-δ)
    for i=1:len
        L = TrueChol[i]
        j = size(L,1)
        sizelist[i] = j
        evals = eigvals(SymTridiagonal(L*L'))
        Vec[i] = count(>(thresh), evals)
    end

    freq = Dict{Int, Int}()
    for v in Vec
        freq[v] = get(freq, v, 0) + 1
    end
    Nbr = findmax(freq)[2]
    maxval,maxidx = findmax(sizelist)
    L = TrueChol[maxidx]
    evals = eigvals(SymTridiagonal(L*L'))
    if !ThreshOut
        return Nbr, evals[end-Nbr+1:end]
    else
        return Nbr, evals[end-Nbr+1:end], Thresh
    end
end

function AsympSCM(
    W::AbstractMatrix{Float64};
    vecNbr::Int=50,
    vecList::AbstractMatrix{Float64}=randn(Float64,size(W,1),vecNbr),
    xgrid=nothing,
    nx::Int=500,
    eps::Float64=1e-3,
    tol::Float64=3*sum(abs,diag(W))/(size(W,1)*sqrt(size(W,1))),
    seq_len::Int=floor(Int, log(size(W,1))÷2),
    jmp::Int=2,
    max_iter::Int=ceil(Int, max(6log(size(W,1))+24, size(W,1)/4, sqrt(size(W,1)))),
    δ::Float64=0.25,
    c::Float64=1.0,
    workspace::Dict=Dict(),
    compute_density::Bool=true,
    compute_spikes::Bool=true,
    errplot::Bool=false
)
    @assert size(vecList,2)==vecNbr "vecNbr and size of vecList not matching!"
    n = size(W,1)

    vecList = @view vecList[:,1:vecNbr]

    TChol, ModChol = CholeskyList(W, vecNbr, vecList=vecList, tol=tol, seq_len=seq_len, jmp=jmp, max_iter=max_iter)
    γmin, γplus = EstimSupp(ModChol)

    result = Dict{Symbol, Any}()

    if compute_density
        if errplot
            x = collect(range(γmin+0.2, γplus - 0.2; length=convert(Int64,ceil(nx*(γplus-γmin)))))
        elseif xgrid==nothing
            x = collect(range(max(0.0, γmin - 0.05*(γplus-γmin)), γplus + 0.05*(γplus-γmin); length=convert(Int64,ceil(nx*(γplus-γmin)))))
        else
            x = collect(xgrid)
        end
        dens = EstimDensity(x, ModChol; eps=eps)
        result[:x] = x
        result[:density] = dens
    end

    if compute_spikes
        n_spikes, spike_eigs = EstimSpike(TChol, n; δ=δ, c=c, ThreshOut=false)
        result[:spikes_nbr] = n_spikes
        result[:spikes_loc] = spike_eigs
    end

    if compute_density || compute_spikes
        result[:γmin] = γmin
        result[:γplus] = γplus
    end

    return result
end

