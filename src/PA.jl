function PA(X::AbstractMatrix{<:Float64},svals::AbstractVector{Float64};
    np::Int=19,
    α::Int=100
)
    N,M = size(X)
    p = min(N,M)
    svalsMat = zeros(Float64,p,np)
    Xperm = similar(X)

    @inbounds for i in 1:np
        Xperm .= X
        for col in 1:M
            shuffle!(view(Xperm, :, col))
        end
        svalsMat[:, i] .= svdvals!(Xperm)
    end

    k=0
    idx = convert(Int64,floor(np*α/100))
    Hvec = sort(@view svalsMat[k+1,:])
    while k<N-2 && svals[k+1]>Hvec[idx]
        k+=1
        Hvec = sort(svalsMat[k+1,:])
    end
    return k
end

function DPA(X::AbstractMatrix{<:Float64},svals::AbstractVector{Float64};eps::Float64=0.0)
    N,M = size(X)
    d = N/M
    k=0
    D = vec(sum(abs2, X; dims=2) ./ M)
    B = -1/maximum(D)
    z(v) = -1/v+d*sum((D/N)./((1 .+D*v)))
    dz(v) = 1/(v^2) - d*sum((D.^2/N)./((1 .+D*v).^2))
    vcrit = Bisection(dz,B,0.0)
    Uval = z(vcrit)
    ConvFlag = false
    while k < N && svals[k+1]/sqrt(M) >= (1+eps)*sqrt(Uval)
        k += 1
    end
    return k
end

function DDPA(evals::AbstractVector{Float64},p::Int,d::Float64)
    k = 0
    ConvFlag = false
    while k≤p && !ConvFlag
        λ = evals[end-k]
        m = sum(1 ./ (evals[1:end-k-1] .- λ)) / (p-1)
        v = d*m-(1-d)/λ
        D = λ*m*v
        ℓ = 1/D
        dm = (1/(p-1)) * sum((evals[1:end-k-1] .- λ).^(-2.0))
        dv = d*dm+(1-d)/(λ^2)
        dD = m*v+λ*(m*dv+dm*v)
        cr2 = m/(dD*ℓ)
        cl2 = v/(dD*ℓ)
        if λ<4*ℓ^2*cr2*cl2
            k+=1
        else
            ConvFlag = true
        end
    end
    return k
end