function PA(X,svals;np::Integer=19,α::Integer=100)
    N,M = size(X)
    p = min(N,M)
    svalsMat = zeros(Float64,p,np)
    Xperm = similar(X)
    S = zeros(p)
    for i=1:np
        Xperm = mapslices(shuffle, X, dims=2)
        SVDRes = svd(Xperm)
        svalsMat[:,i] = SVDRes.S
    end
    k=0
    idx = convert(Int64,floor(np*α/100))
    Hvec = sort(@view svalsMat[k+1,:])
    while k<N-2 && svals[k+1]>Hvec[idx]
        k+=1
        Hvec = sort(@view svalsMat[k+1,:])
    end
    return k
end

function DPA(svals,X,N::Integer,d::Float64;eps=0)
    k=0
    M = convert(Int64,ceil(N/d))
    D = diag(X*X'/M,0)
    B = -1/maximum(D)
    z = v->-1/v+d*sum((D/N)./((1 .+D*v)))
    dz = v->1/(v^2) - d*sum((D.^2/N)./((1 .+D*v).^2))
    vcrit = Bisection(dz,B,0.0)
    Uval = z(vcrit)
    ConvFlag = false
    while k<N && !ConvFlag
        if (svals[k+1]/sqrt(M))<(1+eps)*sqrt(Uval)
            ConvFlag = true
        else
            k+=1
        end
    end
    return k
end

function DDPA_NoDef(svals,U,V,N::Integer,d::Float64;eps=0)
    k=0
    M = convert(Int64,ceil(N/d))
    ConvFlag = false
    X = zeros(Float64,N,M)
    D = zeros(Float64,N,N)
    while k<N && !ConvFlag
        @. X = (@view U[:,1:end-k])*Diagonal((@view svals[1:end-k]))*(@view V[:,1:end-k])'
        @. D = diag(X*X'/M,0)
        B = -1/maximum(D)
        z = v->-1/v+d*sum((D/N)./((1 .+D*v)))
        dz = v->1/(v^2) - d*sum((D.^2/N)./((1 .+D*v).^2))
        vcrit = Bisection(dz,B,0.0)
        Uval = z(vcrit)
        if (svals[k+1]/sqrt(M))<(1+eps)*sqrt(Uval)
            ConvFlag = true
        else
            k+=1
        end
    end
    return k
end

function DDPA(evals,p::Integer,d::Float64)
    k = 0
    ConvFlag = false
    while k≤p && !ConvFlag
        λ = evals[end-k]
        m = (1/(p-1))*sum(((@view evals[1:end-k-1]).-λ).^(-1.0))
        v = d*m-(1-d)/λ
        D = λ*m*v
        ℓ = 1/D
        dm = (1/(p-1))*sum(((@view evals[1:end-k-1]).-λ).^(-2.0))
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