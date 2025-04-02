function Cholesky(T::SymTridiagonal)
    n = size(T)[1]
    H = Tridiagonal(copy(T))
    for k = 1:n-1
        H[k+1,k+1] = H[k+1,k+1] - H[k+1,k]^2/H[k,k]
        H[k:k+1,k] = H[k:k+1,k]/sqrt(H[k,k])
        H[k,k+1] = 0.
    end
    H[n,n] = sqrt(H[n,n])
    return H
end

function CholeskyConv(mat,tol::Float64,k::Integer,jmp::Integer,max_iter::Integer;v=randn(size(mat,1)),orth::Bool=true)
    m, n = size(mat)
    Q = zeros(eltype(mat),n, max_iter)
    q = v / norm(v)
    Q[:, 1] .= q
    d = zeros(eltype(mat),max_iter)
    od = zeros(eltype(mat),max_iter-1)
    z = similar(q)
    dAvrg_old, odAvrg_old, dStd_old, odStd_old = zeros(Float64,4)
    idx = k
    i=1
    Convflag = false
    while i<=max_iter && !Convflag 
        z .= mat * q
        d[i] = dot(q, z)
        if orth
            Qview = @view Q[:, 1:i]
            z .-= Qview * (Qview' * z)
            z .-= Qview * (Qview' * z)
        else
            z .-= d[i] * q
            if i > 1
                z .-= (od[i-1]) * (@view Q[:, i-1])
            end
        end
        if i < max_iter
            od[i] = norm(z)
            if od[i]==0
                return Cholesky(SymTridiagonal(d[1:i],od[1:i-1]))
            end
            q .= z / od[i]
            Q[:, i+1] .= q
            if i==idx 
                dAvrg = sum(@view d[i-k+1:i])/k
                odAvrg = sum(@view od[i-k+1:i])/k
                dStd = sqrt(sum(((@view d[i-k+1:i]).-dAvrg).^2)/(k-1))
                odStd = sqrt(sum(((@view od[i-k+1:i]).-odAvrg).^2)/(k-1))
                if dStd<tol && odStd<tol && dStd_old<tol && odStd_old<tol && abs(dAvrg-dAvrg_old)<tol && abs(odAvrg-odAvrg_old)<tol
                    z .= mat * q
                    d[i+1] = dot(q, z)
                    Convflag = true
                else
                    dAvrg_old, odAvrg_old, dStd_old, odStd_old = dAvrg, odAvrg, dStd, odStd
                    idx = idx+jmp
                end
            end
        end
        i+=1
    end
    i = min(i,max_iter)
    return Convflag, Cholesky(SymTridiagonal(d[1:i],od[1:i-1]))
end

function CholeskyList(mat,tol::Float64,k::Integer,jmp::Integer,max_iter::Integer,vecNbr::Integer;Modtol::Float64=tol,vecList=randn(size(mat,1),vecNbr))
    ModChol = Vector{Matrix{Float64}}(undef, vecNbr)
    TChol = Vector{Matrix{Float64}}(undef, vecNbr)
    sizelist = zeros(Int64,vecNbr)
    d = zeros(Float64,max_iter)
    od = zeros(Float64,max_iter)
    for j=1:vecNbr
        Convflag, TChol[j] = CholeskyConv(mat,tol,k,jmp,max_iter,v=vecList[:,j])
        L = TChol[j]
        idx = size(L,1)-1
        d[1:idx+1] .= diag(L,0)
        od[1:idx] .= diag(L,-1)
        i,d_Sum,od_Sum,dAsymp,odAsymp = zeros(Float64,5)
        if Convflag
            i = idx-jmp-k
            d_Sum= sum(@view d[idx-jmp-k+1:idx+1])
            od_Sum= sum(@view od[idx-jmp-k+1:idx])
            dAsymp = d_Sum/(jmp+k+1)
            odAsymp = od_Sum/(jmp+k)
            while i>0 && abs(d[i]-dAsymp)<Modtol && abs(od[i]-odAsymp)<Modtol
                d_Sum+=d[i]
                od_Sum+=od[i]
                i-=1
            end
            dAsymp = d_Sum/(idx-i+1)
            odAsymp = od_Sum/(idx-i)
        else
            i = max_iter-2
            d_Sum = d[max_iter-1]
            od_Sum = od[max_iter-1]
            while abs(d[i]-d[max_iter-1])<Modtol && abs(od[i]-od[max_iter-1])<Modtol
                d_Sum+=d[i]
                od_Sum+=od[i]
                i-=1
            end
            dAsymp = d_Sum/(max_iter-1-i)
            odAsymp = od_Sum/(max_iter-1-i)
        end
        d[i+1], d[i+2], od[i+1] = dAsymp, dAsymp, odAsymp
        ModChol[j] = Tridiagonal(od[1:i+1],d[1:i+2],0*od[1:i+1])
        sizelist[j] = i+2
    end
    dAsymp = 0
    odAsymp = 0
    for j=1:vecNbr
        dAsymp += (ModChol[j][sizelist[j],sizelist[j]]) + (ModChol[j][sizelist[j]-1,sizelist[j]-1])
        odAsymp += (ModChol[j][sizelist[j],sizelist[j]-1])
    end
    dAsymp/=(2*vecNbr)
    odAsymp/=vecNbr
    for j=1:vecNbr
        ModChol[j][sizelist[j],sizelist[j]] = dAsymp
        ModChol[j][sizelist[j]-1,sizelist[j]-1] = dAsymp
        ModChol[j][sizelist[j],sizelist[j]-1] = odAsymp
    end
    return TChol,ModChol
end

BiRel = (m,z,d,ℓ) -> 1/(-z+d^2-d^2*ℓ^2*(m/(1+ℓ^2*m)))
BiRef = (z,d,ℓ) -> (-z+d^2-ℓ^2+sqrt(z-(d+ℓ)^2)*sqrt(z-(d-ℓ)^2))/(2*z*ℓ^2)
function BiRec(z::Number,L,N::Integer;eps::Float64=1e-3)
    n = size(L,1)
    eps_min = min(N^(-1/6),eps)
    m0 = BiRef(z+1im*eps_min,L[n-1,n-1],L[n,n-1])
    for j = n-2:-1:1
            m0 = BiRel(m0,z+1im*eps_min,L[j,j],L[j+1,j])
    end
    return m0
end

function EstimSupp(ModChol)
    n = size(ModChol[1],1)
    dAsymp = ModChol[1][n,n]
    odAsymp = ModChol[1][n,n-1]
    γmin = (dAsymp-odAsymp)^2
    γplus = (dAsymp+odAsymp)^2
    return γmin, γplus
end

function EstimDensity(x,ModChol,N::Integer)
    len_x = length(x)
    dens = zeros(Float64,len_x)
    len = length(ModChol)
    for i=1:len_x
        for j=1:len
            dens[i]+=imag(BiRec(x[i],ModChol[j],N))/π
        end
    end
    return dens/len
end

function EstimSpike(TrueChol,N::Integer;δ::Float64=0.25,c::Float64=1.0,ThreshOut::Bool=false)
    len = length(TrueChol)
    Vec = zeros(Float64,len)
    sizelist = zeros(Int64,len)
    γplus = (TrueChol[1][end,end]+TrueChol[1][end,end-1])^2
    for i=1:len
        L = TrueChol[i]
        evals = eigvals(SymTridiagonal(L*L'))
        j = size(L,1)
        sizelist[i] = j
        while j>1 && evals[j]>γplus+c*N^(-δ)
            Vec[i] += 1
            j-=1
        end
    end
    freq = Dict{Int, Int}()
    for v in Vec
        freq[v] = get(freq, v, 0) + 1
    end
    Nbr = findmax(freq)[2]
    maxval,maxidx = findmin(sizelist)
    L = TrueChol[maxidx]
    evals = eigvals(SymTridiagonal(L*L'))
    if !ThreshOut
        return Nbr, evals[end-Nbr+1:end]
    else
        return Nbr, evals[end-Nbr+1:end], γplus+c*N^(-δ)
    end
end

function LanczosConv(mat,tol::Float64,k::Integer,jmp::Integer,max_iter::Integer;v=randn(size(mat,1)),orth::Bool=true)
    m, n = size(mat)
    Q = zeros(eltype(mat),n, max_iter)
    q = v / norm(v)
    Q[:, 1] .= q
    d = zeros(eltype(mat),max_iter)
    od = zeros(eltype(mat),max_iter-1)
    z = similar(q)
    dAvrg_old, odAvrg_old, dStd_old, odStd_old = zeros(Float64,4)
    idx = k
    i=1
    Convflag = false
    while i<=max_iter && !Convflag 
        z .= mat * q
        d[i] = dot(q, z)
        if orth
            Qview = @view Q[:, 1:i]
            z .-= Qview * (Qview' * z)
            z .-= Qview * (Qview' * z)
        else
            z .-= d[i] * q
            if i > 1
                z .-= (od[i-1]) * (@view Q[:, i-1])
            end
        end
        if i < max_iter
            od[i] = norm(z)
            if od[i]==0
                return SymTridiagonal(d[1:i], od[1:i-1])
            end
            q .= z / od[i]
            Q[:, i+1] .= q
            if i==idx 
                dAvrg = sum(@view d[i-k+1:i])/k
                odAvrg = sum(@view od[i-k+1:i])/k
                dStd = sqrt(sum(((@view d[i-k+1:i]).-dAvrg).^2)/(k-1))
                odStd = sqrt(sum(((@view od[i-k+1:i]).-odAvrg).^2)/(k-1))
                if dStd<tol && odStd<tol && dStd_old<tol && odStd_old<tol && abs(dAvrg-dAvrg_old)<tol && abs(odAvrg-odAvrg_old)<tol
                    z .= mat * q
                    d[i+1] = dot(q, z)
                    Convflag = true
                else
                    dAvrg_old, odAvrg_old, dStd_old, odStd_old = dAvrg, odAvrg, dStd, odStd
                    idx = idx+jmp
                end
            end
        end
        i+=1
    end
    i = min(i,max_iter)
    return Convflag, SymTridiagonal(d[1:i],od[1:i-1])
end

function JacobiList(mat,tol::Float64,k::Integer,jmp::Integer,max_iter::Integer,vecNbr::Integer;Modtol::Float64=tol,vecList=randn(size(mat,1),vecNbr))
    ModJac = Vector{Matrix{Float64}}(undef, vecNbr)
    TJac = Vector{Matrix{Float64}}(undef, vecNbr)
    sizelist = zeros(Int64,vecNbr)
    d = zeros(Float64,max_iter)
    od = zeros(Float64,max_iter)
    for j=1:vecNbr
        Convflag, TJac[j] = LanczosConv(mat,tol,k,jmp,max_iter,v=vecList[:,j])
        T = TJac[j]
        idx = size(T)-1
        d[1:idx+1] .= diag(T,0)
        od[1:idx] .= diag(T,-1)
        i,d_Sum,od_Sum,dAsymp,odAsymp = zeros(Float64,5)
        if Convflag
            i = idx-jmp-k
            d_Sum= sum(@view d[idx-jmp-k+1:idx+1])
            od_Sum= sum(@view od[idx-jmp-k+1:idx])
            dAsymp = d_Sum/(jmp+k+1)
            odAsymp = od_Sum/(jmp+k)
            while i>0 && abs(d[i]-dAsymp)<Modtol && abs(od[i]-odAsymp)<Modtol
                d_Sum+=d[i]
                od_Sum+=od[i]
                i-=1
            end
            dAsymp = d_Sum/(idx-i+1)
            odAsymp = od_Sum/(idx-i)
        else
            i = max_iter-2
            d_Sum = d[max_iter-1]
            od_Sum = od[max_iter-1]
            while abs(d[i]-d[max_iter-1])<Modtol && abs(od[i]-od[max_iter-1])<Modtol
                d_Sum+=d[i]
                od_Sum+=od[i]
                i-=1
            end
            dAsymp = d_Sum/(max_iter-1-i)
            odAsymp = od_Sum/(max_iter-1-i)
        end
        d[i+1], d[i+2], od[i+1] = dAsymp, dAsymp, odAsymp
        sizelist[j] = i+2
        ModJac[j] = Tridiagonal(od[1:i+1],d[1:i+2],od[1:i+1])
    end
    dAsymp = 0
    odAsymp = 0
    for j=1:vecNbr
        dAsymp += (ModJac[j][sizelist[j],sizelist[j]]) + (ModJac[j][sizelist[j]-1,sizelist[j]-1])
        odAsymp += (ModJac[j][sizelist[j],sizelist[j]-1])
    end
    dAsymp/=(2*vecNbr)
    odAsymp/=vecNbr
    for j=1:vecNbr
        ModJac[j][sizelist[j],sizelist[j]] = dAsymp
        ModJac[j][sizelist[j]-1,sizelist[j]-1] = dAsymp
        ModJac[j][sizelist[j],sizelist[j]-1] = odAsymp
    end
    return TJac,ModJac
end

TriRel = (m,z,d,ℓ) -> 1/(d-z-ℓ^2*m)
TriRef = (z,d,ℓ) -> (d-z+sqrt(z-d-2*ℓ)*sqrt(z-d+2*ℓ))/(2*ℓ^2)
function TriRec(z::Number,T,N::Integer;eps::Float64=1e-3)
    n = size(T,1)
    eps_min = min(N^(-1/6),eps)
    m0 = TriRef(z+1im*eps_min,T[n-1,n-1],T[n,n-1])
    for j = n-2:-1:1
            m0 = TriRel(m0,z+1im*eps_min,T[j,j],T[j+1,j])
    end
    return m0
end

function EstimSupp_Tri(ModJac)
    n = size(ModJac[1],1)
    dAsymp = ModJac[1][n,n]
    odAsymp = ModJac[1][n,n-1]
    γmin = dAsymp-2*odAsymp
    γplus = dAsymp+2*odAsymp
    return γmin, γplus
end

function EstimDensity_Tri(x::Number,ModJac,N::Integer)
    len_x = length(x)
    dens = zeros(Float64,len_x)
    len = length(ModJac)
    for i=1:len_x
        for j=1:len
            dens[i]+=imag(TriRec(x[i],ModJac[j],N))/π
        end
    end
    return  dens/len
end

function EstimSpike_Tri(TrueJac,N::Integer;δ::Float64=0.25,c::Float64=1.0,ThreshOut::Bool=false)
    len = length(TrueJac)
    Vec = zeros(Float64,len)
    sizelist = zeros(Int64,len)
    γplus = TrueJac[1][end,end]+2*TrueJac[1][end,end-1]
    for i=1:len
        T = TrueJac[i]
        evals = eigvals(T)
        j = size(T,1)
        sizelist[i] = j
        while j>1 && evals[j]>γplus+c*N^(-δ)
            Vec[i] += 1
            j-=1
        end
    end
    freq = Dict{Int, Int}()
    for v in Vec
        freq[v] = get(freq, v, 0) + 1
    end
    Nbr = findmax(freq)[2]
    maxval,maxidx = findmin(sizelist)
    T = TrueJac[maxidx]
    evals = eigvals(T)
    if !ThreshOut
        return Nbr, evals[end-Nbr+1:end]
    else
        return Nbr, evals[end-Nbr+1:end], γplus+c*N^(-δ)
    end
end