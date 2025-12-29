function Bisection(f::Function, x::Float64, y::Float64; 
    max_iter::Int=500, 
    tol::Float64=1e-12
)
    fx = f(x)
    fy = f(y)
    if fx*fy>0
        return nothing
    end
    if abs(fx)<tol
        return x
    elseif abs(fy)<tol
        return y
    end
    iter = 0
    for iter in 1:max_iter
        z = 0.5*(x+y)
        fz = f(z)
        if abs(fz)<tol
            return z
        end
        if fx*fz<0
            y = z
            fy = fz
        else
            x = z
            fx = fz
        end
    end
    return 0.5*(x+y)
end

function LanczosTri(mat::AbstractMatrix{Float64}; 
    k::Int=size(mat,1), 
    v::AbstractVector{Float64}=randn(size(mat,1)), 
    opt::Int=2
)
    n = size(mat,1)
    Q = Matrix{Float64}(undef, n, k)
    q = copy(v) / norm(v)
    Q[:, 1] = q
    proj = zeros(Float64, k)

    d = zeros(Float64, k)
    od = zeros(Float64, k-1)
    z = similar(q)

    for i in 1:k
        mul!(z, mat, q)
        d[i] = dot(q, z)

        if opt == 2
            # reorthogonalization x2
            Qi = @view Q[:,1:i]
            pv = @view proj[1:i]
            mul!(pv, Qi', z)
            mul!(z, Qi, pv, -1.0, 1.0)
            mul!(pv, Qi', z)
            mul!(z, Qi, pv, -1.0, 1.0)
        elseif opt == 1
            # reorthogonalization
            Qi = @view Q[:,1:i]
            pv = @view proj[1:i]
            mul!(pv, Qi', z)
            mul!(z, Qi, pv, -1.0, 1.0)
        else
            z .-= d[i] * q
            if i > 1
                z .-= od[i-1] * Q[:, i-1]
            end
        end

        if i < k
            od[i] = norm(z)
            if od[i] == 0
                return SymTridiagonal(d[1:i], od[1:i-1])
            end
            q .= z / od[i]
            Q[:, i+1] .= q
        end
    end

    return SymTridiagonal(d, od)
end

function Cholesky(T::SymTridiagonal)
    n = length(T.dv)
    d = copy(T.dv)
    e = copy(T.ev) 
    for k in 1:n-1
        r = sqrt(d[k])
        e[k] /= r
        d[k+1] -= e[k]^2
        d[k] = r
    end
    d[n] = sqrt(d[n])
    return Tridiagonal(e, d, zeros(n-1))
end

@inline function Mab(x::Float64, a::Float64, b::Float64)
    return ((b-a)/2)*x+(a+b)/2
end

@inline function invMab(x::Float64, a::Float64, b::Float64)
    return (2*x)/(b-a)-(a+b)/(b-a)
end

function LegQuad(N::Integer)
    b = zeros(Float64, N-1)
    @inbounds for i in 1:N-1
        b[i] = 1.0 / sqrt(4.0 - 1.0/(i^2))
    end
    A = SymTridiagonal(zeros(Float64, N), b)
    Eig = eigen(A)
    nodes = Eig.values
    v1 = @view Eig.vectors[1, :]
    weights = 2.0*v1.^2
    return nodes,weights
end

function LegQuadInt(h::Function, a::Float64, b::Float64, nodes::AbstractVector{Float64}, weights::AbstractVector{Float64})
    n = length(nodes)
    s = zero(Float64)
    scale = (b - a)/2.0
    @inbounds for i in 1:n
        xi = Mab(nodes[i], a, b)
        s += h(xi) * (weights[i] * scale)
    end
    return s
end