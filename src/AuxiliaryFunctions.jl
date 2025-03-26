function Bisection(f::Function,x::Float64,y::Float64;max_iter::Integer=500,tol::Float64=1e-12)
    fx = f(x)
    fy = f(y)
    if fx*fy>0
        return nothing
    end
    if fx ≈ 0
        return x
    elseif fy ≈ 0
        return y
    end
    iter = 0
    while(abs(x-y)>tol && iter<=max_iter)
        z = (x+y)/2
        fz = f(z)
        if fz ≈ 0
            return z
        elseif fx*fz < 0
           y = z
           fy = fz
        else
            x = z
            fx = fz
        end
        iter+=1
    end
    z = (x+y)/2
    fz = f(z)
    if abs(fz)<1
        return z
    else
        return nothing
    end
end

function LanczosTri(mat; k::Integer=size(mat,1), v=randn(size(mat,1)),opt::Integer=1)
    m, n = size(mat)
    Q = zeros(eltype(mat),n, k)
    q = v / norm(v)
    Q[:, 1] .= q
    d = zeros(eltype(mat),k)
    od = zeros(eltype(mat),k-1)
    z = similar(q)
    for i = 1:k
        z .= mat * q
        d[i] = dot(q, z)
        if opt==1
            Qview = @view Q[:, 1:i]
            z .-= Qview * (Qview' * z)
            z .-= Qview * (Qview' * z)
        elseif opt==2
            Qview = @view Q[:, 1:i]
            z .-= Qview * (Qview' * z)
        else
            z .-= d[i] * q
            if i > 1
                z .-= (od[i-1]) * (@view Q[:, i-1])
            end
        end
        if i < k
            od[i] = norm(z)
            if od[i]==0
                return SymTridiagonal(d[1:i], od[1:i-1]),@view Q[:,1:i]
            end
            q .= z / od[i]
            Q[:, i+1] .= q
        end
    end
    return SymTridiagonal(d, od),Q
end

function Mab(x,a,b)
    return (a+b)/2+(b-a)*x/2
end

function invMab(x,a,b)
    return (2*x)/(b-a)-(a+b)/(b-a)
end

function Legendre(N::Integer)
    A = SymTridiagonal(fill(0.0,N),[1/sqrt(4-i^(-2)) for i=1:N-1])
    Eig = eigen(A)
    evals = Eig.values
    evects = Eig.vectors
    nodes = evals
    weights = 2*(@view evects[1,1:N]).^2
    return nodes,weights
end
                                                                                                                                        
function weight_scaling(w,a,b)
    return ((b-a)/2)*w
end

function QuadInt(h,a,b,Legendre_nodes,Legendre_weights)
    x = Mab.(Legendre_nodes,a,b)
    return sum(h.(x).*weight_scaling(Legendre_weights,a,b))
end