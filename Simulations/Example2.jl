using LinearAlgebra, Plots, LaTeXStrings

function dMP(x,d,ℓ)
    dp = (1+sqrt(d))^2
    dm = (1-sqrt(d))^2
    if ℓ>1+sqrt(d) && x==ℓ+ℓ*d/(ℓ-1)
        return ((ℓ-1)^2-d)/((ℓ-1)*(ℓ-1+d))
    end
    if dm<=x<=dp
        return (ℓ*sqrt(dp-x)*sqrt(x-dm))/(2*pi*x*(ℓ^2+ℓ*(d-1-x)+x))
    end
    return 0
end

N = 10000
d = 0.5
M = convert(Int64,ceil(N/d))
X = randn(N,M)
σ = 1
sqrtΣ = Diagonal(σ*ones(N))
ℓvec = [1,1.4,1+sqrt(d),2.2]
len = length(ℓvec)
dm = (1-sqrt(d))^2
dp = (1+sqrt(d))^2
x = dm-0.1:0.01:dp+0.1
len_x = length(x)
for i=1:len
    ℓ = ℓvec[i]
    sqrtΣ[1,1] = sqrt(ℓ)
    W = sqrtΣ*(1/M*X*X')*sqrtΣ'|>Symmetric
    Eig = eigen(W)
    evals = Eig.values
    evects = Eig.vectors
    w = abs2.(evects[1,:])
    if i==4
        w[end] = w[end]/20
    end
    x0 = ℓ+ℓ*d/(ℓ-1)
    y_dMP = zeros(Float64,len_x)
    for j=1:len_x
        xj = x[j]
        y_dMP[j] = dMP(xj,d,ℓ)
    end
    p = histogram(evals,bins=evals[1]-0.2:0.05:evals[end]+0.2,weights=20*w,label="VESD", legendfontsize=12, framestyle=:box, xtickfontsize=12, ytickfontsize=12)
    p = plot!(x,y_dMP,linecolor=:red,linewidth=3,label="VASD")
    if ℓ>1+sqrt(d)
        p = plot!([x0,x0],[0,dMP(x0,d,ℓ)],linecolor=:red,linewidth=3,label="")
    end
    savefig(p,"Density"*string(i)*".pdf")
end