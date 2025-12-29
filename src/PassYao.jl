function PassYao(evals::AbstractVector{Float64},d::Float64,N::Int;SampleNbr::Int=100)
    M = convert(Int64,ceil(N/d))
    p = length(evals)
    i = 1
    SigmaTol = 1e-3
    Diff = zeros(Float64,SampleNbr)
    sqrtD_vec = zeros(Float64, N)
    ρ = zeros(Float64, p)
    SpikeFlag = false
    while i≤p-2 && !SpikeFlag
        SigmaFlag = false
        old_σ2 = (1/(N-i))*(sum(evals[1:p-i]))
        new_σ2 = 0.0
        while !SigmaFlag
            for j=1:i
                Δ = (evals[p-j+1]+old_σ2-old_σ2*(N-i)/M)^2-4*evals[p-j+1]*old_σ2
                if Δ<=0
                    SigmaFlag = true
                end
                if !SigmaFlag
                    ρ[j] = ((evals[p-j+1]+old_σ2-old_σ2*(N-i)/M)+sqrt(Δ))/2
                end
            end
            new_σ2 = (1/(N-i))*(sum(evals[1:p-i])+sum(evals[p:-1:p-i+1]-ρ[1:i]))
            if abs(new_σ2-old_σ2)<SigmaTol
                SigmaFlag = true
            end
            if !SigmaFlag
                old_σ2 = new_σ2
            end
        end
        σ = sqrt(old_σ2)
        sqrtD_vec .= σ
        X = Matrix{Float64}(undef, N, M)

        for j in 1:SampleNbr
            randn!(X)
            if p == N
                W = Symmetric(@views (sqrtD_vec .* X) * (sqrtD_vec .* X)' / M)
            else
                W = Symmetric(X' * Diagonal(sqrtD_vec.^2) * X / M)
            end
            evals_W = eigvals(W)
            Diff[j] = evals_W[end] - evals_W[end-1]
        end

        sorted_Diff = sort(Diff, rev=true)
        check = convert(Int64,ceil(SampleNbr*0.02))
        thresh = (sorted_Diff[check]+sorted_Diff[check+1])/2
        if evals[p-i+1]-evals[p-i]>thresh || evals[p-i]-evals[p-i-1]>thresh
            i+=1
        else
            SpikeFlag=true
        end
    end
    return i-1
end