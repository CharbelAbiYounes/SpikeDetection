function PassYao(evals,d::Float64,N::Integer;SampleNbr::Integer=500)
    M = convert(Int64,ceil(N/d))
    p = length(evals)
    i=1
    SigmaTol = 1e-3
    Diff = zeros(Float64,SampleNbr)
    sqrtD = zeros(Float64,N,N)
    SpikeFlag = false
    while i≤p-2 && !SpikeFlag
        SigmaFlag = false
        old_σ2 = (1/(N-i))*(sum(evals[1:p-i]))
        new_σ2 = 0
        ρ = zeros(Float64,i)
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
            new_σ2 = (1/(N-i))*(sum(evals[1:p-i])+sum(evals[p:-1:p-i+1]-ρ))
            if abs(new_σ2-old_σ2)<SigmaTol
                SigmaFlag = true
            end
            if !SigmaFlag
                old_σ2 = new_σ2
            end
        end
        σ2 = old_σ2
        σ = sqrt(σ2)
        sqrtD = Diagonal(σ*ones(N))
        if p==N
            for j=1:SampleNbr
                X = randn(N,M)
                W = sqrtD*X*X'*sqrtD/M |>Symmetric
                nullevals = eigvals(W)
                Diff[j] = nullevals[end]-nullevals[end-1]
            end
        else
            for j=1:SampleNbr
                X = randn(N,M)
                W = X'*sqrtD*sqrtD*X/M |>Symmetric
                nullevals = eigvals(W)
                Diff[j] = nullevals[end]-nullevals[end-1]
            end
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