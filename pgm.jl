using LinearAlgebra, Distributions, StaticArrays

#parameters for the liquidation problem (AAPL)
const beta = 1.03*10^-5
const gamma = 7.27*10^-6
const sigw = sqrt(0.022)
const delta = beta - gamma/2
const epsilon = 10^-8
const phi = 5*10^-6

#inital state
const E_x00 = 193
const E_x01 = 500
const var_x00 = 57
const var_x01 = 1
const E_x00_2 = var_x00 + E_x00^2
const E_x01_2 = var_x01 + E_x01^2

function solve_exact()
    A = I(2)
    B = [-gamma, -1]
    Qt = Diagonal([epsilon, phi*sigw^2])
    QT = Diagonal([epsilon, delta + phi*sigw^2])
    Rt = delta

    # Find P
    P = Array{Float64}(undef, 11, 2, 2)
    P[11, :, :] .= QT
    for t = 10:-1:1
        Pt_plus = P[t+1, :, :]
        fact = inv(B'*Pt_plus*B + Rt)
        P[t, :, :] .= Qt + A' * Pt_plus * A - A' *Pt_plus*B*fact*B'*Pt_plus*A
    end

    K_star = Array{Float64}(undef, 2, 10)
    for t = 1:10
        K_star[:, t] = inv(B' * P[t+1, :, :] * B + Rt) * B' * P[t+1, :, :]* A
    end

    return P, K_star
end

function simulate(K)::Float64
    @assert(size(K) == (2, 10))

    x00 = randn() * sqrt(var_x00) + E_x00
    x01 = randn() * sqrt(var_x01) + E_x01
    x = @MArray [x00, x01]

    total_cost = 0.0

    for t = 1:10
        k = K[:, t]
        u = -dot(k, x)
        
        # cost
        stage_cost = epsilon*x[1]^2 + phi*sigw^2*x[2]^2
        action_cost = delta*u^2
        total_cost += (stage_cost + action_cost)

        # transition
        x[1] += randn()*sigw
        x[2] -= u
        # @info "time = $t, state = $x, action = $u"
    end

    # Terminal stage cost
    term_stage_cost = epsilon*x[1]^2 + (phi*sigw^2+delta)*x[2]^2
    total_cost += term_stage_cost
    # @info "total_cost = $total_cost"

    return total_cost
end

# Finite difference
function mygrad(K_in::Matrix{Float64}, t::Int; r::Float64 = 0.6)::Vector{Float64}
    z::Float64 = 2.0 * pi * rand()
    u = [cos(z), sin(z)] * r
    
    D = 2.0
    
    K_in[:, t] += u
    v = D / r^2 * simulate(K_in) * u

    # We changed K_in. Changing it back now to avoid issues.
    K_in[:, t] -= u
        
    return v
end

# Two-sided difference
function mygrad_twoside(K_in::Matrix{Float64}, t::Int; r::Float64 = 0.6)::Vector{Float64}
    z::Float64 = 2.0 * pi * rand()
    u = [cos(z), sin(z)] * r
    
    D = 2.0
    
    K_in[:, t] += u
    v1 = simulate(K_in) * u
    K_in[:, t] -= u
        
    K_in[:, t] -= u
    v2 = simulate(K_in) * (-u)
    K_in[:, t] += u

    v = D / (2 * r^2) * (v1 + v2)
    return v
end

function estimate_gradient(K, oracle)
    @assert(size(K) == (2, 10))
    nablaCK = zeros(2, 10)
    m = 200
    for t = 1:10
        nablaCK[:, t] = sum([oracle(K, t) for i = 1:m]) / Float64(m)
    end

    return nablaCK, norm(nablaCK)
end

function approx_eval(K; N = 10000)
    sum_x = 0.0
    sum_x2 = 0.0

    for i = 1:N
        v = simulate(K)
        sum_x += v
        sum_x2 += v^2
    end

    stddev_hat = sqrt(sum_x2 / (N-1))
    mean_hat = sum_x / N

    return mean_hat, 1.96 * stddev_hat / sqrt(N)
end

function gd(grad_oracle;
    max_iter::Int = 100,
    true_cost::Union{Nothing, Float64} = nothing,
    eta = 0.05
    )
    rel_error = Float64[]
    K = Matrix{Float64}(undef, 2, 10)
    K .= -0.2*ones(2, 10)
    for iter = 1:max_iter
        dK, gnorm= estimate_gradient(K, grad_oracle)
        K -= (eta * dK)
        
        # Evaluate if needed
        if true_cost !== nothing
            v = approx_eval(K)[1]
            push!(rel_error, (v - true_cost) / true_cost)
        end
        @info "iter = $iter, g = $gnorm"
    end
    return K, rel_error
end

# ??? How to exactly evaluate?
# function exact_eval(K)
#     x00 = Normal(E_x00, sqrt(var_x00))
#     x01 = Normal(E_x01, sqrt(var_x01))
#     noise_dist = Normal(0, sigw)
#     x_dist = product_distribution([x00, x01])
    
# end

# block_grad_oracle is mygrad or mygrad_twoside
function bcd(block_grad_oracle;
    max_iter::Int = 100,
    true_cost::Union{Nothing, Float64} = nothing,
    eta = 0.05
    )
    rel_error = Float64[]
    K = Matrix{Float64}(undef, 2, 10)
    K .= -0.2*ones(2, 10)


    for iter = 1:max_iter
        for t = 10:-1:1
            nablaKt = zeros(2)
            for rep = 1:200
                nablaKt .= nablaKt + block_grad_oracle(K, t) / 200
            end
            K[:, t] .= K[:, t] - eta * nablaKt
            if true_cost !== nothing
                v = approx_eval(K)[1]
                push!(rel_error, (v - true_cost) / true_cost)
            end
        end
        @info "iter=$iter"
    end

    return K, rel_error
end

