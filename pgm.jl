using LinearAlgebra, Distributions, StaticArrays

#parameters for the liquidation problem (AAPL)
beta = 1.03*10^-5
gamma = 7.27*10^-6
sigw = sqrt(0.022)
delta = beta - gamma/2
epsilon = 10^-8
phi = 5*10^-6

#inital state
E_x00 = 193
E_x01 = 500
var_x00 = 57
var_x01 = 1
E_x00_2 = var_x00 + E_x00^2
E_x01_2 = var_x01 + E_x01^2


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

function gradient_descent(K, oracle; eta = 0.05)
    @assert(size(K) == (2, 10))
    nablaCK = zeros(2, 10)
    m = 200
    for t = 1:10
        nablaCK[:, t] = sum([oracle(K, t) for i = 1:m]) / Float64(m)
    end

    return K - eta * nablaCK, norm(nablaCK)
end

function approx_eval(K; N = 10000)
    
end

function gd()
    K = Matrix{Float64}(undef, 2, 10)
   
    K .= -0.2*ones(2, 10)
    for iter = 1:100
        # K, gnorm= gradient_descent(K, mygrad)
        K, gnorm= gradient_descent(K, mygrad)
        
        @info "iter = $iter, g = $gnorm"
    end
end

@time std(i for i in 1:100000)