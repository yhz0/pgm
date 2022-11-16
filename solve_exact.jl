using LinearAlgebra

beta = 1.03*10^-5
gamma = 7.27*10^-6
sigw = sqrt(0.022)
delta = beta - gamma/2
epsilon = 10^-8
phi = 5*10^-6

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

include("pgm.jl")
@time approx_eval(K_star; N = 100000)
