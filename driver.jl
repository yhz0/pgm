using Plots
include("pgm.jl")

P_star, K_star = solve_exact()
# C_star = approx_eval(K_star; N = 10000000)
C_star = 0.25942212137305265

# One sided
K, rel_error = gd(mygrad; max_iter=300, true_cost = C_star)

# Two sided
K2, rel_error2 = gd(mygrad_twoside; max_iter=150, true_cost = C_star)

# # Plot results
plot((1:300)*2000, rel_error, yaxis=:log, label="One Side")
plot!((1:150)*4000, rel_error2, yaxis=:log, label="Two Sides")
xlabel!("Number of Zeroth Order Oracle Calls")
ylabel!("Relative Error")


K3, rel_error3 = bcd(mygrad; true_cost = C_star, max_iter=300)
plot!((1:3000)*200, rel_error3, yaxis=:log)

xlims!(0.0, 3e5)

K4, rel_error4 = bcd(mygrad_twoside; true_cost = C_star, max_iter=150)
plot!((1:1500)*400, rel_error4, yaxis=:log)
