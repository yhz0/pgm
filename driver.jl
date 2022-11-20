using Plots
include("pgm.jl")

K_star = solve_exact()
# C_star = approx_eval(K_star; N = 10000000)
C_star = 0.25942212137305265

# One sided
K, rel_error = gd(mygrad; max_iter=300, true_cost = C_star)

# Two sided
K2, rel_error2 = gd(mygrad_twoside; max_iter=150, true_cost = C_star)

# # Plot results
plot(rel_error, yaxis=:log, label="One Side")
plot!(1:2:300, rel_error2, yaxis=:log, label="Two Sides")
xlabel!("Number of Samples")
ylabel!("Relative Error")
