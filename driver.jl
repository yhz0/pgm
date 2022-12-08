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
plot(1:300, rel_error, yaxis=:log, label="One Side")
plot!(collect(1:2:300), rel_error2, yaxis=:log, label="Two Sides")
xlabel!("Computational Effort (GD Iterations)")
ylabel!("Relative Error")
savefig("twoside.png")

# Block coordinate descent
K3, rel_error3 = bcd(mygrad; true_cost = C_star, max_iter=300)
plot(1:300, rel_error, yaxis=:log, label="One Side")
plot!(collect(0.1:0.1:300), rel_error3, yaxis=:log, label="Cyclic BCD")
xlabel!("Computational Effort (GD Iterations)")
ylabel!("Relative Error")
savefig("bcd.png")

# two sided + bcd
# K4, rel_error4 = bcd(mygrad_twoside; true_cost = C_star, max_iter=150)
# plot!((1:1500)*400, rel_error4, yaxis=:log)
