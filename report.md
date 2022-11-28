# ISE 599 Reinforcement Learning Project Report

Based on paper _Policy Gradient Methods for the Noisy Linear Quadratic Regulator over a Finite Horizon_ by Ben Hambly, Renyuan Xu and Huining Yang.

## Setup

We will briefly summarize the paper in this section.

We would like to solve a finite horizon linear quadratic regulator problem.

$$
\begin{align}
\min_{u_t, t = 1 \dots T} & E[\sum_{t=0}^{T-1} (x_t^\top Q x_t + u_t^\top R_t u_t) + x_T^\top Q_T x_T] \\
\text{subject to } & x_{t+1} = A x_t + B u_t + w_t, x_0 \sim \mathcal{D}
\end{align}
$$

$x_t$ is the state of the system at time $t$. $u_t$ is the control applied at time $t$.

The system parameters are as follows:
1. $A, B$ are deterministic system transition matrices of appropriate dimension.
2. $Q_t, R_t$ are deterministic cost matrices. They are assumed to be positive definite.
3. $w_t$ are zero-mean IID random noises with finite variance.

