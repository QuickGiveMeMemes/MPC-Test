from cartpole_mod import CartPoleEnv
import casadi as ca
import numpy as np

# Euler / direct shooting MPC

# Constat
POLE_LEN = 0.5
FORCE = 10

g = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
TOTAL_MASS = POLE_MASS + CART_MASS
POLEMASS_LENGTH = POLE_MASS * POLE_LEN

opti = ca.Opti()

env = CartPoleEnv(render_mode="human")



env.reset()

def gen_dynamics(dt):
    x = ca.SX.sym("x", 4)
    u = ca.SX.sym("u")

    x_c, d_x_c, theta, theta_dot = ca.vertsplit(x)

    # x, x_dot, theta, theta_dot = self.state
    costheta = ca.cos(theta)
    sintheta = ca.sin(theta)

 
    temp = (
        u + POLE_LEN * (theta_dot * theta_dot) * sintheta
    ) / TOTAL_MASS
    thetaacc = (g * sintheta - costheta * temp) / (
        POLE_LEN
        * (4.0 / 3.0 - POLE_MASS * (costheta * costheta) / TOTAL_MASS)
    )
    xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS

    x_c_next = x_c + dt * d_x_c
    d_x_c_next = d_x_c + dt * xacc
    theta_next = theta + dt * theta_dot
    theta_dot_next = theta_dot + dt * thetaacc
    
    x_next = ca.vertcat(x_c_next, d_x_c_next, theta_next, theta_dot_next)
    
    dynam_f = ca.Function("dynamics", [x, u], [x_next])
    return dynam_f

def setup_mpc(x0):
    x = x0
    dt = 0.02 * 2
    x_threshold = 2.4
    n = 20
    u = opti.variable(n) # 1 sec lookahead

    apply_dynamics = gen_dynamics(dt)

    # opti.subject_to(u ** 2 - 100 < 1e-5)

    J = 0

    for i in range(n):
        x = apply_dynamics(x, u[i])

        J += x[2] ** 2 + x[0]**2 
        
        # J += x[3]**2
        opti.subject_to(opti.bounded(-x_threshold, x[0], x_threshold))
        opti.subject_to(opti.bounded(-10, u[i], 10))
        
    opti.minimize(J)

    return u
    

x0 = opti.parameter(4) # init state [cart pos, cart vel, pole angle, pole angle vel]
u = setup_mpc(x0)

ipopt_settings = {
    "ipopt.print_frequency_iter": 25,
    "ipopt.print_level": 0,
    "print_time": 0,
    "ipopt.sb": "no",
    "ipopt.max_iter": 500,
    "detect_simple_bounds": True,
    "ipopt.linear_solver": "ma97",
    "ipopt.mu_strategy": "adaptive",
    "ipopt.nlp_scaling_method": "gradient-based",
    "ipopt.bound_relax_factor": 1e-4,
    # "ipopt.hessian_approximation": "exact",
    "ipopt.tol": 1e-4,
    "ipopt.hessian_approximation": "limited-memory",
    "ipopt.limited_memory_max_history": 10,
    "ipopt.limited_memory_update_type": "bfgs",
    "ipopt.derivative_test": "none",
}
opti.solver("ipopt", ipopt_settings)

i = 0
opti.set_value(x0, ca.vertcat(0, 0, ca.pi, 0))

while True:
    print(f"Solving with {opti.nx} variables.")
    # try:
    sol = opti.solve()
    stats = sol.stats()
    print(f"Solve iteration succeeded in {stats['iter_count']} iterations")
    # except:
    #     sol = opti.debug
    #     stats = sol.stats()
    #     print(f"Solve iteration failed after {stats['iter_count']} iteration...")


    force = sol.value(u)[0]
    # force = sol.value(u)
    print(force)

    # for f in force:
    #     observation, reward, terminated, truncated, info = env.step(1 if f > 0 else 0)


    observation, reward, terminated, truncated, info = env.step(float(force))

    opti.set_value(x0, ca.vertcat(*observation))
    
    if terminated: break

    # i += 1


env.close()