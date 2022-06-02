# manipulator differential equations
import numpy as np
import torch
from scipy.integrate import odeint
import random

ndata = 10

# model parameters
M = 0.05 #0.05
B = 1 #1
Cm = 10 #10
H = 0.5 #0.5
N = 10 #10
D = 1 #1
# Active / coupling component
G0 = 1
G1 = 1 / D
G2 = 1 / M
u_max = 1e4

def u_exp(t):
    return 0.01*np.sin(2 * np.pi * t)

def xdes(t):
    xd = 0 #np.sin(2 * np.pi * t)
    xddot = 0 #(2 * np.pi) * np.cos(2 * np.pi * t)
    xdDot = 0 #-(2 * np.pi) ** 2 * np.sin(2 * np.pi * t)
    return xd, xddot, xdDot

def manip(x, t, C, gps, gps_safe, beta,err_des):

    omega_f = 1e2
    zeta = 1
    x1 = np.atleast_2d(x[0])
    x2 = np.atleast_2d(x[1])
    x3 = np.atleast_2d(x[2])

    w1 = np.atleast_2d(x[0])
    w2 = np.atleast_2d(x[:2])
    w3 = np.atleast_2d(x[:3])

    if not gps:
        dxdt = np.zeros(3)
        u_sat = u_exp(t)
    else:

        z11 = x[3]
        z12 = x[4]
        z21 = x[5]
        z22 = x[6]
        xi1 = x[7]
        xi2 = x[8]
        xi3 = 0

        x1d, x1ddot, x1dDot = xdes(t)

        # taking the mean is only required in the fully Bayesian setting. Calling torch.mean does nothing otherwise
        muF0 = torch.mean(gps[0](torch.as_tensor(w1).float()).mean).detach().numpy()
        muF1 = torch.mean(gps[1](torch.as_tensor(w2).float()).mean).detach().numpy()
        muF2 = torch.mean(gps[2](torch.as_tensor(w3).float()).mean).detach().numpy()

        if not gps_safe:
            C1 = C[0]
            C2 = C[1]
            C3 = C[2]
        else:
            # taking the mean is only required in the fully Bayesian setting. Calling torch.mean does nothing otherwise
            stddev1 = torch.mean(gps_safe[0](torch.as_tensor(w1).float()).stddev).detach().numpy()
            stddev2 = torch.mean(gps_safe[1](torch.as_tensor(w2).float()).stddev).detach().numpy()
            stddev3 = torch.mean(gps_safe[2](torch.as_tensor(w3).float()).stddev).detach().numpy()

            sum_sig = np.sqrt(beta)/err_des *np.sqrt(stddev1**2 + stddev2**2 + stddev3**2)

            C1 = sum_sig
            C2 = sum_sig
            C3 = sum_sig

            # C1 = stddev1 * np.sqrt(beta)/err_des
            # C2 = stddev2 * np.sqrt(beta) / err_des
            # C3 = stddev3 * np.sqrt(beta) / err_des

        x2d = z11
        x2ddot = omega_f * z12
        x3d = z21
        x3ddot = omega_f * z22

        # Pseudocontrol signals and derivatives
        alpha1 = 1 / G0 * (-C1 * (x1 - x1d) + x1ddot - muF0)
        alpha2 = 1 / G1 * (-C2 * (x2 - x2d) + x2ddot - muF1 - G0 * ((x1 - x1d) - xi1))
        u = 1 / G2 * (-C3 * (x3 - x3d) + x3ddot - muF2 - G1 * ((x2 - x2d) - xi2))

        xi1dot = -C1 * xi1 + G0 * (x2d - alpha1) + G0 * xi2
        xi2dot = -C2 * xi2 + G1 * (x3d - alpha2) + G1 * xi3
        z11dot = omega_f * z12
        z12dot = -2 * zeta * omega_f * z12 - omega_f * (z11 - alpha1)
        z21dot = omega_f * z22
        z22dot = -2 * zeta * omega_f * z22 - omega_f * (z21 - alpha2)

        u_unsat = 1/G2*(-C3*(x3-x3d) + x3ddot - muF2 -G1*((x2-x2d)-xi2))
        # u_sat = u_max*(2/(1+np.exp(-2*u_unsat/u_max))-1)
        u_sat = u_unsat

        dxdt = np.zeros(9)
        dxdt[3:] = [z11dot,z12dot,z21dot,z22dot,xi1dot,xi2dot]

    # Passive component
    F0 = 0
    F1 = (-N * np.sin(x[0]) - B * x[1]) / D
    F2 = (-Cm * x[1] - H * x[2]) / M

    dxdt[:3] = [F0 + G0*x2, F1 + G1*x3, F2 + G2*u_sat]
    return dxdt

def get_data():

    C = np.ones(3)
    t = np.linspace(0, 10, 201)
    x0 = [0.0, 0.0, 0.0]

    sol = odeint(manip, x0, t, args=(C, [], [], [], []))
    perm = list(np.random.permutation(list(range(sol.shape[0]))))

    X_data = np.stack(sol[perm[0:ndata]], axis=0)
    tau_data = np.stack(t[perm[0:ndata]])
    Y_data = []
    for i in list(range(ndata)):
        funknown = np.asarray(manip(X_data[i], tau_data[i], C, [], [], [], []))
        fknown = np.asarray([G0 *X_data[i,1], G1*X_data[i,2], G2*u_exp(tau_data[i])])
        # add noise
        dy = 0.01 * np.random.random(funknown.shape)
        noise = np.random.normal(0, dy)
        funknown += noise
        Y_data.append(funknown - fknown)
    Y_data = np.asarray(Y_data)
    return X_data, Y_data

def get_xd(x, t):

    z11 = x[3]
    z21 = x[5]

    x1d, _, _ = xdes(t)
    x2d = z11
    x3d = z21

    xd = [x1d,x2d,x3d]

    return xd