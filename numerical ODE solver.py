import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from time import time

l   = 1.            # (m) Length of pendulum
g   = 9.81          # (m/s^2)
m   = 0.001         # (kg) Mass of the bob
rho = 1.225         # (kg/m^3) Density of air
r   = 0.05          # (m) Radius of sphere
A   = np.pi * r**2  # (m^2) Cross-secional area
c_d = 0.47          # (.) Drag coefficient of a sphere

alpha = c_d * rho * A / 2.   # Initially assume air-resitance is zero
# alpha = 0
T_final = 10000.   # (s) Final time
N = 60000

# (.) Number of time-steps
dt = T_final / (N)

yp_rk4 = 0
y_rk4 = 1
y_history_rk4 = np.zeros(N+1)
y_history_rk4[0] = y_rk4

yp_fe_1 = 0
y_fe_1 = 1
y_history_fe_1 = np.zeros(N+1)
y_history_fe_1[0] = y_fe_1

yp_fe_2 = 0
y_fe_2 = 1
y_history_fe_2 = np.zeros(N+1)
y_history_fe_2[0] = y_fe_2

x = np.linspace(0, T_final, N+1)

def dypdt(y, yp):
    return (-m*g*np.sin(y)/l-alpha*l*yp*abs(l*yp))/m
    # return -yp-y


def y_exact(x):
    return 2/3*np.sqrt(3)*np.exp(-x/2)*np.sin(np.sqrt(3)/2*x+np.pi/3)


def rk4(dypdt):
    d1 = dt*yp_rk4
    dp1 = dt*dypdt(y_rk4, yp_rk4)
    d2 = dt*(yp_rk4+0.5*dp1)
    dp2 = dt*dypdt(y_rk4+0.5*d1, yp_rk4+0.5*dp1)\
        # (yp+0.5*dp1+y+0.5*d1)
    d3 = dt*(yp_rk4+0.5*dp2)
    dp3 = dt*dypdt(y_rk4+0.5*d2, yp_rk4+0.5*dp2)\
        # (yp+0.5*dp2+y+0.5*d2)
    d4 = dt*(yp_rk4+dp3)
    dp4 = dt*dypdt(y_rk4+d3, yp_rk4+dp3)\
        # (yp+dp3+y+d3)
    dy = 1/6*(d1+2*d2+2*d3+d4)
    dyp = 1/6*(dp1+2*dp2+2*dp3+dp4)
    return dy, dyp


def fe(dypdt, yp):
    ypp = dypdt(y_fe_1, yp)
    dyp = ypp*dt
    dy = yp*dt
    return dy, dyp


def fe_inc(dypdt, yp1):
    ypp = dypdt(y_fe_2, yp_fe_2)
    dyp = ypp*dt
    yp2 = yp1+dyp
    dy = yp2*dt
    return dy, yp2

print(N, "<--N")
for i in range(1, N+1):

    # ypp = (-m*g*np.sin(y)/l-alpha*l*yp*abs(l*yp))/m
    # yp += ypp*dt
    # ypp = -yp - y
    # yp += ypp*dt
    # y += yp*dt
    dy_rk4, dyp_rk4 = rk4(dypdt)
    y_rk4 += dy_rk4
    yp_rk4 += dyp_rk4
    y_history_rk4[i] = y_rk4

    dy_fe, dyp_fe = fe(dypdt, yp_fe_1)
    y_fe_1 += dy_fe
    yp_fe_1 += dyp_fe
    y_history_fe_1[i] = y_fe_1

    dy_fe_2, yp_fe_2 = fe_inc(dypdt, yp_fe_2)
    y_fe_2 += dy_fe_2
    y_history_fe_2[i] = y_fe_2

    # ypp_1 = (-m*g*np.sin(y_fe_1)/l-alpha*l*yp_fe_1*abs(l*yp_fe_1))/m
    # y_fe_1 += yp_fe_1*dt
    # yp_fe_1 += ypp_1*dt
    # y_history_fe_1[i] = y_fe_1
    #
    # ypp_1 = -yp_fe_1-y_fe_1
    # y_fe_1 += yp_fe_1*dt
    # yp_fe_1 += ypp_1*dt
    # y_history_fe_1[i] = y_fe_1

    # ypp_2 = -yp_fe_2-y_fe_2
    # yp_fe_2 += ypp_2*dt
    # y_fe_2 += yp_fe_2*dt
    # yp_his[i] = yp_fe_2
    # y_history_fe_2[i] = y_fe_2

    # ypp_2 = (-m*g*np.sin(y_fe_2)/l-alpha*l*yp_fe_2*abs(l*yp_fe_2))/m
    # yp_fe_2 += ypp_2*dt
    # y_fe_2 += yp_fe_2*dt
    # y_history_fe_2[i] = y_fe_2


y_solution = y_exact(x)

# rk4_error = y_history_rk4-y_solution
# fe1_error = y_history_fe_1-y_solution
# fe2_error = y_history_fe_2-y_solution

plt.plot(x, y_history_rk4, label='4th order runge kutta')
plt.plot(x, y_history_fe_1, label='forward euler *correct*')
plt.plot(x, y_history_fe_2, label='forward euler *incorrect*')
# plt.plot(x, y_solution, label='exact solution')

# plt.plot(x, rk4_error, label='4th order runge kutta error')
# plt.plot(x, fe1_error, label='forward euler error *correct*')
# plt.plot(x, fe2_error, label='forward euler error *incorrect*')
plt.legend(loc="upper right")
plt.show()
