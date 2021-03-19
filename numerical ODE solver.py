import numpy as np
import matplotlib.pyplot as plt
import timeit

l   = 1.            # (m) Length of pendulum
g   = 9.81          # (m/s^2)
m   = 0.001         # (kg) Mass of the bob
rho = 1.225         # (kg/m^3) Density of air
r   = 0.05          # (m) Radius of sphere
A   = np.pi * r**2  # (m^2) Cross-secional area
c_d = 0.47          # (.) Drag coefficient of a sphere
# alpha = c_d * rho * A / 2.   # Initially assume air-resitance is zero
alpha = 0

T_final = 20     # (s) Final time
initial_conditions = [0,1]      #[yp,y]

error_list = [[[],[]],[[],[]],[[],[]],[]]


def dypdt(y, yp):
    # return (-m*g*np.sin(y)/l-alpha*l*yp*abs(l*yp))/m
    # return -yp-y
    return -y


def y_exact(x):
    # return 2/3*np.sqrt(3)*np.exp(-x/2)*np.sin(np.sqrt(3)/2*x+np.pi/3)
    return np.cos(x)


def rk4(dypdt, dh):
    d1 = dh*yp_rk4
    dp1 = dh*dypdt(y_rk4, yp_rk4)
    d2 = dh*(yp_rk4+0.5*dp1)
    dp2 = dh*dypdt(y_rk4+0.5*d1, yp_rk4+0.5*dp1)\
        # (yp+0.5*dp1+y+0.5*d1)
    d3 = dh*(yp_rk4+0.5*dp2)
    dp3 = dh*dypdt(y_rk4+0.5*d2, yp_rk4+0.5*dp2)\
        # (yp+0.5*dp2+y+0.5*d2)
    d4 = dh*(yp_rk4+dp3)
    dp4 = dh*dypdt(y_rk4+d3, yp_rk4+dp3)\
        # (yp+dp3+y+d3)
    dy = 1/6*(d1+2*d2+2*d3+d4)
    dyp = 1/6*(dp1+2*dp2+2*dp3+dp4)
    return dy, dyp


def fe(dypdt, yp, dh):
    ypp = dypdt(y_fe_1, yp)
    dyp = ypp*dh
    dy = yp*dh
    return dy, dyp


def fe_inc(dypdt, yp1, dh):
    ypp = dypdt(y_fe_2, yp_fe_2)
    dyp = ypp*dh
    yp2 = yp1+dyp
    dy = yp2*dh
    return dy, yp2


for j in range(4):
    toc = timeit.default_timer()

    base = 4
    N = base**(j+1)
    print(f"{N} ({base}^{j+1})<--number of steps")
    dt = T_final / (N)

    yp_rk4 = initial_conditions[0]
    y_rk4 = initial_conditions[1]
    y_history_rk4 = np.zeros(N + 1)
    y_history_rk4[0] = y_rk4

    yp_fe_1 = initial_conditions[0]
    y_fe_1 = initial_conditions[1]
    y_history_fe_1 = np.zeros(N + 1)
    y_history_fe_1[0] = y_fe_1

    yp_fe_2 = initial_conditions[0]
    y_fe_2 = initial_conditions[1]
    y_history_fe_2 = np.zeros(N + 1)
    y_history_fe_2[0] = y_fe_2

    x = np.linspace(0, T_final, N + 1)

    for i in range(1, N+1):
        dy_rk4, dyp_rk4 = rk4(dypdt, dt)
        y_rk4 += dy_rk4
        yp_rk4 += dyp_rk4
        y_history_rk4[i] = y_rk4

        dy_fe, dyp_fe = fe(dypdt, yp_fe_1, dt)
        y_fe_1 += dy_fe
        yp_fe_1 += dyp_fe
        y_history_fe_1[i] = y_fe_1

        dy_fe_2, yp_fe_2 = fe_inc(dypdt, yp_fe_2, dt)
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
    solution_plot_title = f"time step:{dt}"

    index = -1
    y_solution = y_exact(x)
    rk4_error = abs(y_history_rk4-y_solution)
    fe1_error = abs(y_history_fe_1-y_solution)
    fe2_error = abs(y_history_fe_2-y_solution)
    rk4_error_sum = 0
    fe1_error_sum = 0
    fe2_error_sum = 0


    for i in range(len(y_history_fe_1)):
        if i == 0 or i == len(y_history_fe_1)-1:
            rk4_error_sum += 1 / 3 * dt * (rk4_error[i])
            fe1_error_sum += 1 / 3 * dt * (fe1_error[i])
            fe2_error_sum += 1 / 3 * dt * (fe2_error[i])
        elif i % 2 == 1:
            rk4_error_sum += 4 / 3 * dt * (rk4_error[i])
            fe1_error_sum += 4 / 3 * dt * (fe1_error[i])
            fe2_error_sum += 4 / 3 * dt * (fe2_error[i])
        elif i % 2 == 0:
            rk4_error_sum += 2 / 3 * dt * (rk4_error[i])
            fe1_error_sum += 2 / 3 * dt * (fe1_error[i])
            fe2_error_sum += 2 / 3 * dt * (fe2_error[i])

    error_list[0][0].append(np.log(abs(rk4_error[index]))/np.log(10))
    error_list[0][1].append(np.log(abs(rk4_error_sum))/np.log(10))
    # error_list[0][1].append(abs(rk4_error_sum))
    error_list[1][0].append(np.log(abs(fe1_error[index]))/np.log(10))
    error_list[1][1].append(np.log(abs(fe1_error_sum))/np.log(10))
    # error_list[1][1].append(abs(fe1_error_sum))
    error_list[2][0].append(np.log(abs(fe2_error[index]))/np.log(10))
    error_list[2][1].append(np.log(abs(fe2_error_sum))/np.log(10))
    # error_list[2][1].append(abs(fe2_error_sum))
    error_list[3].append(-np.log(dt)/np.log(10))

    # y_history_fe_2.delete()
    # y_history_fe_1.delete()
    # y_history_rk4.delete()

    tic = timeit.default_timer()
    print(f"{tic-toc} <-- time taken for {base}^{j+1} steps (step size = {dt})\n\n")

# fig, subfig = plt.subplots(2,2)
# fig.suptitle('stacked subplots')
# subfig[0][0].plot(x, y_history_rk4, label='4th order runge kutta')
# subfig[0][0].plot(x, y_history_fe_1, label='forward euler *correct*')
# subfig[0][0].plot(x, y_history_fe_2, label='forward euler *incorrect*')
# subfig[0][0].plot(x, y_solution, label='exact solution')
# subfig[0][0].legend(loc="upper right")
# subfig[0][0].set_ylabel('y', loc='center')
# subfig[0][0].set_xlabel('t', loc='left')
# subfig[0][0].set_title(f'Interval bound: {T_final}')
# subfig[0][0].grid()
#
# subfig[0][1].plot(error_list[3], error_list[0][0], label='4th order runge kutta', marker='x')
# subfig[0][1].plot(error_list[3], error_list[1][0], label='forward euler *correct*', marker='x')
# subfig[0][1].plot(error_list[3], error_list[2][0], label='forward euler *incorrect*', marker='x')
# subfig[0][1].legend(loc="upper right")
# subfig[0][1].set_ylabel('log10(error)', loc='center')
# subfig[0][1].set_xlabel('-log10(dt)', loc='left')
# subfig[0][1].set_title('error at interval end')
# subfig[0][1].grid()
#
# subfig[1][0].plot(error_list[3], error_list[0][1], label='4th order runge kutta', marker='x')
# subfig[1][0].plot(error_list[3], error_list[1][1], label='forward euler *correct*', marker='x')
# subfig[1][0].plot(error_list[3], error_list[2][1], label='forward euler *incorrect*', marker='x')
# subfig[1][0].legend(loc="upper right")
# subfig[1][0].set_ylabel('log10(Sigma error)', loc='center')
# subfig[1][0].set_xlabel('-log10(dt)', loc='left')
# subfig[1][0].set_title('integral of error')
# subfig[1][0].grid()

# plt.plot(error_list[3], error_list[0][1], label='4th order runge kutta', marker='x')
# plt.plot(error_list[3], error_list[1][1], label='forward euler *correct*', marker='x')
# plt.plot(error_list[3], error_list[2][1], label='forward euler *incorrect*', marker='x')
# plt.legend(loc="upper right")
# plt.grid()

plt.plot(x, y_history_rk4, label='4th order runge kutta')
plt.plot(x, y_history_fe_1, label='forward euler *correct*')
plt.plot(x, y_history_fe_2, label='forward euler *incorrect*')
plt.legend(loc="upper right")
plt.grid()
print(error_list[3])
plt.show()
