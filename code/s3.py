import utils
import dynamic
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from math import pi, atan2



def dynlin(dX, dU, A, B):
    return np.dot(A, dX)+np.dot(B, dU)

def main():

    PLANE = dynamic.Param_737_800()
    Wh = 2 #m/s vitesse vent
    h = 10000
    Ma = 0.9
    ms = 0.5
    km = 1
    va = dynamic.va_of_mach(Ma, h)
    pt_trim = [h, Ma, ms, km]

    PLANE.set_mass_and_static_margin(km, Ma)

    def q1():
        time = np.arange(0,240,0.1)
        Xtrim, Utrim = dynamic.trim(PLANE, {'va':va, 'h':h})
        add_wind = np.array([0, 0, 0, atan2(Wh, Xtrim[2]), 0, 0])
        x=integrate.odeint(dynamic.dyn, Xtrim+add_wind, time, args=(Utrim, PLANE))
        dynamic.plot(time, x)
        plt.show()

    #q1()

    def q1_lin():
        time = np.arange(0, 240, 0.1)
        params = {'va': dynamic.va_of_mach(Ma, h), 'h': h, 'gamma': 0}
        X, U = dynamic.trim(PLANE, params)
        A, B = dynamic.ut.num_jacobian(X, U, PLANE, dynamic.dyn)
        #A_4 = A[2:, 2:]
        #B_4 = B[2:, 2:]
        dU = np.zeros((2,1))
        add_wind = np.array([0, 0, 0, atan2(Wh, X[2]), 0, 0])
        dx=integrate.odeint(dynlin, add_wind, time, args=(dU, A, B))
        dynamic.plot(time, dx)
        plt.show()

    q1_lin()









if __name__ == "__main__":
    main()
