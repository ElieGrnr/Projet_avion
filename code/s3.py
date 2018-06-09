import utils
import dynamic
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import signal
from math import pi, atan2, sqrt



def dynlin(dX, time, dU, A, B):
    """
    computes dX_dot
    :param dX: state
    :param time: array
    :param dU: input
    :param A: state matrix
    :param B: input matrix
    :return:
    """
    return np.dot(A, dX)+np.dot(B, dU)

def mode(pole):
    """
    computes pseudo pulse and damping coeff of pole
    :param pole: cpx
    :return: w0, ksi
    """
    w0 = sqrt(pole.real**2 + pole.imag**2)
    ksi = -pole.real/w0
    return w0, ksi

def state(PLANE, pt_trim):
    """
    Computes model of a trim condition
    :param PLANE:
    :param pt_trim: list of [h, Ma, ms, km]
    :return:
    """
    PLANE.set_mass_and_static_margin(pt_trim[3], pt_trim[2])
    params = {'va': dynamic.va_of_mach(pt_trim[1], pt_trim[0]), 'h': pt_trim[0], 'gamma': 0}
    X, U = dynamic.trim(PLANE, params)
    A, B = dynamic.ut.num_jacobian(X, U, PLANE, dynamic.dyn)
    return X, U, A, B

def step(num, den, time, label):
    """
    Draws output for a step input
    :param num: [a_n, a_n-1, ..., a_], numerator of transfer function
    :param den: idem, denominator of transfer function
    :param time:
    :return:
    """
    F = signal.TransferFunction(num, den)
    ts, yout = signal.step2(F, T=time)
    plt.plot(ts, yout, label=label)

def Bode(nums, dens, labels):
    """
    Draw Bode's diagram for transfert function num[i]/den[i]
    :param nums:list of numerator (list)
    :param dens: list of denominators (list)
    :param labels: list of labels
    :return: magnitude and phase diagrams
    """
    for i in range(len(nums)):
        F = signal.TransferFunction(nums[i], dens[i])
        w, mag, phase = signal.bode(F)
        ax1 = plt.subplot(2,1,1)
        ax1.semilogx(w, mag,label=labels[i])
        plt.legend()
        plt.xlabel("Pulse $\log\omega$")
        plt.ylabel("$\left| H(\omega)\\right|$")
        ax2 = plt.subplot(2, 1, 2)
        ax2.semilogx(w, phase,label=labels[i])
        plt.legend()
        plt.xlabel("Pulse $\log\omega$")
        plt.ylabel("$\phi(\omega)$")


def main():

    PLANE = dynamic.Param_737_800()
    Wh = 2 #m/s wind velocity
    h1, h2 = 10000, 3000
    Ma1, Ma2 = 0.9, 0.4
    ms1, ms2 = 0.5, 0.5
    km1, km2 = 1, 0.1
    va = dynamic.va_of_mach(Ma1, h1)
    pt_trim1 = [h1, Ma1, ms1, km1]
    pt_trim2 = [h2, Ma2, ms2, km2]

    def compare_lin(T, pt_trim):
        X, U, A, B = state(PLANE, pt_trim)
        time = np.arange(0,T,0.1)
        val_p = np.linalg.eigvals(A[2:, 2:])
        add_wind = np.array([0, 0, 0, atan2(Wh, X[2]), 0, 0])
        x=integrate.odeint(dynamic.dyn, X+add_wind, time, args=(U, PLANE))
        dynamic.plot(time, x)
        plt.show()
        dU = np.zeros((4,))
        dx = integrate.odeint(dynlin, add_wind, time, args=(dU, A, B))
        out = np.array([dxi + X for dxi in dx])
        for j, Xj in enumerate(out):
            out[j][0] += out[j][2] * time[j]
        dynamic.plot(time, out)
        plt.show()
        dynamic.plot(time, abs(out-x))
        plt.show()
        print([mode(val_p[i]) for i in range(len(val_p))])

    def q3(T, pt_trim1, pt_trim2):
        _, _, A, B = state(PLANE, pt_trim1)
        X, U, _, _ = state(PLANE, pt_trim2)
        add_wind = np.array([0, 0, 0, atan2(Wh, X[2]), 0, 0])
        time = np.arange(0,T,0.1)
        x=integrate.odeint(dynamic.dyn, X+add_wind, time, args=(U, PLANE))
        dynamic.plot(time, x)
        plt.show()
        dU = np.zeros((4,))
        dx = integrate.odeint(dynlin, add_wind, time, args=(dU, A, B))
        out = np.array([dxi + X for dxi in dx])
        for j, Xj in enumerate(out):
            out[j][0] += out[j][2] * time[j]
        dynamic.plot(time, out)
        plt.show()
        dynamic.plot(time, abs(out-x))
        plt.show()

    def modal_form(pt_trim):
        _, _, A, B = state(PLANE, pt_trim)
        A_4, B_4 = A[2:, 2:], B[2:, :2]
        val_p, M = np.linalg.eig(A_4)
        M_inv = np.linalg.inv(M)
        Am_4 = np.diag(val_p)
        Bm_4 = np.dot(M_inv, B_4)
        return Am_4, Bm_4

    def stability(pt_trim):
        _, _, A, B = state(PLANE, pt_trim)
        return np.linalg.eigvals(A[2:, 2:])

    def controllability(pt_trim):
        _, _, A, B = state(PLANE, pt_trim)
        A_4, B_4 = A[2:, 2:], B[2:, :2]
        Q = np.zeros((4, 4*2))
        for i in range(3):
            Q[:,2*i:2*(i+1)] = np.dot(np.linalg.matrix_power(A_4, i),B_4)
        return Q

    def transfer_function(pt_trim):
        _, _, A, B = state(PLANE, pt_trim)
        A_4, B_4 = A[2:, 2:], B[2:, :2][:, 0].reshape((4, 1))
        C_4 = np.array([0,0,1,0])
        Acc_4 = np.zeros((4,4))
        Bcc_4 = np.array([[0],[0],[0],[1]])
        val_p = np.linalg.eigvals(A_4)
        coef = np.poly(val_p)
        N=4
        for i in range(3):
            Acc_4[3,N-1-i]=-coef[i+1]
            Acc_4[i, i+1]=1
        Acc_4[3,0]=-coef[N]
        Qccc = np.zeros((4, 4))
        Q = np.zeros((4,4))
        for i in range(4):
            Qccc[:,i:i+1] = np.dot(np.linalg.matrix_power(Acc_4, i),Bcc_4)
            Q[:,i:i+1] = np.dot(np.linalg.matrix_power(A_4, i),B_4)
        Mcc = np.dot(Q, np.linalg.inv(Qccc))
        Ccc_4 = np.dot(C_4, Mcc)
        num = list(Ccc_4)
        num.reverse()
        den = list(-Acc_4[3,:])
        den.append(1.)
        den.reverse()
        return num, den, val_p

    def Pade_reduction(pt_trim):
        num, den, val_p = transfer_function(pt_trim)
        p = num[-1]/den[-1]
        q = (num[-2]-(num[-1]/den[-1])*den[-2])/den[-1]
        poles = sorted(val_p, key=lambda x: abs(x))
        pade_poles = poles[0:2]
        den_pade = np.poly(pade_poles)
        b_0 = den_pade[-1]*p
        b_1 = q*den_pade[-1]+b_0*den_pade[-2]/den_pade[-1]
        num_pade = [b_1, b_0]
        return num_pade, den_pade



    compare_lin(240, pt_trim1)
    compare_lin(10, pt_trim1)
    q3(240, pt_trim1, pt_trim2)

    print("A_m = ", modal_form(pt_trim1)[0])
    print("B_m = ", modal_form(pt_trim1)[1])

    print("Les valeurs propres sont : ", stability(pt_trim1))

    print("La matrice de commandabilité est Q= ", controllability(pt_trim1))

    num, den, _ = transfer_function(pt_trim1)
    print("Les coefficients de la fonction de transfert sont : ")
    print("numérateur : ", num)
    print("dénominateur : ", den)
    num_pade, den_pade = Pade_reduction(pt_trim1)
    print("Les coefficients de la fonction de transfert réduite sont : ")
    print("numérateur : ", num_pade)
    print("dénominateur : ", den_pade)

    time = np.arange(0, 240, 0.2)
    plt.figure()
    step(num, den, time, "$F$")
    step(num_pade, den_pade, time, "$F_r$")
    plt.legend()
    plt.xlabel("time ($s$)")
    plt.ylabel("$\\theta$")
    plt.show()

    plt.figure()
    Bode([num, num_pade], [den, den_pade], ["$F$", "$F_r$"] )

    plt.show()

if __name__ == "__main__":
    main()
