import utils, dynamic
import matplotlib.pyplot as plt
import numpy as np
from math import pi

def main():

    PLANE = dynamic.Param_737_800()
    Vt = PLANE.lt*PLANE.St/PLANE.cbar/PLANE.S
    va = dynamic.va_of_mach(0.9, 0)
    km = 1

    ###q1

    def pousse_max(h, mach):
        M = np.linspace(mach[0], mach[1], 100)
        rho0 = utils.isa(0)[1]
        F0 = PLANE.F0
        plt.figure(0)
        for h_i in h :
            F = F0*(utils.isa(h_i)[1]/rho0)**(0.6)*(0.568+0.25*(1.2-M)**3)
            plt.plot(M, F, label='$h={:.0f}$'.format(h_i))
        plt.xlabel("$M_a$")
        plt.ylabel("$F$")
        plt.title("$M_a\mapsto F$")
        plt.legend()
        plt.show()

    ###q2

    def coeff_Cl(a, delta_PHR):
        """

        :param a: a list of two angles (min an max) in degree !
        :param delta_PHR: degree !
        :return:
        """
        alpha = np.linspace(a[0], a[1], 100)
        plt.figure(1)
        for d in delta_PHR:
            Cl = dynamic.get_aero_coefs(va, alpha*pi/180, 0, d*pi/180, PLANE)[0]
            plt.plot(alpha, Cl, label="$\delta_{{PHR}}={:.0f}$".format(d))
        plt.xlabel("$\\alpha$")
        plt.ylabel("$C_L$")
        plt.title("$\\alpha\\mapsto C_l$")
        plt.legend()
        plt.show()

    ###q3

    def coeff_Cm(a, m_s):
        dphr = 0
        alpha = np.linspace(a[0], a[1], 100)
        m_s_original = PLANE.ms
        plt.figure(2)
        for m in m_s:
            PLANE.set_mass_and_static_margin(km, m)
            Cm = dynamic.get_aero_coefs(va, alpha*pi/180, 0, dphr, PLANE)[2]
            plt.plot(alpha, Cm, label="$m_s={}$".format(m))
        plt.xlabel("$\\alpha$")
        plt.ylabel("$C_m$")
        plt.title("$\\alpha \\mapsto C_m$")
        plt.legend()
        plt.show()
        PLANE.set_mass_and_static_margin(km, m_s_original)

    ###q4



    def dphre(a, m_s, vt):
        """

        :param a:couple of alpha angle (min and max) in degrees!
        :param m_s:
        :param vt:
        :return:
        """
        ms_original = PLANE.ms
        alpha = np.linspace(a[0], a[1], 100)
        alpha0 = PLANE.a0*180/pi
        plt.figure(3)
        for m in m_s:
            PLANE.set_mass_and_static_margin(km, m)
            dphre = (PLANE.Cm0 - PLANE.ms * PLANE.CLa * ((alpha - alpha0)*pi/180)) / (Vt * PLANE.CLat)
            plt.plot(alpha, dphre*180/pi, label='$m_s = {}$'.format(m))
        plt.xlabel('$\\alpha_e$')
        plt.ylabel('$\\delta_{{PHRe}}$')
        plt.title("$\\alpha \\mapsto \\delta_{{PHRe}}$")
        plt.legend()
        plt.show()

        PLANE.set_mass_and_static_margin(km, ms_original)
        plt.figure(4)
        for v in vt:
            dphre = (PLANE.Cm0 - PLANE.ms * PLANE.CLa * ((alpha - alpha0) * pi / 180)) / (v * PLANE.CLat)
            plt.plot(alpha, dphre*180/pi, label='$V_t = {:.3f}$'.format(v))
        plt.title("$\\alpha_e \\mapsto \\delta_{{PHRe}}$")
        plt.xlabel('$\\alpha_e$')
        plt.ylabel('$\\delta_{{PHRe}}$')
        plt.legend()
        plt.show()

    ### q5

    def coeff_CLe(a, m_s):
        alpha = np.linspace(a[0], a[1], 100)
        alpha0 = PLANE.a0 * 180 / pi
        plt.figure(5)
        for m in m_s:
            PLANE.set_mass_and_static_margin(km, m)
            dphre = (PLANE.Cm0 - PLANE.ms * PLANE.CLa * ((alpha - alpha0)*pi/180)) / (Vt * PLANE.CLat)
            CL = dynamic.get_aero_coefs(va, alpha*pi/180, 0, dphre, PLANE)[0]
            plt.plot(alpha, CL, label='$m_s={}$'.format(m))
        plt.xlabel("$\\alpha_{eq}$")
        plt.ylabel("$C_{L_e}$")
        plt.title('$\\alpha_e\\mapsto C_L$')
        plt.legend()
        plt.show()


    ###q6

    def polaire(a, m_s):
        alpha = np.linspace(a[0], a[1], 100)
        plt.figure(6)
        fmax = np.zeros(len(m_s))
        for i, m in enumerate(m_s):
            PLANE.set_mass_and_static_margin(km, m)
            dphre = (PLANE.Cm0 - PLANE.ms * PLANE.CLa * (alpha*pi/180)) / (Vt * PLANE.CLat)
            CL, CD = dynamic.get_aero_coefs(va, alpha*pi/180, 0,dphre, PLANE)[0:2]
            plt.plot(CD, CL, label='$m_s={}$'.format(m))
            fmax[i] = max(CL / CD)
        plt.title('Polaire équilibrée')
        plt.xlabel('$C_D$')
        plt.ylabel('$C_L$')
        plt.legend()
        plt.show()
        return fmax

    alpha = [-10, 20]
    pousse_max([3000, 10000], [0.4, 0.9])
    coeff_Cl(alpha, [-30, 20])
    coeff_Cm(alpha, [-0.1, 0, 0.2, 1])
    dphre(alpha, [-0.1, 0, 0.2, 1], [0.9 * Vt, Vt, 1.1 * Vt])
    coeff_CLe(alpha, [0.2, 1])
    polaire(alpha, [0.2, 1])
    print(polaire(alpha, [0.2, 1]))

if __name__ == "__main__":
    main()









