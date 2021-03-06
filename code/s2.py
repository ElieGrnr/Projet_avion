import utils
import dynamic
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from math import pi

file = "image_s2/"
format = ".png"

def main():

    PLANE = dynamic.Param_737_800()

    h = [3000, 10000]
    Ma = [0.4,0.9]
    ms = [-0.2,0.5]
    km = [0.1,1]
    va0 = dynamic.va_of_mach(Ma[0], 0)
    va1 =dynamic.va_of_mach(Ma[1], 0)


    def q1():
        lh = np.linspace(h[0], h[1], 100)
        n = len(lh)
        for i, ms_i in enumerate(ms):
            for j, km_i in enumerate(km):
                PLANE.set_mass_and_static_margin(km_i, ms_i)
                alpha0 = np.zeros(n)
                dphr0 = np.zeros(n)
                dth0 = np.zeros(n)
                alpha1 = np.zeros(n)
                dphr1 = np.zeros(n)
                dth1 = np.zeros(n)
                for k, h_i in enumerate(lh):
                    va0 = dynamic.va_of_mach(Ma[0], h_i)
                    va1 = dynamic.va_of_mach(Ma[1], h_i)
                    alpha0[k] = dynamic.trim(PLANE, {'va':va0, 'h':h_i})[0][3]
                    dphr0[k] = dynamic.trim(PLANE,{'va':va0, 'h':h_i})[1][0]
                    dth0[k] = dynamic.trim(PLANE, {'va': va0, 'h': h_i})[1][1]
                    alpha1[k] = dynamic.trim(PLANE, {'va': va1, 'h': h_i})[0][3]
                    dphr1[k] = dynamic.trim(PLANE, {'va': va1, 'h': h_i})[1][0]
                    dth1[k] = dynamic.trim(PLANE, {'va': va1, 'h': h_i})[1][1]

                plt.figure("m_s={}, k_m={}".format(ms_i, km_i))
                plt.subplot(1, 3, 1)
                plt.title("$h \\mapsto \\alpha$")
                #plt.title("$m_s={}, k_m={}$".format(ms_i, km_i))
                plt.plot(lh, alpha0, label="$M_a=0.4$")
                plt.plot(lh, alpha1, label="$M_a=0.9$")
                plt.legend()
                #plt.xlabel("$h$")
                #plt.ylabel("$\\alpha$")
                #plt.show()

                #plt.figure()
                plt.subplot(1, 3, 2)
                #plt.title("$m_s={}, k_m={}$".format(ms_i, km_i))
                plt.title("$h \\mapsto \\delta_{{th}}$")
                plt.plot(lh, dth0, label="$M_a=0.4$")
                plt.plot(lh, dth1, label="$M_a=0.9$")
                #plt.legend()
                #plt.xlabel("$h$")
                #plt.ylabel("$\\delta_{{th}}$")
                #plt.show()

                #plt.figure()
                plt.subplot(1, 3, 3)
                #plt.title("$m_s={}, k_m={}$".format(ms_i, km_i))
                plt.title("$h \\mapsto \\delta_{{PHR}}$")
                plt.plot(lh, dphr0, label="$M_a=0.4$")
                plt.plot(lh, dphr1, label="$M_a=0.9$")
                #plt.xlabel("$h$")
                #plt.ylabel("$\\delta_{{PHR}}$")
                #plt.legend()
                plt.tight_layout(.5)
                plt.savefig(file + "q1_" + str(i) + str(j) + format)
                #plt.show()

        plt.show()


    def q2():
        lm = np.linspace(Ma[0], Ma[1])
        n = len(lm)
        for i, ms_i in enumerate(ms):
            for j, km_i in enumerate(km):
                PLANE.set_mass_and_static_margin(km_i, ms_i)
                F0 = np.zeros(n)
                F1 = np.zeros(n)
                for k, m_i in enumerate(lm):
                    va0 = dynamic.va_of_mach(m_i, h[0])
                    va1 = dynamic.va_of_mach(m_i, h[1])
                    X0, U0 = dynamic.trim(PLANE, {'va':va0, 'h':h[0]})
                    X1, U1 = dynamic.trim(PLANE, {'va': va1, 'h': h[1]})
                    F0[k] = dynamic.propulsion_model(X0, U0, PLANE)
                    F1[k] = dynamic.propulsion_model(X1, U1, PLANE)
                plt.subplot(2,2,i+2*j+1)
                plt.plot(lm, F0, label="$h={}$".format(h[0]))
                plt.plot(lm, F1, label="$h={}$".format(h[1]))
                plt.legend()
                plt.title("$m_s={}, k_m={}$".format(ms_i, km_i))
                plt.xlabel("$M_a$")
                plt.ylabel("$F$")
        plt.tight_layout(.5)
        plt.savefig(file + "q2" + format)
        plt.show()

    def get_Cl():
        Va = dynamic.va_of_mach(Ma[1],h[1])
        PLANE.set_mass_and_static_margin(km[1], ms[1])
        rho = utils.isa(h[1])[1]
        Cl = (PLANE.m*PLANE.g)/(0.5*rho*PLANE.S*Va**2)
        return Cl

    def get_dth():
        Cd = 0.030 # obtenu par lecture graphique à partir de la courbe polaire séance1
        # Cl = 0.3706 : fct get_Cl()
        # alpha_eq = 3.66° : courbe q5 séance 1
        # delat_THR_eq = -8.77° : courbe q4 séance1
        f = get_Cl()/Cd
        F = PLANE.m*PLANE.g/f
        rho0 = utils.isa(h[1])[1]
        rho = utils.isa(h[1])[1]
        dth = F/(PLANE.F0*(rho/rho0)**(0.6)*(0.568+0.25*(1.2-Ma[1])**3))
        return dth

    def get_dPHR_alpha_num():
        St_over_S = PLANE.St/PLANE.S
        CLta = PLANE.CLat
        CLwba = PLANE.CLa
        a0 = PLANE.a0
        CL0 = (St_over_S*0.25*CLta-CLwba)*a0
        CLa = CLwba+St_over_S*CLta*(1-0.25)
        #CLq = PLANE.lt*St_over_S*PLANE.CLat*PLANE.CLq
        CLdphr = St_over_S*CLta
        Cm0 = PLANE.Cm0
        Cma = -PLANE.ms*CLwba
        Cmdphr = PLANE.Cmd
        CL = get_Cl()
        alpha_eq = (Cm0-Cma*a0+Cmdphr*(CL-CL0)/CLdphr)/(-Cma+Cmdphr*CLa/CLdphr)
        dphr_eq = (CL-CL0-CLa*alpha_eq)/CLdphr
        return alpha_eq*180/pi, dphr_eq*180/pi


    def simu():
        time = np.arange(0,100,0.1)
        PLANE.set_mass_and_static_margin(km[1],Ma[1])
        va = dynamic.va_of_mach(Ma[1],h[1])
        Xtrim, Utrim = dynamic.trim(PLANE, {'va':va, 'h':h[1]})
        x=integrate.odeint(dynamic.dyn, Xtrim, time, args=(Utrim, PLANE))
        dynamic.plot(time, x)
        plt.savefig(file+"simu"+format)
        plt.show()


    def val_propre(n):
        vp = np.zeros((n,n,n,n,4), dtype=complex)
        lh = np.linspace(h[0],h[1],n)
        lMa = np.linspace(Ma[0],Ma[1],n)
        lms = np.linspace(ms[0],ms[1],n)
        lkm = np.linspace(km[0],km[1],n)
        for i, h_i in enumerate(lh):
            for j, Ma_j in enumerate(lMa):
                for k, ms_k in enumerate(lms):
                    for l, km_l in enumerate(lkm):
                        PLANE.set_mass_and_static_margin(km_l, ms_k)
                        params = {'va': dynamic.va_of_mach(Ma_j, h_i),'h': h_i, 'gamma': 0}
                        X, U = dynamic.trim(PLANE, params)
                        A, B = dynamic.ut.num_jacobian(X, U, PLANE, dynamic.dyn)
                        A_4 = A[2:, 2:]
                        liste_vp = np.linalg.eigvals(A_4)
                        for m, ev in enumerate(liste_vp):
                            vp[i][j][k][l][m] = ev
        #print(A_4)
        #print(np.linalg.eigvals(A_4))
        return vp


    def plot_h(n,vp):
        for j, Ma_j in enumerate(Ma):
            for k, ms_k in enumerate(ms):
                for l, km_l in enumerate(km):
                    vp_Va = vp[:,j*(n-1),k*(n-1),l*(n-1),0]
                    vp_a = vp[:,j*(n-1),k*(n-1),l*(n-1),1]
                    vp_theta = vp[:,j*(n-1),k*(n-1),l*(n-1),2]
                    vp_q = vp[:,j*(n-1),k*(n-1),l*(n-1),3]
                    plt.figure()
                    plt.title("$M_a={}, m_s={}, k_m={}, h\in [{},{}]$".format(Ma_j, ms_k, km_l, h[0], h[1]))
                    decorate(vp_Va, vp_a, vp_theta, vp_q)
                    if j==0 and k==1 and l==1:
                        plt.savefig(file+"vp_h"+format)
        plt.show()

    def plot_Ma(n, vp):
        for i, h_i in enumerate(h):
            for k, ms_k in enumerate(ms):
                for l, km_l in enumerate(km):
                    vp_Va = vp[i*(n-1),:,k*(n-1),l*(n-1),0]
                    vp_a = vp[i*(n-1),:,k*(n-1),l*(n-1),1]
                    vp_theta = vp[i*(n-1),:,k*(n-1),l*(n-1),2]
                    vp_q = vp[i*(n-1),:,k*(n-1),l*(n-1),3]
                    plt.figure()
                    plt.title("$h={}, m_s={}, k_m={}, M_a\in [{},{}]$".format(h_i, ms_k, km_l, Ma[0], Ma[1]))
                    decorate(vp_Va, vp_a, vp_theta, vp_q)
                    if i==1 and k==1 and l==1:
                        plt.savefig(file+"vp_Ma"+format)
        plt.show()

    def plot_ms(n, vp):
            for i, h_i in enumerate(h):
                for j, Ma_j in enumerate(Ma):
                    for l, km_l in enumerate(km):
                        vp_Va = vp[i*(n-1),j*(n-1),:,l*(n-1),0]
                        vp_a = vp[i*(n-1),j*(n-1),:,l*(n-1),1]
                        vp_theta = vp[i*(n-1),j*(n-1),:,l*(n-1),2]
                        vp_q = vp[i*(n-1),j*(n-1),:,l*(n-1),3]
                        plt.figure()
                        plt.title("$h={}, M_a={}, k_m={}, m_s\in [{},{}]$".format(h_i, Ma_j, km_l, ms[0], ms[1]))
                        decorate(vp_Va, vp_a, vp_theta, vp_q)
                        if i == 1 and j == 0 and l == 1:
                            plt.savefig(file + "vp_ms" + format)
            plt.show()

    def plot_km(n, vp):
            for i, h_i in enumerate(h):
                for j, Ma_j in enumerate(Ma):
                    for k, ms_k in enumerate(ms):
                        vp_Va = vp[i*(n-1),j*(n-1),k*(n-1),:,0]
                        vp_a = vp[i*(n-1),j*(n-1),k*(n-1),:,1]
                        vp_theta = vp[i*(n-1),j*(n-1),k*(n-1),:,2]
                        vp_q = vp[i*(n-1),j*(n-1),k*(n-1),:,3]
                        plt.figure()
                        plt.title("$h={}, M_a={}, m_s={}, k_m\in [{},{}]$".format(h_i, Ma_j, ms_k, km[0], km[1]))
                        decorate(vp_Va, vp_a, vp_theta, vp_q)
                        if i == 1 and j == 0 and k == 1:
                            plt.savefig(file + "vp_km" + format)
            plt.show()


    #q1()
    #q2()
    print("Au point de trim (h, Ma, ms, km)=(10000, 0.9, 0.5, 1), Cl = " + str(get_Cl()))
    print("Lecture graphique")
    print("Cd = 0.030 obtenu par lecture graphique à partir de la courbe polaire séance1")
    print("alpha_eq = 3.66° : courbe q5 séance 1")
    print("delat_PHR_eq = -8.77° : courbe q4 séance1")
    print("on calcule dth : dth=" + str(get_dth()))
    print("Calcul numérique : ")
    print("dPHR = " + str(get_dPHR_alpha_num()[1]))
    print("alpha = " + str(get_dPHR_alpha_num()[0]))
    #simu()
    n = 10
    vp = val_propre(n)
    plot_h(n, vp)
    plot_Ma(n, vp)
    plot_ms(n, vp)
    plot_km(n, vp)


def decorate(vp_Va, vp_a, vp_theta, vp_q):
    plt.plot(vp_Va.real, vp_Va.imag, label="$V_a$")
    plt.plot(vp_a.real, vp_a.imag, label="$\\alpha$")
    plt.plot(vp_theta.real, vp_theta.imag, label="$\\theta$")
    plt.plot(vp_q.real, vp_q.imag, label="$q$")
    plt.plot([vp_Va[0].real], [vp_Va[0].imag], marker='x')
    plt.plot([vp_a[0].real], [vp_a[0].imag], marker='x')
    plt.plot([vp_theta[0].real], [vp_theta[0].imag], marker='x')
    plt.plot([vp_q[0].real], [vp_q[0].imag], marker='x')
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.grid(linestyle='--')
    plt.legend()





if __name__ == "__main__":
    main()