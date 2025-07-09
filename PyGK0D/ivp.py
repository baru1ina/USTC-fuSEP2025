import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, iv
from scipy.signal import find_peaks
import time
import math
from numba import njit, complex128, float64


@njit('float64[:](float64[:])')
def safe_log(x):
    x = np.asarray(x)
    x_safe = np.where(x <= 0, 1e-20, x)
    return np.log(x_safe)


# @njit
# def compute_w_coeffs(ky, kz, tau, mi, vx, vy, kapn, kapt):
#     wdi = ky
#     wde = -wdi * tau
#     ki = ky
#     ke = -ky * np.sqrt(tau / mi)
#     kzi = kz
#     kze = kz * np.sqrt(tau * mi)
#     Gamma0i = iv(0, ki ** 2) * np.exp(-ki ** 2)
#     Gamma0e = iv(0, ke ** 2) * np.exp(-ke ** 2)
#     G0coef = 1.0 / ((1.0 - Gamma0i) + (1.0 - Gamma0e) / tau) / np.sqrt(2.0 * np.pi)
#
#     wDiv = wdi * (vx ** 2 + vy ** 2 / 2) + kzi * vx
#     wTiv = wdi * (kapn + (0.5 * (vx ** 2 + vy ** 2) - 1.5) * kapt)
#     wDev = wde * (vx ** 2 + vy ** 2 / 2) + kze * vx
#     wTev = wde * (kapn + (0.5 * (vx ** 2 + vy ** 2) - 1.5) * kapt)
#
#     return wDiv, wTiv, wDev, wTev, G0coef


# @njit
# def rk4_step(gi, ge, phi, wDiv, wTiv, wDev, wTev, F0, vy, dvx, dvy, dt, tau, G0coef, ki, ke):
#     # Шаг 1
#     dgi1 = -1j * (wDiv * gi + (wDiv - wTiv) * phi * jv(0, ki * vy) ** 2 * F0)
#     dge1 = -1j * (wDev * ge + (wDev - wTev) * phi * jv(0, ke * vy) ** 2 * F0)
#
#     # Шаг 2
#     gitmp = gi + 0.5 * dt * dgi1
#     getmp = ge + 0.5 * dt * dge1
#     phitmp = G0coef * np.sum((gitmp + getmp / tau) * vy) * dvx * dvy
#
#     dgi2 = -1j * (wDiv * gitmp + (wDiv - wTiv) * phitmp * jv(0, ki * vy) ** 2 * F0)
#     dge2 = -1j * (wDev * getmp + (wDev - wTev) * phitmp * jv(0, ke * vy) ** 2 * F0)
#
#     # Шаг 3
#     gitmp = gi + 0.5 * dt * dgi2
#     getmp = ge + 0.5 * dt * dge2
#     phitmp = G0coef * np.sum((gitmp + getmp / tau) * vy) * dvx * dvy
#
#     dgi3 = -1j * (wDiv * gitmp + (wDiv - wTiv) * phitmp * jv(0, ki * vy) ** 2 * F0)
#     dge3 = -1j * (wDev * getmp + (wDev - wTev) * phitmp * jv(0, ke * vy) ** 2 * F0)
#
#     # Шаг 4
#     gitmp = gi + dt * dgi3
#     getmp = ge + dt * dge3
#     phitmp = G0coef * np.sum((gitmp + getmp / tau) * vy) * dvx * dvy
#
#     dgi4 = -1j * (wDiv * gitmp + (wDiv - wTiv) * phitmp * jv(0, ki * vy) ** 2 * F0)
#     dge4 = -1j * (wDev * getmp + (wDev - wTev) * phitmp * jv(0, ke * vy) ** 2 * F0)
#
#     gi_new = gi + dt / 6.0 * (dgi1 + 2.0 * dgi2 + 2.0 * dgi3 + dgi4)
#     ge_new = ge + dt / 6.0 * (dge1 + 2.0 * dge2 + 2.0 * dge3 + dge4)
#     phi_new = G0coef * np.sum((gi_new + ge_new / tau) * vy) * dvx * dvy
#
#     return gi_new, ge_new, phi_new


def run(irun=0, nt=500, plot_results=True, real_time_plot=False):

    runtime_start = time.time()

    data = np.array([
        [0.1, 0.1277 + 2.8258j],
        [0.2, 0.2487 + 2.8211j],
        [0.5, 0.6369 + 2.6664j],
        [1.0, 1.3108 + 2.2697j],
        [1.5, 2.1182 + 1.9633j],
        [2.0, 3.1047 + 1.9143j]
    ])

    id = 3
    ky = data[id, 0]
    wr = data[id, 1].real
    wi = data[id, 1].imag

    tau = 1.0  # Te/Ti
    epsn = 0.2
    kapn = 1 / epsn  # R/L_n
    kapt = 0.1 * kapn  # R/L_T
    eta = kapt / kapn

    kz = 0.0
    mi = 1836.0

    wdi = ky
    wde = -wdi * tau
    ki = ky
    ke = -ky * np.sqrt(tau / mi)
    kzi = kz
    kze = kz * np.sqrt(tau * mi)
    Gamma0i = iv(0, ki ** 2) * np.exp(-ki ** 2)
    Gamma0e = iv(0, ke ** 2) * np.exp(-ke ** 2)
    G0coef = 1.0 / ((1.0 - Gamma0i) + (1.0 - Gamma0e) / tau) / np.sqrt(2.0 * np.pi)

    nvx, nvy =  32, 64
    dt = 0.02 / 1
    vxmax, vymax = 5.0, 5.0
    vxmin = -vxmax
    vymin = 0.0

    dvx = (vxmax - vxmin) / nvx
    dvy = (vymax - vymin) / nvy

    Vx = np.arange(vxmin, vxmax + dvx / 2, dvx)
    Vy = np.arange(vymin, vymax + dvy / 2, dvy)

    vx, vy = np.meshgrid(Vx, Vy)

    wDiv = wdi * (vx ** 2 + vy ** 2 / 2) + kzi * vx
    wTiv = wdi * (kapn + (0.5 * (vx ** 2 + vy ** 2) - 1.5) * kapt)
    wDev = wde * (vx ** 2 + vy ** 2 / 2) + kze * vx
    wTev = wde * (kapn + (0.5 * (vx ** 2 + vy ** 2) - 1.5) * kapt)

    # wDiv, wTiv, wDev, wTev, G0coef = compute_w_coeffs(ky, kz, tau, mi, vx, vy, kapn, kapt)

    F0 = np.exp(-0.5 * (vx ** 2 + vy ** 2))  # initial distribution
    # F0 = 2.0 /np.sqrt(math.pi) * vy * np.exp(-0.5 * (vx ** 2 + vy ** 2) ** 2)  # initial distribution
    gi = 0.001 * F0
    ge = 0.001 * F0

    phi = 0.1
    phit = np.zeros(nt, dtype=complex)

    rk = 1
    # rk = 0

    for it in range(nt):
        if rk == 0:  # Метод Эйлера ????
            phi = G0coef * np.sum((gi + ge / tau) * vy) * dvx * dvy

            gi = gi - 1j * (wDiv * gi + (wDiv - wTiv) * phi * (jv(0, ki * vy) ** 2 * F0)) * dt
            ge = ge - 1j * (wDev * ge + (wDev - wTev) * phi * (jv(0, ke * vy) ** 2 * F0)) * dt

            # gi[:, 0] = 0
            # gi[:, -1] = 0
            # gi[0, :] = 0
            # gi[-1, :] = 0
            # ge[:, 0] = 0
            # ge[:, -1] = 0
            # ge[0, :] = 0
            # ge[-1, :] = 0

        else:
            # Шаг 1
            dgi1 = -1j * (wDiv * gi + (wDiv - wTiv) * phi * jv(0, ki * vy) ** 2 * F0)
            dge1 = -1j * (wDev * ge + (wDev - wTev) * phi * jv(0, ke * vy) ** 2 * F0)

            # Шаг 2
            gitmp = gi + 0.5 * dt * dgi1
            getmp = ge + 0.5 * dt * dge1
            phitmp = G0coef * np.sum(np.sum((gitmp + getmp / tau) * vy)) * dvx * dvy

            dgi2 = -1j * (wDiv * gitmp + (wDiv - wTiv) * phitmp * jv(0, ki * vy) ** 2 * F0)
            dge2 = -1j * (wDev * getmp + (wDev - wTev) * phitmp * jv(0, ke * vy) ** 2 * F0)

            # Шаг 3
            gitmp = gi + 0.5 * dt * dgi2
            getmp = ge + 0.5 * dt * dge2
            phitmp = G0coef * np.sum(np.sum((gitmp + getmp / tau) * vy)) * dvx * dvy

            dgi3 = -1j * (wDiv * gitmp + (wDiv - wTiv) * phitmp * jv(0, ki * vy) ** 2 * F0)
            dge3 = -1j * (wDev * getmp + (wDev - wTev) * phitmp * jv(0, ke * vy) ** 2 * F0)

            # Шаг 4
            gitmp = gi + dt * dgi3
            getmp = ge + dt * dge3
            phitmp = G0coef * np.sum(np.sum((gitmp + getmp / tau) * vy)) * dvx * dvy

            dgi4 = -1j * (wDiv * gitmp + (wDiv - wTiv) * phitmp * jv(0, ki * vy) ** 2 * F0)
            dge4 = -1j * (wDev * getmp + (wDev - wTev) * phitmp * jv(0, ke * vy) ** 2 * F0)

            gi = gi + dt / 6.0 * (dgi1 + 2.0 * dgi2 + 2.0 * dgi3 + dgi4)
            ge = ge + dt / 6.0 * (dge1 + 2.0 * dge2 + 2.0 * dge3 + dge4)
            phi = G0coef * np.sum(np.sum((gi + ge / tau) * vy)) * dvx * dvy

            # gi, ge, phi = rk4_step(gi, ge, phi, wDiv, wTiv, wDev, wTev, F0, vy, dvx, dvy, dt, tau, G0coef, ki, ke)

        phit[it] = phi

        elapsed_time = time.time() - runtime_start

        if real_time_plot and (it % 10 == 9):
            plt.figure(1, figsize=(12, 6))
            plt.clf()

            plt.subplot(121)
            plt.plot(np.arange(1, it + 2) * dt, np.real(phit[:it + 1]),
                     np.arange(1, it + 2) * dt, np.imag(phit[:it + 1]), 'r--', linewidth=2)
            plt.xlabel('t')
            plt.legend(['Re$\phi$', 'Im$\phi$'])
            plt.title(f'$\phi$, it={it + 1}/{nt}')

            plt.subplot(122)
            plt.plot(np.arange(1, it + 2) * dt, safe_log(np.abs(np.real(phit[:it + 1]))), 'b',
                     np.arange(1, it + 2) * dt, safe_log(np.abs(np.imag(phit[:it + 1]))), 'b--',
                     np.arange(1, it + 2) * dt, safe_log(np.abs(np.abs(phit[:it + 1]))), 'k:', linewidth=2)

            plt.title(f'$\omega^T$ = {wr + 1j * wi}')
            plt.xlabel(f't')
            plt.ylabel('log(|$\phi$|)')

            plt.draw()
            plt.pause(0.01)

        runtime_start = time.time() - elapsed_time

    runtime = time.time() - runtime_start

    if plot_results:
        analyze_results(phit, dt, nt, vx, vy, gi, ky, kz, epsn, eta, tau,
                        nvx, nvy, vxmax, vymax, runtime, wr, wi)

    return phit, gi, ge, runtime


def analyze_results(phit, dt, nt, vx, vy, gi, ky, kz, epsn, eta, tau,
                    nvx, nvy, vxmax, vymax, runtime, wr, wi):
    plt.figure()

    t = np.arange(1, nt+1) * dt

    plt.subplot(221)
    plt.plot(t, np.real(phit), t, np.imag(phit), 'r--', linewidth=2)
    plt.xlabel(f't, runtime={runtime:.2f}s')
    plt.legend(['Re$\phi$', 'Im$\phi$'])
    plt.title(f'(a) $k_\perp$={ky}, $k_z$={kz}, $\epsilon_n$={epsn}, $\eta_i$={eta}, $\\tau$={tau}')

    lndEr = safe_log(np.abs(np.real(phit[:nt])))  # log(|Re(phi)|)
    lndEi = safe_log(np.abs(np.imag(phit[:nt])))  # log(|Im(phi)|)
    lndEa = safe_log(np.abs(phit[:nt]))  # log(|phi|)

    it0 = int(nt * 5.8 / 20)
    it1 = int(nt * 19.9 / 20)
    yy = lndEr[it0:it1]

    plt.subplot(222)
    plt.plot(t, lndEr, 'b', t, lndEi, 'b--', t, lndEa, 'k:', linewidth=2)

    withpeak = 1  # 1 - пики, 0 - весь диапазон

    if len(yy) > 1:
        y2, y1, t2, t1 = 0, 0, 0, 0
        if withpeak == 1:
            # extrMaxIndex, _ = find_peaks(yy)
            extrMaxIndex = np.where(np.diff(np.sign(np.diff(yy))) == -2)[0] + 1

            if len(extrMaxIndex) >= 1:
                t1 = t[it0 + extrMaxIndex[0]]
                t2 = t[it0 + extrMaxIndex[-1]]
                y1 = yy[extrMaxIndex[0]]
                y2 = yy[extrMaxIndex[-1]]
                nw = len(extrMaxIndex) - 1
                omega = np.pi / ((t2 - t1) / nw) if nw > 0 else 0
            else:
                t1 = t[it0]
                t2 = t[it1]
                y1 = yy[0]
                y2 = yy[-1]
                nw = 0
                omega = 0
        else:
            t1 = t[it0]
            t2 = t[it1]
            y1 = yy[0]
            y2 = yy[-1]
            nw = 0
            omega = 0

        gammas = (np.real(y2) - np.real(y1)) / (t2 - t1)

        plt.plot([t1, t2], [y1, y2], 'r*--', linewidth=2)
        title = f'(b) $\omega$={omega:.3f}, $\gamma$={gammas:.3f}'
        if withpeak == 1:
            title += f', nw={nw}'
        plt.title(title)
        plt.xlabel(f'$\omega^T$ = {wr + 1j * wi}')
        # plt.xlabel(f't')
        # plt.ylabel('log(|$\phi$|)')

    plt.subplot(223)
    plt.contourf(vx, vy, np.real(gi), 30, cmap="jet")
    plt.colorbar()
    plt.xlabel('$v_{||}$')
    plt.ylabel('$v_\perp$')
    plt.title(f'(c) Re $g_i$, nvx={nvx}, nvy={nvy}')

    plt.subplot(224)
    plt.contourf(vx, vy, np.imag(gi), 30, cmap="jet")
    plt.colorbar()
    plt.xlabel('$v_{||}$')
    plt.ylabel('$v_\perp$')
    plt.title(f'(d) Im $g_i$, vxmax={vxmax}, vymax={vymax}')

    plt.tight_layout()
    plt.savefig(f"rezults/entropy_ivp.png")
    plt.show()


if __name__ == "__main__":
    phit, gi, ge, runtime = run(real_time_plot=True)
