import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import time

from dr import fun_entropy_dr


def solve_entropy_mode():
    start_time = time.time()

    tau = 1.0
    epsn = 0.2
    kapn = 1 / epsn
    kapt = 0.1 * kapn
    # eta = kapt / kapn

    kz = 0.0
    mi = 1836.0

    kk = np.arange(0.1, 2.1, 0.1)
    x0 = np.array([0.01, 0.03])

    wk = np.zeros(len(kk), dtype=complex)

    tol = 1e-6
    xmax = 1.0e1
    ymax = 1.0e1

    for jk, k in enumerate(kk):
        wdi = k
        wde = -wdi * tau
        ki = k
        ke = -k * np.sqrt(tau / mi)
        kzi = kz
        kze = kz * np.sqrt(tau * mi)

        def fdr_real_imag(x):
            w = x[0] + 1j*x[1]
            ion_part = fun_entropy_dr(w, ki, wdi, kapn, kapt, kzi, tol, xmax, ymax)
            electron_part = (1 / tau) * fun_entropy_dr(w, ke, wde, kapn, kapt, kze, tol, xmax, ymax)
            result = ion_part + electron_part
            return [result.real, result.imag]

        x = fsolve(fdr_real_imag, x0, xtol=1e-10, maxfev=1000)
        x0 = x
        wk[jk] = x[0] + 1j*x[1]

    runtime = time.time() - start_time

    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.plot(kk, np.real(wk), 'bd--', linewidth=2)
    plt.xlabel('kρ_i')
    plt.ylabel('ω_r R/v_{ti}')
    plt.xlim([0, max(kk)])
    plt.title(f'Entropy mode, runtime={runtime:.2f}s')

    plt.subplot(122)
    plt.plot(kk, np.imag(wk), 'bd--', linewidth=2)
    plt.xlim([0, max(kk)])
    plt.xlabel('kρ_i')
    plt.ylabel('ω_i R/v_{ti}')

    # params = f'epsn={epsn}, eta={eta:.1f}, tau={tau}, kz={kz}'
    params = f'epsn={epsn}, tau={tau}, kz={kz}'
    plt.title(params)

    plt.tight_layout()
    plt.savefig(f'rezults/disp_relation_epsn={epsn},tau={tau},kz={kz}.png')
    plt.show()


    return kk, wk, runtime


if __name__ == "__main__":
    kk, wk, runtime = solve_entropy_mode()
