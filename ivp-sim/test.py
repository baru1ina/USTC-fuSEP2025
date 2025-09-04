import numpy as np
from core import run

from utils import compute_Fp
from plotting import plot_results, plot_gamma_vs_kz, plot_gamma_vs_ftrap

def scan_gamma_vs_kz(ftrap_vals, ky,
        wr,
        wi, epsn=0.2, kapt=0.5, kz_vals=np.linspace(0.0, 1.0, 11)):
    rez = []
    for ftrap in ftrap_vals:
        gamma_vals = []
        for kz in kz_vals:
            print(f"Running: kz = {kz:.3f}, f_trap = {ftrap}")
            _, _, _, _, gamma, omega = run(ky=ky, wr=wr, wi=wi, nt=500, epsn=epsn, kapt=kapt, kz=kz, f_trap=ftrap, plot_results=True)
            gamma_vals.append(gamma)
        rez.append(gamma_vals)
    return kz_vals, rez, ftrap_vals


def scan_gamma_vs_ftrap(ky,
        wr,
        wi, epsn=0.2, kapt=0.5, kz=0.0, ftrap_vals=np.linspace(0.0, 1.0, 11)):
    gamma_vals = []
    for f_trap in ftrap_vals:
        print(f"Running: f_trap = {f_trap:.3f}, kz = {kz}")
        _, _, _, _, gamma, omega = run(ky=ky, wr=wr, wi=wi, nt=500, epsn=epsn, kapt=kapt, kz=kz, f_trap=f_trap, plot_results=True)
        gamma_vals.append(gamma)
    return ftrap_vals, gamma_vals


def main():
    data = np.array(
        [
            [0.1, 0.1277 + 2.8258j],
            [0.2, 0.2487 + 2.8211j],
            [0.5, 0.6369 + 2.6664j],
            [1.0, 1.3108 + 2.2697j],
            [1.5, 2.1182 + 1.9633j],
            [2.0, 3.1047 + 1.9143j],
        ]
    )
    id = 3
    ky = float(data[id, 0])
    wr = float(np.real(data[id, 1]))
    wi = float(np.imag(data[id, 1]))


    f_trap_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    kapts = np.linspace(0, 10, 10)
    epsns = np.linspace(0.1, 1.0, 10)

    results = []

    for epsn_ in epsns:
        Fp_values = []
        gamma_for_epsn = []

        for f_trap in f_trap_values:
            gamma_for_f_trap = []
            for kapt_ in kapts:
                phit, gi, ge, runtime, gammas_, omega = run(ky=ky, wr=wr, wi=wi, kapt=kapt_, epsn=epsn_, f_trap=f_trap, plot_results=True)
                gamma_for_f_trap.append(gammas_)
            gamma_for_epsn.append(gamma_for_f_trap)

        for kapt_ in kapts:
            Fp = compute_Fp(epsn_, kapt_)
            Fp_values.append(Fp)

        results.append({
            'epsn': epsn_,
            'Fp_values': Fp_values,
            'gamma_values': gamma_for_epsn
        })

    for result in results:
        plot_results(Fp_values=result['Fp_values'], gamma_values_list=result['gamma_values'],
            param_list=f_trap_values, epsn=result['epsn'], param_name="f_trap")


def main2():
    data = np.array(
        [
            [0.1, 0.1277 + 2.8258j],
            [0.2, 0.2487 + 2.8211j],
            [0.5, 0.6369 + 2.6664j],
            [1.0, 1.3108 + 2.2697j],
            [1.5, 2.1182 + 1.9633j],
            [2.0, 3.1047 + 1.9143j],
        ]
    )
    id = 3
    ky = float(data[id, 0])
    wr = float(np.real(data[id, 1]))
    wi = float(np.imag(data[id, 1]))


    epsn = 0.2
    kapt = 0.5

    N = 3

    kz_vals = np.linspace(0.0, 0.4, N)
    ftrap_vals = np.linspace(0.0, 1.0, N)

    gamma_kz = scan_gamma_vs_kz(ky=ky, wr=wr, wi=wi, epsn=epsn, ftrap_vals=ftrap_vals, kapt=kapt, kz_vals=kz_vals)
    plot_gamma_vs_kz(*gamma_kz, epsn=epsn)

    # ftrap_vals = np.linspace(0.0, 1.0, 10)
    # gamma_ftrap = scan_gamma_vs_ftrap(epsn=epsn, kapt=kapt, kz=0.1, ftrap_vals=ftrap_vals)
    # plot_gamma_vs_ftrap(*gamma_ftrap, epsn=epsn, kz=0.1)


if __name__ == "__main__":
    main2()