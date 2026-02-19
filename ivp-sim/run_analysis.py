import numpy as np
from analysis.manager import AnalysisManager

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


# ky=ky, wr=wr, wi=wi
base_params = dict(
    ky=1.0, wr=3.1047, wi=1.9143,
    tau=1.0, epsn=0.2, kz=0.0, kapt=0.5,
    f_trap=1.0, f_fast=0.01, tau_f=12.0, eta_f=0.6
)

manager = AnalysisManager(base_params)

print("Доступные типы анализа:")
print(manager.list_available())
# Пример: ['gamma_vs_kz', 'gamma_vs_ftrap', 'gamma_vs_Fp', 'gamma_vs_tau_f']

f_trap_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
kapts = np.linspace(0, 10, 10)
epsns = np.linspace(0.1, 1.0, 10)

manager.run_analysis("gamma_vs_tau_f", tau_f_vals=np.linspace(7, 40, 30), nt=500)
# manager.run_analysis("gamma_vs_Fp", epsn_vals=epsns, kapt_vals=kapts, ftrap_vals=f_trap_values, nt=500)
# manager.run_analysis("gamma_vs_ftrap", ftrap_vals=f_trap_values, nt=1000)