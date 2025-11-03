import numpy as np
from core import IVPModel
from numba import njit


@njit
def compute_Fp(epsn, kapt):
    kapn = 1 / epsn
    return kapn / (kapn + kapt)


class ModelScanner:
    """
    Универсальный сканер параметров модели.
    Каждая scan_* функция возвращает словарь с массивами данных.
    """
    def __init__(self, base_params: dict):
        self.base_params = base_params

    def _make_model(self, **kwargs) -> IVPModel:
        params = self.base_params.copy()
        params.update(kwargs)
        model = IVPModel(**params)
        return model

    def scan_gamma_vs_kz(self, kz_vals, ftrap_vals, nt=500):
        results = []
        for f_trap in ftrap_vals:
            gamma_list = []
            for kz in kz_vals:
                model = self._make_model(f_trap=f_trap, kz=kz)
                model.initialize()
                _, _, _, _, gamma, _ = model.run(nt=nt, plot_results=False)
                gamma_list.append(gamma)
            results.append(gamma_list)
        return {"x_vals": kz_vals, "y_vals": results, "param_vals": ftrap_vals, "xlabel": "kz", "ylabel": "gamma"}

    def scan_gamma_vs_ftrap(self, ftrap_vals, nt=500):
        gamma_list = []
        for f_trap in ftrap_vals:
            model = self._make_model(f_trap=f_trap)
            model.initialize()
            _, _, _, _, gamma, _ = model.run(nt=nt, plot_results=False)
            gamma_list.append(gamma)
        return {"x_vals": ftrap_vals, "y_vals": [gamma_list], "param_vals": [None], "xlabel": "f_trap", "ylabel": "gamma"}

    def scan_gamma_vs_Fp(self, epsn_vals, kapt_vals, ftrap_vals, nt=500):
        all_datasets = []

        for epsn in epsn_vals:
            Fp_vals = [compute_Fp(epsn, k) for k in kapt_vals]

            # Для каждого epsn делаем отдельный набор кривых по f_trap
            gamma_sets = []
            for f_trap in ftrap_vals:
                gamma_list = []
                for kapt in kapt_vals:
                    model = self._make_model(epsn=epsn, kapt=kapt, f_trap=f_trap)
                    model.initialize()
                    _, _, _, _, gamma, _ = model.run(nt=nt, plot_results=False)
                    gamma_list.append(gamma)
                gamma_sets.append(gamma_list)

            # Добавляем в список один словарь в формате plot_dependency
            all_datasets.append({
                "x_vals": Fp_vals,
                "y_vals": gamma_sets,
                "param_vals": ftrap_vals,
                "xlabel": "Fp",
                "ylabel": "gamma",
                "title_prefix": f"epsn={epsn:.3f}"
            })

        return all_datasets

    def scan_gamma_vs_tau_f(self, tau_f_vals, nt=500):
        gamma_list = []
        for tau_f in tau_f_vals:
            model = self._make_model(tau_f=tau_f)
            model.initialize()
            _, _, _, _, gamma, _ = model.run(nt=nt, plot_results=False)
            gamma_list.append(gamma)
        return {
            "x_vals": tau_f_vals,
            "y_vals": [gamma_list],
            "param_vals": [None],
            "xlabel": "tau_f (Ti/Tf)",
            "ylabel": "gamma"
        }

