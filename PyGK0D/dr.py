import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv
from scipy.optimize import root
import logging
from colorama import init, Fore
from scipy.integrate import dblquad
from numba import njit, complex128, float64

init(autoreset=True)


# 2025-07-08 10:11:23 CST Sofia

# В коде реализуется вычисление двумерного интеграла с использованием
# адаптивного метода Симпсона для расчета энтропийной моды
#
# Входные параметры:
#   w - массив частот,
#   k - волновое число,
#   wd - частота дрейфа,
#   kapn, kapt - параметры, связанные с градиентами плотности и температуры
#   kz - компонента волнового вектора вдоль магнитного поля
#   tol - допуск (по умолчанию 1e-10)
#   xmax, ymax - пределы интегрирования (по умолчанию 10)
#
# Функция f (подынтегральное выражение) включает:
#   Функцию Бесселя нулевого порядка jv(0,k*abs(y)) в квадрате
#   Комбинацию физических параметров
#   Экспоненциальный множитель np.exp(-(x ** 2 + y_abs ** 2) / 2)
#
# Логика вычислений:
#   Для каждого значения частоты w[jw] вычисляется двумерный интеграл
#   В зависимости от значения kz выбираются разные пределы интегрирования:
#   При kz=0 интегрирование по x от 0 до xmax
#   При kz!=0 интегрирование по x от -xmax до xmax
#   Для численного интегрирования используется функция dblquad из библиотеки scipy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@njit(complex128(complex128, float64, float64, float64, float64, float64, float64, float64))
def integrand(w, x, y, wd, kapn, kapt, kz, jv_val):
    y_abs = np.abs(y)
    denominator = w - kz * x - wd * (x ** 2 + y_abs ** 2 / 2)

    if np.abs(denominator) < 1e-15:
        return 0.0 + 0.0j

    exp_val = np.exp(-(x ** 2 + y_abs ** 2) / 2)
    numerator = (w - wd * (kapn + kapt * ((x ** 2 + y_abs ** 2) / 2 - 1.5)))

    result = jv_val ** 2 * numerator * y_abs/denominator * exp_val
    return np.nan_to_num(result)


def fun_entropy_dr(w, k, wd, kapn, kapt, kz=0.0, tol=1e-6, xmax=10.0, ymax=10.0):

    if kz == 0:
        xmin, ymin = 0.0, 0.0
        factor = 2.0 / np.sqrt(2 * np.pi)
    else:
        xmin, ymin = -xmax, 0.0
        factor = 1.0 / np.sqrt(2 * np.pi)

    w = np.atleast_1d(w)
    fdr = np.zeros_like(w, dtype=complex)

    for jw, w_val in enumerate(w):
        logger.info(f"\nProcessing w[{jw}] = {w_val}")

        def integrand_wrapper(y, x):
            jv_val = jv(0, k * np.abs(y))
            return integrand(w_val, x, y, wd, kapn, kapt, kz, jv_val)

        # Разделяем на реальную и мнимую части
        def real_part(y, x):
            return integrand_wrapper(y, x).real

        def imag_part(y, x):
            return integrand_wrapper(y, x).imag

        integral_real, error_real = dblquad(
            real_part,
            xmin, xmax,
            lambda x: ymin, lambda x: ymax,
            epsabs=tol
        )

        logger.info(f"Real part integration result: {integral_real:.6e} ± {error_real:.2e}")

        if np.iscomplexobj(w_val):
            integral_imag, error_imag = dblquad(
                imag_part,
                xmin, xmax,
                lambda x: ymin, lambda x: ymax,
                epsabs=tol
            )
            integral = integral_real + 1j * integral_imag
            logger.info(f"Imaginary part result: {integral_imag:.6e} ± {error_imag:.2e}")
        else:
            integral = integral_real

        fdr[jw] = 1.0 - factor * integral
        logger.info(f"Computed fdr value: {fdr[jw]}")

    return fdr[0] if len(w) == 1 else fdr

