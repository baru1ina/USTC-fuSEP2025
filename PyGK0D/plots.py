import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import jv

from Params import *


def plot_integrand(k=0.5, w=0.1 + 0.03j, wd=0.5, kapn=5.0, kapt=0.5, kz=0.0,
                   xmax=10.0, ymax=10.0, nx=100, ny=100):
    """
    Параметры:
    k, w, wd, kapn, kapt, kz - физические параметры
    xmax, ymax - пределы интегрирования
    nx, ny - количество точек для построения графиков
    """

    x = np.linspace(-xmax if kz != 0 else 0, xmax, nx)
    y = np.linspace(0, ymax, ny)
    X, Y = np.meshgrid(x, y)

    denominator = w - kz * X - wd * (X ** 2 + Y ** 2 / 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = (jv(0, k * np.abs(Y)) ** 2 * (w - wd * (kapn + kapt * ((X ** 2 + Y ** 2) / 2 - 3 / 2))) *
             np.abs(Y) / denominator * np.exp(-(X ** 2 + Y ** 2) / 2))
        Z = np.nan_to_num(Z)

        plt.figure(figsize=(15, 10))

        # plt.subplot(2, 2, 1)
        y_idx = ny // 2
        plt.plot(x, Z[y_idx, :])
        plt.title(f'Срез по x при y={y[y_idx]:.2f}')
        plt.xlabel('x')
        plt.ylabel('f(x,y)')
        plt.grid(True)

        # plt.subplot(2, 2, 2)
        plt.figure(figsize=(15, 10))
        x_idx = nx // 2
        plt.plot(y, Z[:, x_idx])
        plt.title(f'Срез по y при x={x[x_idx]:.2f}')
        plt.xlabel('y')
        plt.ylabel('f(x,y)')
        plt.grid(True)
        #
        # plt.subplot(2, 2, 3)
        # plt.imshow(np.abs(Z), extent=[x.min(), x.max(), y.min(), y.max()],
        #            aspect='auto', origin='lower', cmap='viridis')
        # plt.colorbar(label='|f(x,y)|')
        # plt.title('2D тепловая карта (модуль)')
        # plt.xlabel('x')
        # plt.ylabel('y')

        # plt.subplot(2, 2, 4, projection='3d')
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 1, 1, projection='3d')
        surf = plt.gca().plot_surface(X, Y, np.abs(Z), cmap=cm.viridis,
                                      linewidth=0, antialiased=False)
        plt.colorbar(surf, shrink=0.5, aspect=5, label='|f(x,y)|')
        plt.title('3D поверхность (модуль)')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.tight_layout()
        plt.show()

        print(f"Параметры:\nw={w}\nk={k}\nwd={wd}\nkapn={kapn}\nkapt={kapt}\nkz={kz}")


if __name__ == "__main__":
    # tau = 1.0
    # epsn = 0.2
    # kapn = 1 / epsn
    # kapt = 0.1 * kapn
    # kz = 0.0
    # mi = 1836.0
    k = 0.5
    wdi = k
    w = 0.1 + 0.03j

    plot_integrand(k=k, w=w, wd=wdi, kapn=kapn, kapt=kapt, kz=kz)
