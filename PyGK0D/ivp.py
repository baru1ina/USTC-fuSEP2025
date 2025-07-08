import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, iv
import time


irun = 0
nt = 500
runtimea = time.time()
it = 0

if irun == 0:
    data = np.array([
        [0.1, 0.1277 + 2.8258j],
        [0.2, 0.2487 + 2.8211j],
        [0.5, 0.6369 + 2.6664j],
        [1.0, 1.3108 + 2.2697j],
        [1.5, 2.1182 + 1.9633j],
        [2.0, 3.1047 + 1.9143j]
    ])

    id = 4
    ky = data[id, 0]  # k_perp
    wr = data[id, 1].real
    wi = data[id, 1].imag

    tau = 1.0
    epsn = 0.2
    kapn = 1 / epsn  # R/L_n
    kapt = 0.1 * kapn  # R/L_T
    eta = kapt / kapn

    kz = 0.0  # k_z*R
    mi = 1836.0  # mi/me

    wdi = ky  # *v_ti/R
    wde = -wdi * tau
    ki = ky
    ke = -ky * np.sqrt(tau / mi)
    kzi = kz
    kze = kz * np.sqrt(tau * mi)
    Gamma0i = iv(0, ki ** 2) * np.exp(-ki ** 2)
    Gamma0e = iv(0, ke ** 2) * np.exp(-ke ** 2)
    G0coef = 1.0 / ((1.0 - Gamma0i) + (1.0 - Gamma0e) / tau) / np.sqrt(2.0 * np.pi)

    nvx = 32
    nvy = 64
    dt = 0.02
    vxmax = 5.0
    vxmin = -vxmax
    vymax = 5.0
    vymin = 0.0

    dvx = (vxmax - vxmin) / nvx
    dvy = (vymax - vymin) / nvy

    Vx = np.linspace(vxmin, vxmax, nvx + 1)
    Vy = np.linspace(vymin, vymax, nvy + 1)
    vx, vy = np.meshgrid(Vx, Vy)

    wDiv = wdi * (vx ** 2 + vy ** 2 / 2) + kzi * vx
    wTiv = wdi * (kapn + (0.5 * (vx ** 2 + vy ** 2) - 1.5) * kapt)
    wDev = wde * (vx ** 2 + vy ** 2 / 2) + kze * vx
    wTev = wde * (kapn + (0.5 * (vx ** 2 + vy ** 2) - 1.5) * kapt)

    F0 = np.exp(-0.5 * (vx ** 2 + vy ** 2))
    gi = 0.001 * F0
    ge = 0.001 * F0

    phi = 0.1
    phit = np.zeros(nt, dtype=complex)
    rk = 1
    nta = 1
    ntb = nt
else:
    # nta = it + 1
    # ntb = nta + nt - 1
    nta = 1
    ntb = nt - 1

plt.figure(figsize=(12, 6))
for it in range(nta - 1, ntb):
    if rk == 0:
        phi = G0coef * np.sum((gi + ge / tau) * vy) * dvx * dvy
        gi = gi - 1j * (wDiv * gi + (wDiv - wTiv) * phi * j0(ki * vy) ** 2 * F0) * dt
        ge = ge - 1j * (wDev * ge + (wDev - wTev) * phi * j0(ke * vy) ** 2 * F0) * dt
    else:
        dgi1 = -1j * (wDiv * gi + (wDiv - wTiv) * phi * j0(ki * vy.astype(np.float64)) ** 2 * F0)
        dge1 = -1j * (wDev * ge + (wDev - wTev) * phi * j0(ke * vy.astype(np.float64)) ** 2 * F0)

        # RK-4, 2nd step
        gitmp = gi + 0.5 * dt * dgi1
        getmp = ge + 0.5 * dt * dge1
        phitmp = G0coef * np.sum((gitmp + getmp / tau) * vy) * dvx * dvy
        dgi2 = -1j * (wDiv * gitmp + (wDiv - wTiv) * phitmp * j0(ki * vy.astype(np.float64)) ** 2 * F0)
        dge2 = -1j * (wDev * getmp + (wDev - wTev) * phitmp * j0(ke * vy.astype(np.float64)) ** 2 * F0)

        # RK-4, 3rd step
        gitmp = gi + 0.5 * dt * dgi2
        getmp = ge + 0.5 * dt * dge2
        phitmp = G0coef * np.sum((gitmp + getmp / tau) * vy) * dvx * dvy
        dgi3 = -1j * (wDiv * gitmp + (wDiv - wTiv) * phitmp * j0(ki * vy.astype(np.float64)) ** 2 * F0)
        dge3 = -1j * (wDev * getmp + (wDev - wTev) * phitmp * j0(ke * vy.astype(np.float64)) ** 2 * F0)

        # RK-4, 4th step
        gitmp = gi + dt * dgi3
        getmp = ge + dt * dge3
        phitmp = G0coef * np.sum((gitmp + getmp / tau) * vy) * dvx * dvy
        dgi4 = -1j * (wDiv * gitmp + (wDiv - wTiv) * phitmp * j0(ki * vy.astype(np.float64)) ** 2 * F0)
        dge4 = -1j * (wDev * getmp + (wDev - wTev) * phitmp * j0(ke * vy.astype(np.float64)) ** 2 * F0)

        # RK-4, push
        gi = gi + dt / 6.0 * (dgi1 + 2.0 * dgi2 + 2.0 * dgi3 + dgi4)
        ge = ge + dt / 6.0 * (dge1 + 2.0 * dge2 + 2.0 * dge3 + dge4)

        phi = G0coef * np.sum((gi + ge / tau) * vy) * dvx * dvy

    phit[it] = phi

    if (it + 1) % 10 == 0:
        plt.subplot(121)
        plt.plot(np.arange(1, it + 2) * dt, np.real(phit[:it + 1]),
                 np.arange(1, it + 2) * dt, np.imag(phit[:it + 1]), 'r--', linewidth=2)
        plt.xlabel('t')
        plt.legend(['Re$\phi$', 'Im$\phi$'])
        plt.title(f'$\phi$, it={it + 1}/{ntb}')

        plt.subplot(122)
        plt.plot(np.arange(1, it + 2) * dt, np.log(np.real(phit[:it + 1])), linewidth=2)
        plt.xlabel('t')
        plt.ylabel('log(phit)')

        plt.draw()
        plt.pause(0.01)

runtime = time.time() - runtimea

plt.close('all')
fig = plt.figure(figsize=(12, 14))

t = np.arange(1, ntb + 1) * dt

plt.subplot(221)
plt.plot(t, np.real(phit), t, np.imag(phit), 'r--', linewidth=2)
plt.xlabel(f't, runtime={runtime:.2f}s')
plt.legend(['Re$\phi$', 'Im$\phi$'])
plt.title(f'(a) $k_\perp$={ky}, $k_z$={kz}, $\epsilon_n$={epsn}, $\eta_i$={eta}, $\\tau$={tau}')

lndE = np.log(np.real(phit[:ntb]))
lndEi = np.log(np.imag(phit[:ntb]))
lndEa = np.log(np.abs(phit[:ntb]))
it0 = int(ntb * 5.8 / 20)
it1 = int(ntb * 19.9 / 20)
yy = lndE[it0:it1]

plt.subplot(222)
plt.plot(t, lndE, t, lndEi, 'b--', t, lndEa, 'k:', linewidth=2)
withpeak = 1
if withpeak == 1:
    dyy = np.diff(yy)
    extrMaxIndex = np.where(np.diff(np.sign(dyy)) == -2)[0] + 1
    t1 = t[it0 + extrMaxIndex[0]]
    t2 = t[it0 + extrMaxIndex[-1]]
    y1 = yy[extrMaxIndex[0]]
    y2 = yy[extrMaxIndex[-1]]
    nw = len(extrMaxIndex) - 1
    omega = np.pi / ((t2 - t1) / nw)
else:
    t1 = t[it0]
    t2 = t[it1]
    y1 = yy[0]
    y2 = yy[-1]
    nw = 0
    omega = 0

plt.plot([t1, t2], [y1, y2], 'r*--', linewidth=2)
gammas = (y2.real - y1.real) / (t2 - t1)
plt.title(f'(b) $\omega^S$={omega}, $\gamma^S$={gammas}, nw={nw}')
plt.xlabel(f'$\omega^T$={wr + 1j * wi}')

plt.subplot(223)
plt.contourf(vx, vy, np.real(gi), 30)
plt.xlabel('$v_{||}$')
plt.ylabel('$v_\perp$')
plt.title(f'(c) Re $g_i$, nvx={nvx}, nvy={nvy}')

plt.subplot(224)
plt.contourf(vx, vy, np.imag(gi), 30)
plt.xlabel('$v_{||}$')
plt.ylabel('$v_\perp$')
plt.title(f'(d) Im $g_i$, vxmax={vxmax}, vymax={vymax}')

str = (f'mgk0d_ivp_ky={ky},epsn={epsn},eta={eta},tau={tau},kz={kz},'
       f'vxmax={vxmax},vymax={vymax},nvx={nvx},nvy={nvy},dt={dt}')

plt.tight_layout()
plt.show()