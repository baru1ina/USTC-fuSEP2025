import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, iv
import time
import math
from numba import njit
from typing import Tuple, Optional

@njit('float64[:](float64[:])')
def safe_log(x):
    x = np.asarray(x)
    x_safe = np.where(x <= 0, 1e-20, x)
    return np.log(x_safe)


@njit
def compute_Fp(epsn, kapt):
    kapn = 1 / epsn
    Fp = kapn / (kapn + kapt)
    return Fp


mi = 1836.0


class IVPModel:
    """
    Initial Value Problem Model for plasma physics simulations.
    Solves the gyrokinetic equations for trapped electron modes.
    Includes fast-ion (resonant) contribution.
    """
    def __init__(
            self,
            *,
            ky,
            wr,
            wi,
            tau: float = 1.0,
            f_trap: float = 1.0,
            epsn: float = 0.2,
            kz: float = 0.0,
            kapt: float = 0.5,
            nvx: int = 32,
            nvy: int = 64,
            dt: float = 0.02,
            vxmax: float = 5.0,
            vymax: float = 5.0,
            vymin: float = 0.0,
            vxmin: float = -5.0,
            # FAST IONS
            f_fast: float = 0.0,       # fast-ion density fraction (n_fast / n_total)
            tau_f: float = 12.0,      # T_fast / T_e
            eta_f: float = 1.0        # Ln/LT ratio for fast ions (controls drive)
    ) -> None:
        """
        Initialize the IVP model with physical and numerical parameters.

        Args:
            ky: Perpendicular wave number
            wr: Real part of complex frequency
            wi: Imaginary part of complex frequency (growth rate)
            tau: Temperature ratio Ti/Te
            f_trap: Trapped electron fraction
            epsn: Normalized density gradient
            kz: Parallel wave number
            kapt: Normalized temperature gradient
            nvx: Number of grid points in parallel velocity space
            nvy: Number of grid points in perpendicular velocity space
            dt: Time step size
            vxmax: Maximum parallel velocity
            vymax: Maximum perpendicular velocity
            vymin: Minimum perpendicular velocity
            vxmin: Minimum parallel velocity
        """
        # Physical parameters
        self.ky = float(ky)
        self.wr = float(wr)
        self.wi = float(wi)
        self.tau = float(tau)
        self.epsn = float(epsn)
        self.kz = float(kz)
        self.kapt = float(kapt)
        self.f_trap = float(f_trap)

        # FAST IONS
        self.f_fast = float(f_fast)   # fast ion fraction
        self.tau_f = float(tau_f)     # T_fast / T_e
        self.eta_f = float(eta_f)     # eta (L_n/L_T) for fast ions

        # Numerical parameters
        self.nvx = int(nvx)
        self.nvy = int(nvy)
        self.dt = float(dt)
        self.vxmax = float(vxmax)
        self.vymax = float(vymax)
        self.vymin = float(vymin)
        self.vxmin = float(vxmin)

        # Derived parameters
        self.kapn: Optional[float] = None  # Normalized density gradient
        self.eta: Optional[float] = None  # Ratio of temperature to density gradient
        self.wdi = None  # Ion diamagnetic frequency
        self.wde = None  # Electron diamagnetic frequency
        self.ki = None  # Ion wave number
        self.ke = None  # Electron wave number
        self.kzi = None  # Ion parallel wave number
        self.kze = None  # Electron parallel wave number
        self.G0coef = None  # Coefficient for field equation

        # Velocity space grids and operators
        self.vx = None  # Parallel velocity array
        self.vy = None  # Perpendicular velocity array
        self.vx_grid = None  # 2D parallel velocity grid
        self.vy_grid = None  # 2D perpendicular velocity grid
        self.J0ki2 = None  # Squared Bessel function for ions
        self.J0ke2 = None  # Squared Bessel function for electrons
        self.F0 = None  # Maxwellian distribution function for bulk
        self.wDiv = None  # Ion drift frequency operator
        self.wTiv = None  # Ion temperature gradient operator
        self.wDev = None  # Electron drift frequency operator
        self.wTev = None  # Electron temperature gradient operator
        self.dvx = None  # Parallel velocity grid spacing
        self.dvy = None  # Perpendicular velocity grid spacing

        # FAST IONS
        self.wFv = None
        self.wTFv = None
        self.J0kf2 = None
        self.F0f = None

        # Distribution functions and potential
        self.gi = None  # Ion distribution function perturbation
        self.ge = None  # Electron distribution function perturbation
        self.gf = None  # FAST IONS: fast-ion perturbation
        self.phit = None  # Time history of electrostatic potential
        self.runtime = None  # Computational runtime

    def set_params(self) -> None:
        """
        Calculate derived physical parameters from input parameters.
        Sets up the coefficients for the field equation.
        """
        if self.epsn == 0.0:
            raise ZeroDivisionError("Division by zero: epsn cannot be zero")
        self.kapn = 1.0 / self.epsn
        self.eta = self.kapt / self.kapn if self.kapn != 0.0 else np.inf

        self.wdi = self.ky
        self.wde = -self.wdi * self.tau

        self.ki = self.ky
        self.ke = -self.ky * np.sqrt(self.tau / mi)

        self.kzi = self.kz
        self.kze = self.kz * np.sqrt(self.tau * mi)

        Gamma0i = iv(0, self.ki ** 2) * np.exp(-self.ki ** 2)
        Gamma0e = iv(0, self.ke ** 2) * np.exp(-self.ke ** 2)
        denominator = (1.0 - Gamma0i) + (1.0 - Gamma0e) / self.tau
        if np.isclose(denominator, 0.0):
            raise ZeroDivisionError("Zero denominator in G0coef calculation")
        self.G0coef = 1.0 / denominator / np.sqrt(2.0 * np.pi)

    def _build_velocity_grid(self) -> None:
        """
        Create the velocity space grid for numerical integration.
        Sets up vx and vy arrays and their 2D meshgrid.
        """
        self.dvx = (self.vxmax - self.vxmin) / self.nvx
        self.dvy = (self.vymax - self.vymin) / self.nvy
        self.vx = np.arange(self.vxmin, self.vxmax + self.dvx / 2, self.dvx)
        self.vy = np.arange(self.vymin, self.vymax + self.dvy / 2, self.dvy)
        self.vx_grid, self.vy_grid = np.meshgrid(self.vx, self.vy)

    def _precompute_operators(self) -> None:
        """
        Precompute operators and functions needed for time integration.
        Includes drift frequencies, Bessel functions, and Maxwellian distribution.
        """
        vx, vy = self.vx_grid, self.vy_grid
        self.wDiv = self.wdi * (vx ** 2 + vy ** 2 / 2.0) + self.kzi * vx
        self.wTiv = self.wdi * (self.kapn + (0.5 * (vx ** 2 + vy ** 2) - 1.5) * self.kapt)
        self.wDev = self.wde * (vx ** 2 + vy ** 2 / 2.0) + self.kze * vx
        self.wTev = self.wde * (self.kapn + (0.5 * (vx ** 2 + vy ** 2) - 1.5) * self.kapt)

        self.J0ki2 = jv(0, self.ki * vy) ** 2
        self.J0ke2 = jv(0, self.ke * vy) ** 2

        self.F0 = np.exp(-0.5 * (vx ** 2 + vy ** 2))

        # FAST IONS
        # We represent fast ions on the same vx,vy grid but with temperature scaling tau_f.
        # Fast-ion drift frequency and gradient-operator are scaled with tau_f and eta_f.
        # This is a practical reduced-model embedding of the resonant effect.
        # NOTE: wFv is kept real-like (drift frequency operator); resonance handled via regularization later.
        self.wFv = self.ky * (vx ** 2 + vy ** 2 / 2.0) / self.tau_f + self.kz * vx
        # wTFv acts like the "temperature-gradient" driven operator for fast ions (re-using kapn/kapt scaled by eta_f)
        # This is a modeling choice consistent with the article's reduced-Vlasov form for drive.
        self.wTFv = self.ky * (self.kapn + (0.5 * (vx ** 2 + vy ** 2) - 1.5) * (self.kapt / max(1e-12, self.eta_f)))
        # Bessel for fast ions: characteristic k_perp for fast species scales with sqrt(tau_f*mi) (approx.)
        # Using same ky but scaled to mimic Larmor radius differences (reduced model)
        k_fast = self.ky / np.sqrt(self.tau_f * mi + 1e-12)
        self.J0kf2 = jv(0, k_fast * vy) ** 2
        # Maxwellian for fast ions with T_f: scale exponent by tau_f
        self.F0f = np.exp(-0.5 * (vx ** 2 + vy ** 2) / max(1e-12, self.tau_f))

    def _init_g(self) -> None:
        """
        Initialize the distribution function perturbations with small noise.
        """
        base = (0.001 * self.F0).astype(np.complex128)
        self.gi = base.copy()
        self.ge = base.copy()
        # FAST IONS
        # Fast-ion perturbation initialised similarly using F0f
        self.gf = (0.001 * self.F0f).astype(np.complex128)

    def initialize(self) -> None:
        """
        Complete initialization of the model by calling all setup methods.
        """
        self.set_params()
        self._build_velocity_grid()
        self._precompute_operators()
        self._init_g()

    def run(self,
            nt: int = 500,
            *,
            plot_results: bool = True,
            real_time_plot: bool = False,
            rk: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        """
        Run the time integration for the specified number of time steps.

        Returns:
            Tuple containing potential history, distribution functions, runtime, growth rate, and frequency
        """
        if self.G0coef is None or self.vx_grid is None:
            self.initialize()

        self.phit = np.zeros(nt, dtype=np.complex128)

        runtime_start = time.time()

        # pass wr, wi down to integrator to use for resonance regularization
        if real_time_plot:
            plt.figure(1, figsize=(12, 6))
            for it in range(nt):
                self.gi, self.ge, self.gf, phit_tmp = integrate_rk4_core(
                    self.gi,
                    self.ge,
                    self.gf,  # FAST IONS
                    self.phit,
                    nt,
                    self.dt,
                    self.wDiv,
                    self.wTiv,
                    self.wDev,
                    self.wTev,
                    self.wFv,    # FAST IONS
                    self.wTFv,   # FAST IONS
                    self.J0ki2,
                    self.J0ke2,
                    self.J0kf2,  # FAST IONS
                    self.F0,
                    self.F0f,    # FAST IONS
                    self.vy_grid,
                    self.dvx,
                    self.dvy,
                    self.G0coef,
                    self.f_trap,
                    self.f_fast,  # FAST IONS
                    self.tau,
                    self.tau_f,   # FAST IONS
                    self.wr,      # FAST IONS: pass wr, wi for resonance regularization
                    self.wi,
                    rk,
                    single_step=True,
                )
                phi = phit_tmp[0]
                self.phit[it] = phi

                if it % 10 == 9:
                    plt.figure(1, figsize=(12, 6))
                    plt.clf()
                    t_axis = np.arange(1, it + 2) * self.dt
                    plt.subplot(121)
                    plt.plot(t_axis, np.real(self.phit[: it + 1]), t_axis, np.imag(self.phit[: it + 1]), "r--",
                             linewidth=2)
                    plt.xlabel("t")
                    plt.legend(["Re$\\phi$", "Im$\\phi$"])
                    plt.title(f"$\\phi$, it={it + 1}/{nt}")

                    plt.subplot(122)
                    plt.plot(
                        t_axis,
                        safe_log(np.abs(np.real(self.phit[: it + 1]))),
                        "b",
                        t_axis,
                        safe_log(np.abs(np.imag(self.phit[: it + 1]))),
                        "b--",
                        t_axis,
                        safe_log(np.abs(np.abs(self.phit[: it + 1]))),
                        "k:",
                        linewidth=2,
                    )
                    plt.title(f"$\\omega^T$ = {self.wr + 1j * self.wi}")
                    plt.xlabel("t")
                    plt.ylabel("log(|$\\phi$|)")
                    plt.draw()
                    plt.pause(0.01)
        else:
            self.gi, self.ge, self.gf, self.phit = integrate_rk4_core(
                self.gi,
                self.ge,
                self.gf,  # FAST IONS
                self.phit,
                nt,
                self.dt,
                self.wDiv,
                self.wTiv,
                self.wDev,
                self.wTev,
                self.wFv,    # FAST IONS
                self.wTFv,   # FAST IONS
                self.J0ki2,
                self.J0ke2,
                self.J0kf2,  # FAST IONS
                self.F0,
                self.F0f,    # FAST IONS
                self.vy_grid,
                self.dvx,
                self.dvy,
                self.G0coef,
                self.f_trap,
                self.f_fast,  # FAST IONS
                self.tau,
                self.tau_f,   # FAST IONS
                self.wr,
                self.wi,
                rk,
                single_step=False,
            )

        self.runtime = time.time() - runtime_start

        gamma, omega_r = self.analyze_results(plot=plot_results)
        return self.phit, self.gi, self.ge, self.runtime, gamma, omega_r

    def analyze_results(self, *, plot: bool = True) -> Tuple[float, float]:
        """
        Analyze simulation results to extract growth rate and frequency.
        Optionally plots the results.

        Returns:
            Tuple of (growth rate, real frequency)
        """
        if self.phit is None:
            raise RuntimeError("Run the simulation first with run()")

        nt = self.phit.shape[0]
        t = np.arange(1, nt + 1) * self.dt

        lndEr = safe_log(np.abs(np.real(self.phit[:nt])))
        gamma, omega, t1, t2, y1, y2, nw = compute_gamma_omega_numba(lndEr, self.dt)

        if plot:
            plt.figure(figsize=(10, 8))

            plt.subplot(221)
            plt.plot(t, np.real(self.phit), t, np.imag(self.phit), "r--", linewidth=2)
            plt.xlabel(f"t, runtime={self.runtime:.2f}s")
            plt.legend(["Re$\\phi$", "Im$\\phi$"])
            plt.title(
                f"(a) $k_\\perp$={self.ky}, $k_z$={self.kz}, $\\epsilon_n$={self.epsn}, $\\eta_i$={self.eta}, $\\tau$={self.tau}"
            )

            lndEi = safe_log(np.abs(np.imag(self.phit[:nt])))
            lndEa = safe_log(np.abs(self.phit[:nt]))
            plt.subplot(222)
            plt.plot(t, lndEr, "b", t, lndEi, "b--", t, lndEa, "k:", linewidth=2)
            if not np.isnan(gamma):
                plt.plot([t1, t2], [y1, y2], "r*--", linewidth=2)
                title = f"(b) $\\omega$={omega:.3f}, $\\gamma$={gamma:.3f}, nw={int(nw)}"
            else:
                title = f"(b) estimation not available? moderate nt"
            plt.title(title)
            plt.xlabel(f"$\\omega^T$ = {self.wr + 1j * self.wi}")

            plt.subplot(223)
            plt.contourf(self.vx_grid, self.vy_grid, np.real(self.gi), 30, cmap="jet")
            plt.colorbar()
            plt.xlabel("$v_{||}$")
            plt.ylabel("$v_\\perp$")
            plt.title(f"(c) Re $g_i$, nvx={self.nvx}, nvy={self.nvy}")

            plt.subplot(224)
            plt.contourf(self.vx_grid, self.vy_grid, np.imag(self.gi), 30, cmap="jet")
            plt.colorbar()
            plt.xlabel("$v_{||}$")
            plt.ylabel("$v_\\perp$")

            plt.title(f"(d) Im $g_i$, vxmax={self.vxmax}, vymax={self.vymax}")

            plt.tight_layout()
            plt.show()
            plt.close()

        return float(gamma), float(omega)


@njit(cache=True, fastmath=True)
def integrate_rk4_core(
        gi,
        ge,
        gf,                # FAST IONS
        phit,
        nt,
        dt,
        wDiv,
        wTiv,
        wDev,
        wTev,
        wFv,               # FAST IONS
        wTFv,              # FAST IONS
        J0ki2,
        J0ke2,
        J0kf2,             # FAST IONS
        F0,
        F0f,               # FAST IONS
        vy,
        dvx,
        dvy,
        G0coef,
        f_trap,
        f_fast,            # FAST IONS
        tau,
        tau_f,             # FAST IONS
        wr,
        wi,
        rk,
        single_step=False,
):
    """
    Core integration function using either Euler or RK4 method.
    Now includes fast-ion dynamics and a resonance regularization to avoid division-by-zero in F1-like responses.
    """
    phi_local = 0.0 + 0.0j

    # resonance regularization parameter: small frequency width to avoid singular denominator
    # choose a floor to prevent zero; scaled by imaginary part of target mode if present
    omega_reg = max(1e-6, abs(wi) * 0.1)

    if single_step:
        phit_tmp = np.zeros(1, dtype=np.complex128)

        if rk == 0:
            # Euler method
            # compute a weight for fast-ions to regularize resonant enhancement:
            # fast_weight = 1/(1 + ((wr - wFv)/omega_reg)^2)  (Lorentz-like smoothing)
            fast_weight = 1.0 / (1.0 + ((wr - wFv) / omega_reg) ** 2)

            phi_local = G0coef * np.sum((gi + (1 - f_trap) * ge / tau + f_fast * gf / tau_f * fast_weight) * vy) * dvx * dvy

            gi[:] = gi - 1j * (wDiv * gi + (wDiv - wTiv) * phi_local * (J0ki2 * F0)) * dt
            ge[:] = ge - 1j * (wDev * ge + (wDev - wTev) * phi_local * (J0ke2 * F0)) * dt
            # FAST IONS
            gf[:] = gf - 1j * (wFv * gf + (wFv - wTFv) * phi_local * (J0kf2 * F0f)) * dt

            gi[:, 0] = 0.0 + 0.0j
            gi[:, -1] = 0.0 + 0.0j
            gi[0, :] = 0.0 + 0.0j
            gi[-1, :] = 0.0 + 0.0j
            ge[:, 0] = 0.0 + 0.0j
            ge[:, -1] = 0.0 + 0.0j
            ge[0, :] = 0.0 + 0.0j
            ge[-1, :] = 0.0 + 0.0j
            gf[:, 0] = 0.0 + 0.0j
            gf[:, -1] = 0.0 + 0.0j
            gf[0, :] = 0.0 + 0.0j
            gf[-1, :] = 0.0 + 0.0j
        else:
            # RK4 method
            # step 1
            dgi1 = -1j * (wDiv * gi + (wDiv - wTiv) * phi_local * (J0ki2 * F0))
            dge1 = -1j * (wDev * ge + (wDev - wTev) * phi_local * (J0ke2 * F0))
            dgf1 = -1j * (wFv * gf + (wFv - wTFv) * phi_local * (J0kf2 * F0f))

            gitmp = gi + 0.5 * dt * dgi1
            getmp = ge + 0.5 * dt * dge1
            gftmp = gf + 0.5 * dt * dgf1

            fast_weight = 1.0 / (1.0 + ((wr - wFv) / omega_reg) ** 2)
            phitmp = G0coef * np.sum((gitmp + (1 - f_trap) * getmp / tau + f_fast * gftmp / tau_f * fast_weight) * vy) * dvx * dvy

            # step 2
            dgi2 = -1j * (wDiv * gitmp + (wDiv - wTiv) * phitmp * (J0ki2 * F0))
            dge2 = -1j * (wDev * getmp + (wDev - wTev) * phitmp * (J0ke2 * F0))
            dgf2 = -1j * (wFv * gftmp + (wFv - wTFv) * phitmp * (J0kf2 * F0f))

            gitmp = gi + 0.5 * dt * dgi2
            getmp = ge + 0.5 * dt * dge2
            gftmp = gf + 0.5 * dt * dgf2

            fast_weight = 1.0 / (1.0 + ((wr - wFv) / omega_reg) ** 2)
            phitmp = G0coef * np.sum((gitmp + (1 - f_trap) * getmp / tau + f_fast * gftmp / tau_f * fast_weight) * vy) * dvx * dvy

            # step 3
            dgi3 = -1j * (wDiv * gitmp + (wDiv - wTiv) * phitmp * (J0ki2 * F0))
            dge3 = -1j * (wDev * getmp + (wDev - wTev) * phitmp * (J0ke2 * F0))
            dgf3 = -1j * (wFv * gftmp + (wFv - wTFv) * phitmp * (J0kf2 * F0f))

            gitmp = gi + dt * dgi3
            getmp = ge + dt * dge3
            gftmp = gf + dt * dgf3

            fast_weight = 1.0 / (1.0 + ((wr - wFv) / omega_reg) ** 2)
            phitmp = G0coef * np.sum((gitmp + (1 - f_trap) * getmp / tau + f_fast * gftmp / tau_f * fast_weight) * vy) * dvx * dvy

            # step 4
            dgi4 = -1j * (wDiv * gitmp + (wDiv - wTiv) * phitmp * (J0ki2 * F0))
            dge4 = -1j * (wDev * getmp + (wDev - wTev) * phitmp * (J0ke2 * F0))
            dgf4 = -1j * (wFv * gftmp + (wFv - wTFv) * phitmp * (J0kf2 * F0f))

            gi[:] = gi + dt / 6.0 * (dgi1 + 2.0 * dgi2 + 2.0 * dgi3 + dgi4)
            ge[:] = ge + dt / 6.0 * (dge1 + 2.0 * dge2 + 2.0 * dge3 + dge4)
            gf[:] = gf + dt / 6.0 * (dgf1 + 2.0 * dgf2 + 2.0 * dgf3 + dgf4)

            fast_weight = 1.0 / (1.0 + ((wr - wFv) / omega_reg) ** 2)
            phi_local = G0coef * np.sum((gi + f_trap * ge / tau + f_fast * gf / tau_f * fast_weight) * vy) * dvx * dvy

        phit_tmp[0] = phi_local
        return gi, ge, gf, phit_tmp

    else:
        for it in range(nt):
            if rk == 0:
                # Euler method
                fast_weight = 1.0 / (1.0 + ((wr - wFv) / omega_reg) ** 2)
                phi_local = G0coef * np.sum((gi + (1 - f_trap) * ge / tau + f_fast * gf / tau_f * fast_weight) * vy) * dvx * dvy
                gi[:] = gi - 1j * (wDiv * gi + (wDiv - wTiv) * phi_local * (J0ki2 * F0)) * dt
                ge[:] = ge - 1j * (wDev * ge + (wDev - wTev) * phi_local * (J0ke2 * F0)) * dt
                gf[:] = gf - 1j * (wFv * gf + (wFv - wTFv) * phi_local * (J0kf2 * F0f)) * dt

                gi[:, 0] = 0.0 + 0.0j
                gi[:, -1] = 0.0 + 0.0j
                gi[0, :] = 0.0 + 0.0j
                gi[-1, :] = 0.0 + 0.0j
                ge[:, 0] = 0.0 + 0.0j
                ge[:, -1] = 0.0 + 0.0j
                ge[0, :] = 0.0 + 0.0j
                ge[-1, :] = 0.0 + 0.0j
                gf[:, 0] = 0.0 + 0.0j
                gf[:, -1] = 0.0 + 0.0j
                gf[0, :] = 0.0 + 0.0j
                gf[-1, :] = 0.0 + 0.0j
            else:
                # RK4 method
                dgi1 = -1j * (wDiv * gi + (wDiv - wTiv) * phi_local * (J0ki2 * F0))
                dge1 = -1j * (wDev * ge + (wDev - wTev) * phi_local * (J0ke2 * F0))
                dgf1 = -1j * (wFv * gf + (wFv - wTFv) * phi_local * (J0kf2 * F0f))

                gitmp = gi + 0.5 * dt * dgi1
                getmp = ge + 0.5 * dt * dge1
                gftmp = gf + 0.5 * dt * dgf1
                fast_weight = 1.0 / (1.0 + ((wr - wFv) / omega_reg) ** 2)
                phitmp = G0coef * np.sum((gitmp + (1 - f_trap) * getmp / tau + f_fast * gftmp / tau_f * fast_weight) * vy) * dvx * dvy

                dgi2 = -1j * (wDiv * gitmp + (wDiv - wTiv) * phitmp * (J0ki2 * F0))
                dge2 = -1j * (wDev * getmp + (wDev - wTev) * phitmp * (J0ke2 * F0))
                dgf2 = -1j * (wFv * gftmp + (wFv - wTFv) * phitmp * (J0kf2 * F0f))

                gitmp = gi + 0.5 * dt * dgi2
                getmp = ge + 0.5 * dt * dge2
                gftmp = gf + 0.5 * dt * dgf2
                fast_weight = 1.0 / (1.0 + ((wr - wFv) / omega_reg) ** 2)
                phitmp = G0coef * np.sum((gitmp + (1 - f_trap) * getmp / tau + f_fast * gftmp / tau_f * fast_weight) * vy) * dvx * dvy

                dgi3 = -1j * (wDiv * gitmp + (wDiv - wTiv) * phitmp * (J0ki2 * F0))
                dge3 = -1j * (wDev * getmp + (wDev - wTev) * phitmp * (J0ke2 * F0))
                dgf3 = -1j * (wFv * gftmp + (wFv - wTFv) * phitmp * (J0kf2 * F0f))

                gitmp = gi + dt * dgi3
                getmp = ge + dt * dge3
                gftmp = gf + dt * dgf3
                fast_weight = 1.0 / (1.0 + ((wr - wFv) / omega_reg) ** 2)
                phitmp = G0coef * np.sum((gitmp + (1 - f_trap) * getmp / tau + f_fast * gftmp / tau_f * fast_weight) * vy) * dvx * dvy

                dgi4 = -1j * (wDiv * gitmp + (wDiv - wTiv) * phitmp * (J0ki2 * F0))
                dge4 = -1j * (wDev * getmp + (wDev - wTev) * phitmp * (J0ke2 * F0))
                dgf4 = -1j * (wFv * gftmp + (wFv - wTFv) * phitmp * (J0kf2 * F0f))

                gi[:] = gi + dt / 6.0 * (dgi1 + 2.0 * dgi2 + 2.0 * dgi3 + dgi4)
                ge[:] = ge + dt / 6.0 * (dge1 + 2.0 * dge2 + 2.0 * dge3 + dge4)
                gf[:] = gf + dt / 6.0 * (dgf1 + 2.0 * dgf2 + 2.0 * dgf3 + dgf4)

                fast_weight = 1.0 / (1.0 + ((wr - wFv) / omega_reg) ** 2)
                phi_local = G0coef * np.sum((gi + (1 - f_trap) * ge / tau + f_fast * gf / tau_f * fast_weight) * vy) * dvx * dvy

            phit[it] = phi_local

        return gi, ge, gf, phit


@njit(cache=True)
def compute_gamma_omega_numba(lndEr, dt):
    """
    Calculate growth rate and frequency from logarithmic amplitude data.
    """
    nt = lndEr.shape[0]
    it0 = int(nt * 5.8 / 20)
    it1 = int(nt * 19.9 / 20)

    if it1 <= it0 or (it0 < 0) or (it1 > nt):
        return np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    length = it1 - it0
    if length <= 1:
        return np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    yy = np.empty(length, dtype=np.float64)
    for i in range(length):
        yy[i] = lndEr[it0 + i]

    extr = np.empty(length, dtype=np.int64)
    cnt = 0
    for i in range(1, length - 1):
        if yy[i] > yy[i - 1] and yy[i] > yy[i + 1]:
            extr[cnt] = i
            cnt += 1

    if cnt >= 1:
        first = extr[0]
        last = extr[cnt - 1]
        t1 = (it0 + first + 1) * dt
        t2 = (it0 + last + 1) * dt
        y1 = yy[first]
        y2 = yy[last]
        nw = cnt - 1

        if nw > 0 and (t2 - t1) != 0.0:
            omega = math.pi / ((t2 - t1) / nw)
        else:
            omega = 0.0
    else:
        t1 = (it0 + 1) * dt
        t2 = (it1 + 1) * dt
        y1 = yy[0]
        y2 = yy[length - 1]
        nw = 0
        omega = 0.0

    gammas_ = (np.real(y2) - np.real(y1)) / (t2 - t1) if (t2 - t1) != 0.0 else np.nan

    return gammas_, omega, t1, t2, y1, y2, nw



if __name__ == "__main__":
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

    model = IVPModel(ky=ky, wr=wr, wi=wi, tau=1.0, epsn=0.2, kz=0.0, kapt=0.5, f_trap=0,
                     f_fast=0.05, tau_f=12.0, eta_f=0.6)
    phit, gi, ge, runtime, gamma, omega_r = model.run(nt=500, plot_results=True, real_time_plot=False)
    print(f"runtime={runtime:.2f}s, gamma={gamma:.3f}, omega_r={omega_r:.3f}")
