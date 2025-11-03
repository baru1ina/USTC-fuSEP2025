import matplotlib.pyplot as plt

def plot_gamma_vs_kz(kz_vals, gamma_vals, ftrap_vals, epsn=0.2, filename="rezults"):
    plt.figure(figsize=(8, 6))
    for i, gamma in enumerate(gamma_vals):
        plt.plot(kz_vals, gamma, linewidth=2, marker='.', label=f'f_trap={ftrap_vals[i]}')
    plt.xlabel(r'$k_\parallel$', fontsize=12)
    plt.ylabel(r'$\gamma$', fontsize=12)
    title_ = f"gamma-k_par-epsn={epsn}"
    plt.title(fr'$\gamma$-$k_\parallel$ epsn={epsn}')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(filename + title_ + ".png")
    plt.show()


def plot_gamma_vs_ftrap(ftrap_vals, gamma_vals, epsn=0.2, kz=0.0, filename="rezults"):
    plt.figure(figsize=(8, 6))
    for i, gamma in enumerate(gamma_vals):
        plt.plot(ftrap_vals, gamma_vals, '.', linewidth=2)
    plt.xlabel(r'$f_{\text{trap}}$', fontsize=12)
    plt.ylabel(r'$\gamma$', fontsize=12)
    title_ = f"gamma-f_trap-epsn={epsn}-k_par={kz}"
    plt.title(fr'$\gamma$-$f_{{trap}}$ epsn={epsn}, $k_\parallel$={kz}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename + title_ + ".png")
    plt.show()


def plot_results(Fp_values, gamma_values_list, param_list, epsn, param_name="f_trap", filename="rezults"):
    plt.figure(figsize=(10, 6))

    for gamma_values, param in zip(gamma_values_list, param_list):
        plt.plot(Fp_values, gamma_values, marker='o', label=f'{param_name} = {param:.4f}')

    plt.xlabel('F_p', fontsize=12)
    plt.ylabel('Growth rate γ', fontsize=12)
    plt.title(f'Зависимость γ от F_p (epsn = {epsn:.1f})', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    title_ = f"gamma-Fp-epsn={epsn}-{param_name}"
    plt.savefig(f"{filename}{title_}.png")
    plt.show()