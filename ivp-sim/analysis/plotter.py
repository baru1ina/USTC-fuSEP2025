import matplotlib.pyplot as plt


class ModelPlotter:
    """
    Универсальный класс для построения графиков.
    Может принимать данные разных форматов (из ModelScanner).
    """
    def plot_dependency(self, data, title_prefix="", filename=None):
        plt.figure(figsize=(8, 6))
        x = data["x_vals"]
        y_lists = data["y_vals"]
        params = data.get("param_vals", [])
        xlabel = data.get("xlabel", "x")
        ylabel = data.get("ylabel", "y")

        for i, y in enumerate(y_lists):
            label = f"{params[i]}" if params and params[i] is not None else ""
            plt.plot(x, y, marker="o", linewidth=2, label=label)

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f"{title_prefix} {ylabel} vs {xlabel}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
        plt.show()
