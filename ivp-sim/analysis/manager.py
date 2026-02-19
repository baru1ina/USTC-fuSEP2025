from .scanner import ModelScanner
from .plotter import ModelPlotter


class AnalysisManager:
    """
    Универсальный интерфейс для анализа и отрисовки.
    Автоматически находит доступные методы анализа вида scan_<analysis_type>.
    """

    def __init__(self, base_model_params: dict):
        self.scanner = ModelScanner(base_model_params)
        self.plotter = ModelPlotter()

        # Автоматически собираем все методы, начинающиеся на "scan_"
        self.available_analyses = {
            method_name.replace("scan_", ""): getattr(self.scanner, method_name)
            for method_name in dir(self.scanner)
            if callable(getattr(self.scanner, method_name)) and method_name.startswith("scan_")
        }

    def list_available(self):
        """Показать список доступных анализов"""
        return list(self.available_analyses.keys())

    def run_analysis(self, analysis_type: str, **kwargs):
        """
        Вызывает нужный метод анализа автоматически.
        """
        if analysis_type not in self.available_analyses:
            raise ValueError(f"Unknown analysis type '{analysis_type}'. "
                             f"Available: {', '.join(self.list_available())}")

        scan_func = self.available_analyses[analysis_type]
        data = scan_func(**kwargs)

        # Если возвращён список (например, gamma_vs_Fp), строим серию графиков
        if isinstance(data, list):
            for dataset in data:
                self.plotter.plot_dependency(dataset, title_prefix=analysis_type)
        else:
            self.plotter.plot_dependency(data, title_prefix=analysis_type)
