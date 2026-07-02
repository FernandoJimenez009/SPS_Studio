"""Menu and analysis-module controller for SPS Studio."""

from __future__ import annotations

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMessageBox

from modules.anova_analysis import AnovaAnalysisBuilder
from modules.capability_study import CapabilityStudyBuilder
from modules.control_charts import ControlChartsBuilder
from modules.factorial_doe import FactorialDOEBuilder
from modules.gage_study import GageStudyBuilder
from modules.graph_builder import GraphBuilder
from modules.hypothesis_testing import HypothesisTestBuilder
from modules.multiple_regression import MultipleRegressionBuilder
from modules.normality_test import NormalityTestWidget
from modules.simple_regression import SimpleRegressionBuilder


class MenuController:
    """Build menus and open analysis builders.

    Parameters
    ----------
    app_window : SPSStudioMainWindow
        Main application window used as parent and compatibility facade.
    """

    def __init__(self, app_window):
        """Store the main window reference."""
        self.app = app_window

    def build_menu(self) -> None:
        """Build the full application menu bar."""
        menu = self.app.menuBar()

        file_menu = menu.addMenu("File")
        file_menu.addAction(self._action("New Project", self.app.new_project))
        file_menu.addAction(self._action("Open Project...", self.app.open_project))
        file_menu.addSeparator()
        file_menu.addAction(self._action("Save Project", self.app.save_project))
        file_menu.addAction(self._action("Save Project As...", self.app.save_project_as))
        file_menu.addSeparator()
        file_menu.addAction(self._action("Exit", self.app.close))

        worksheet_menu = menu.addMenu("Worksheet")
        worksheet_menu.addAction(self._action("Add Worksheet", self.app.add_worksheet_dialog))
        worksheet_menu.addAction(self._action("Rename Worksheet", self.app.rename_current_worksheet))
        worksheet_menu.addAction(self._action("Delete Worksheet", self.app.delete_current_worksheet))

        graph_menu = menu.addMenu("Graph")
        self._build_graph_menu(graph_menu)

        stat_menu = menu.addMenu("Stat")
        basic_menu = stat_menu.addMenu("Basic Statistics")
        basic_menu.addAction(self._action("Normality Test...", self.open_normality_test))

        hypothesis_menu = basic_menu.addMenu("Hypothesis Tests")
        hypothesis_menu.addAction(
            self._action("1-Sample t / 1 Variance...", lambda: self.open_hypothesis_test("one_sample"))
        )
        hypothesis_menu.addAction(
            self._action("2-Sample t / 2 Variances...", lambda: self.open_hypothesis_test("two_sample"))
        )
        hypothesis_menu.addAction(
            self._action("Paired t / 2 Variances...", lambda: self.open_hypothesis_test("paired"))
        )
        hypothesis_menu.addAction(
            self._action("Proportions...", lambda: self.open_hypothesis_test("proportions"))
        )

        regression_menu = stat_menu.addMenu("Regression")
        regression_menu.addAction(self._action("Simple Regression...", self.open_simple_regression))
        regression_menu.addAction(self._action("Multiple Regression...", self.open_multiple_regression))

        anova_menu = stat_menu.addMenu("ANOVA")
        anova_menu.addAction(self._action("One-Way ANOVA...", self.open_anova_analysis))

        doe_menu = stat_menu.addMenu("DOE")
        doe_menu.addAction(self._action("Factorial Design...", self.open_factorial_doe))

        control_menu = stat_menu.addMenu("Control Charts")
        control_menu.addAction(self._action("Individuals Chart...", self.open_control_charts))

        quality_menu = stat_menu.addMenu("Quality Tools")
        quality_menu.addAction(self._action("Capability Analysis...", self.open_capability_study))

        gage_menu = quality_menu.addMenu("Gage Study")
        gage_menu.addAction(self._action("Gage R&R Study (Crossed)...", self.open_gage_rr_crossed))

        tables_menu = stat_menu.addMenu("Tables")
        tables_menu.addAction(
            self._action("Cross Tabulation...", lambda: self.app.planned("Cross Tabulation"))
        )

        fta_menu = menu.addMenu("FTA")
        fta_menu.addAction(
            self._action("Problem Definition Tree...", lambda: self.app.planned("Problem Definition Tree"))
        )
        fta_menu.addAction(
            self._action("Fault Tree Analysis...", lambda: self.app.planned("Fault Tree Analysis"))
        )
        fta_menu.addAction(
            self._action("Red X Logic Tree...", lambda: self.app.planned("Red X Logic Tree"))
        )

        help_menu = menu.addMenu("Help")
        help_menu.addAction(self._action("About SPS Studio", self.app.about))

    def _action(self, text: str, slot) -> QAction:
        """Create a menu action connected to a slot.

        Parameters
        ----------
        text : str
            Menu label.
        slot : callable
            Slot executed when the action is triggered.

        Returns
        -------
        PySide6.QtGui.QAction
            Configured action object.
        """
        action = QAction(text, self.app)
        action.triggered.connect(slot)
        return action

    def _build_graph_menu(self, graph_menu) -> None:
        """Populate the Graph menu.

        Parameters
        ----------
        graph_menu : PySide6.QtWidgets.QMenu
            Target graph menu.
        """
        families = {
            "Distribution / Exploratory": ["Histogram", "Dotplot", "Empirical CDF", "Symmetry Plot"],
            "Relationship": ["Scatterplot", "Matrix Plot", "Bubble Plot"],
            "Comparison": [
                "Boxplot",
                "Interval Plot",
                "Individual Value Plot",
                "Line Plot",
                "Multi-Vari Chart",
                "Variability Chart",
            ],
            "Categorical / Others": ["Bar Chart", "Heatmap", "Pie Chart"],
            "Time Based": ["Time Series Plot", "Area Graph"],
            "Advanced": ["Contour Plot", "3D Scatterplot", "3D Surface Plot"],
        }

        for family, graphs in families.items():
            sub_menu = graph_menu.addMenu(family)
            for graph in graphs:
                sub_menu.addAction(
                    self._action(
                        f"{graph}...",
                        lambda checked=False, graph_type=graph: self.open_graph_builder(graph_type),
                    )
                )

    def _open_builder(self, builder, tab_name: str, status_message: str) -> None:
        """Open a builder widget in the central tab area.

        Parameters
        ----------
        builder : PySide6.QtWidgets.QWidget
            Builder widget instance.
        tab_name : str
            Central tab name.
        status_message : str
            Message displayed in the status bar.
        """
        self.app.clear_central()
        self.app.central_tabs.addTab(builder, tab_name)
        self.app.central_tabs.setCurrentWidget(builder)
        self.app.statusBar().showMessage(status_message)

    def open_graph_builder(self, graph_type: str = "Histogram") -> None:
        """Open Graph Builder with an initial graph type."""
        builder = GraphBuilder(self.app, initial_graph=graph_type)
        self._open_builder(builder, "Graph Builder", f"Graph Builder opened: {graph_type}")

    def open_gage_rr_crossed(self) -> None:
        """Open the crossed Gage R&R study builder."""
        builder = GageStudyBuilder(self.app, study_type="Crossed")
        self._open_builder(builder, "Gage R&R Study", "Gage R&R Study opened.")

    def open_capability_study(self) -> None:
        """Open Capability Study Builder."""
        builder = CapabilityStudyBuilder(app_window=self.app)
        self.app.capability_study_widget = builder
        self._open_builder(builder, "Capability Study", "Capability Study Builder opened.")

    def open_normality_test(self) -> None:
        """Open Normality Test."""
        builder = NormalityTestWidget(app_window=self.app)
        self._open_builder(builder, "Normality Test", "Normality Test opened.")

    def open_hypothesis_test(self, study_type: str = "two_sample") -> None:
        """Open Hypothesis Test Builder.

        Parameters
        ----------
        study_type : str, default="two_sample"
            Hypothesis test workflow to initialize.
        """
        builder = HypothesisTestBuilder(app_window=self.app, study_type=study_type)
        self._open_builder(builder, "Hypothesis Test", "Hypothesis Test Builder opened.")

    def open_simple_regression(self) -> None:
        """Open Simple Regression Builder."""
        builder = SimpleRegressionBuilder(app_window=self.app)
        self._open_builder(builder, "Simple Regression", "Simple Regression Builder opened.")

    def open_multiple_regression(self) -> None:
        """Open Multiple Regression Builder."""
        builder = MultipleRegressionBuilder(app_window=self.app)
        self._open_builder(builder, "Multiple Regression", "Multiple Regression Builder opened.")

    def open_anova_analysis(self) -> None:
        """Open ANOVA Analysis Builder."""
        builder = AnovaAnalysisBuilder(app_window=self.app)
        self._open_builder(builder, "ANOVA Analysis", "ANOVA Analysis Builder opened.")

    def open_factorial_doe(self) -> None:
        """Open Factorial DOE Builder."""
        builder = FactorialDOEBuilder(app_window=self.app)
        self._open_builder(builder, "Factorial DOE", "Factorial DOE Builder opened.")

    def open_control_charts(self) -> None:
        """Open Control Charts Builder."""
        builder = ControlChartsBuilder(app_window=self.app)
        self._open_builder(builder, "Control Charts", "Control Charts Builder opened.")

    def _entry_or_warn(self, entry_id, expected_type: str):
        """Return a journal entry after validating its type.

        Parameters
        ----------
        entry_id : str
            Journal entry identifier.
        expected_type : str
            Required entry type.

        Returns
        -------
        dict or None
            Matching journal entry or None when validation fails.
        """
        entry = self.app.find_journal_entry(entry_id)
        if not entry:
            QMessageBox.warning(
                self.app,
                "Journal entry not found",
                "The selected journal entry could not be found.",
            )
            return None

        if entry.get("type") != expected_type:
            QMessageBox.information(
                self.app,
                f"Not a {expected_type}",
                f"Only {expected_type} journal entries can be edited here.",
            )
            return None

        return entry

    def open_graph_builder_from_journal(self, entry_id) -> None:
        """Open a graph journal entry for editing."""
        entry = self._entry_or_warn(entry_id, "Graph")
        if not entry:
            return
        graph_type = entry.get("graph") or entry.get("payload", {}).get("graph") or "Histogram"
        builder = GraphBuilder(self.app, initial_graph=graph_type)
        builder.load_from_journal_entry(entry)
        self._open_builder(builder, "Graph Builder", f"Editing journal graph: {entry.get('name', 'Analysis')}")

    def open_gage_study_from_journal(self, entry_id) -> None:
        """Open a Gage Study journal entry for editing."""
        entry = self._entry_or_warn(entry_id, "Gage Study")
        if not entry:
            return
        builder = GageStudyBuilder(self.app, study_type="Crossed")
        builder.load_from_journal_entry(entry)
        self._open_builder(builder, "Gage R&R Study", f"Editing journal Gage Study: {entry.get('name', 'Analysis')}")

    def open_capability_study_from_journal(self, entry_id) -> None:
        """Open a Capability Study journal entry for editing."""
        entry = self._entry_or_warn(entry_id, "Capability Study")
        if not entry:
            return
        builder = CapabilityStudyBuilder(app_window=self.app)
        builder.load_from_journal_entry(entry)
        self._open_builder(builder, "Capability Study", f"Editing journal Capability Study: {entry.get('name', 'Analysis')}")

    def open_normality_test_from_journal(self, entry_id) -> None:
        """Open a Normality Test journal entry for editing."""
        entry = self._entry_or_warn(entry_id, "Normality Test")
        if not entry:
            return
        builder = NormalityTestWidget(app_window=self.app)
        builder.load_from_journal_entry(entry)
        self._open_builder(builder, "Normality Test", "Editing journal Normality Test.")

    def open_hypothesis_test_from_journal(self, entry_id) -> None:
        """Open a Hypothesis Test journal entry for editing."""
        entry = self._entry_or_warn(entry_id, "Hypothesis Test")
        if not entry:
            return
        study_type = entry.get("payload", {}).get("study_type", "two_sample")
        builder = HypothesisTestBuilder(app_window=self.app, study_type=study_type)
        builder.load_from_journal_entry(entry)
        self._open_builder(builder, "Hypothesis Test", f"Editing journal Hypothesis Test: {entry.get('name', 'Analysis')}")

    def open_simple_regression_from_journal(self, entry_id) -> None:
        """Open a Simple Regression journal entry for editing."""
        entry = self._entry_or_warn(entry_id, "Simple Regression")
        if not entry:
            return
        builder = SimpleRegressionBuilder(app_window=self.app)
        builder.load_from_journal_entry(entry)
        self._open_builder(builder, "Simple Regression", f"Editing journal Simple Regression: {entry.get('name', 'Analysis')}")

    def open_multiple_regression_from_journal(self, entry_id) -> None:
        """Open a Multiple Regression journal entry for editing."""
        entry = self._entry_or_warn(entry_id, "Multiple Regression")
        if not entry:
            return
        builder = MultipleRegressionBuilder(app_window=self.app)
        builder.load_from_journal_entry(entry)
        self._open_builder(builder, "Multiple Regression", f"Editing journal Multiple Regression: {entry.get('name', 'Analysis')}")

    def open_anova_analysis_from_journal(self, entry_id) -> None:
        """Open an ANOVA Analysis journal entry for editing."""
        entry = self._entry_or_warn(entry_id, "ANOVA Analysis")
        if not entry:
            return
        builder = AnovaAnalysisBuilder(app_window=self.app)
        builder.load_from_journal_entry(entry)
        self._open_builder(builder, "ANOVA Analysis", "Editing journal ANOVA Analysis.")

    def open_factorial_doe_from_journal(self, entry_id) -> None:
        """Open a Factorial DOE journal entry for editing."""
        entry = self._entry_or_warn(entry_id, "Factorial DOE")
        if not entry:
            return
        builder = FactorialDOEBuilder(app_window=self.app)
        builder.load_from_journal_entry(entry)
        self._open_builder(builder, "Factorial DOE", f"Editing journal Factorial DOE: {entry.get('name', 'Analysis')}")

    def open_control_charts_from_journal(self, entry_id) -> None:
        """Open a Control Chart journal entry for editing."""
        entry = self._entry_or_warn(entry_id, "Control Chart")
        if not entry:
            return
        builder = ControlChartsBuilder(app_window=self.app)
        builder.load_from_journal_entry(entry)
        self._open_builder(builder, "Control Charts", f"Editing journal Control Chart: {entry.get('name', 'Analysis')}")
