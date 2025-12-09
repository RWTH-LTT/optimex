import math

import bw2data as bd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import pyomo.environ as pyo


class PostProcessor:
    """
    A class for post-processing and visualizing results from a solved Pyomo model.

    This class provides plotting utilities with configurable styles for generating
    visualizations such as stacked bar charts, line plots, etc., from model outputs.

    Parameters
    ----------
    solved_model : pyo.ConcreteModel
        A solved Pyomo model instance containing the data to be processed and visualized.

    plot_config : dict, optional
        A dictionary of plot styling options to override default settings. Recognized keys include:
            - "figsize" : tuple of (width, height) in inches
            - "fontsize" : int, font size for labels and titles
            - "grid_alpha" : float, transparency of grid lines
            - "grid_linestyle" : str, line style for grid (e.g., "--", ":", "-.")
            - "rotation" : int, angle of x-axis tick label rotation
            - "bar_width" : float, width of bars in bar charts
            - "colormap" : list of colors used for plotting
            - "line_color" : str, color of lines in line plots
            - "line_marker" : str, marker style for line plots
            - "line_width" : float, width of lines in line plots

        Unrecognized keys are ignored.

    Attributes
    ----------
    m : pyo.ConcreteModel
        The solved Pyomo model.

    _plot_config : dict
        The finalized configuration dictionary used for plotting.
    """

    def __init__(self, solved_model: pyo.ConcreteModel, plot_config: dict = None):
        self.m = solved_model

        # Default plot config
        default_config = {
            "figsize": (10, 6),
            "fontsize": 12,
            "grid_alpha": 0.6,
            "grid_linestyle": "--",
            "rotation": 45,
            "bar_width": 0.8,
            "colormap": plt.colormaps["tab20"].colors,
            "line_color": "black",
            "line_marker": "o",
            "line_width": 2,
        }

        # If user provided config, update defaults with it
        if plot_config:
            default_config.update(
                {k: v for k, v in plot_config.items() if k in default_config}
            )

        self._plot_config = default_config

        # Lazily-populated cache for mapping foreground process codes to names
        self._process_name_map = None
        self._process_color_map = {}

    def _build_process_name_map(self):
        """Create a mapping from process code to human-readable name."""
        mapping = {}

        try:
            databases = bd.databases
        except Exception:
            return mapping

        if "foreground" not in databases:
            return mapping

        for process_code in getattr(self.m, "PROCESS", []):
            name = None
            try:
                node = bd.get_node(database="foreground", code=process_code)
                if hasattr(node, "get"):
                    name = node.get("name")
                if name is None:
                    name = node["name"]
            except Exception:
                name = None

            mapping[process_code] = name or process_code

        return mapping

    def _get_process_name_map(self):
        if self._process_name_map is None:
            self._process_name_map = self._build_process_name_map()
        return self._process_name_map

    def _get_process_color_map(self, resolve_names: bool = True):
        """Assign deterministic colors per process for consistent plotting."""
        cache_key = bool(resolve_names)
        if cache_key in self._process_color_map:
            return self._process_color_map[cache_key]

        name_map = self._get_process_name_map() if resolve_names else {}
        processes = list(getattr(self.m, "PROCESS", []))
        colors = self._plot_config["colormap"]
        color_map = {}
        for idx, process in enumerate(processes):
            display = name_map.get(process, process) if resolve_names else process
            color_map[display] = colors[idx % len(colors)]

        self._process_color_map[cache_key] = color_map
        return color_map

    def _create_clean_axes(self, nrows=1, ncols=1):
        """
        Create a grid of clean axes with consistent formatting.
        Returns fig, flattened list of axes.
        """
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=self._plot_config["figsize"], sharex=True
        )
        axes = axes.flatten() if isinstance(axes, (np.ndarray, list)) else [axes]

        for ax in axes:
            ax.grid(
                axis="y",
                linestyle=self._plot_config["grid_linestyle"],
                alpha=self._plot_config["grid_alpha"],
            )
            ax.tick_params(
                axis="x",
                rotation=self._plot_config["rotation"],
                labelsize=self._plot_config["fontsize"],
            )
            ax.tick_params(axis="y", labelsize=self._plot_config["fontsize"])
            ax.set_xlabel("Time", fontsize=self._plot_config["fontsize"])
            ax.set_ylabel("Value", fontsize=self._plot_config["fontsize"])
        return fig, axes

    def _column_process_label(self, column):
        """Extract process label from single or MultiIndex column."""
        if isinstance(column, tuple):
            return column[0]
        return column

    def _apply_bar_styles(
        self,
        df,
        ax,
        title=None,
        legend_title=None,
        resolve_names: bool = True,
    ):
        """
        Apply standard bar plot styling.
        """
        color_map = self._get_process_color_map(resolve_names=resolve_names)
        colors = [
            color_map.get(self._column_process_label(col), self._plot_config["line_color"])
            for col in df.columns
        ]

        df.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            width=self._plot_config["bar_width"],
            color=colors,
            edgecolor="black",
            legend=False,
        )
        ax.set_title(title or "", fontsize=self._plot_config["fontsize"] + 2)
        ax.set_xticklabels(
            df.index.astype(str),
            rotation=self._plot_config["rotation"],
            ha="right",
        )

        # Build legend without zero-only series
        handles = []
        labels = []
        for col, color in zip(df.columns, colors):
            if df[col].abs().sum() == 0:
                continue
            proc_label = self._column_process_label(col)
            if proc_label in labels:
                continue  # avoid duplicates (e.g., multiple flows for same process)
            labels.append(proc_label)
            handles.append(Patch(facecolor=color, edgecolor="black", label=proc_label))

        if handles:
            ax.legend(
                handles=handles,
                title=legend_title or "",
                fontsize=self._plot_config["fontsize"] - 2,
                loc="upper left",
                frameon=False,
                bbox_to_anchor=(1.02, 1),
                title_fontsize=self._plot_config["fontsize"] - 1,
            )

    def get_impacts(self, resolve_names: bool = True) -> pd.DataFrame:
        """
        Extracts the specific impacts from the model and returns them as a DataFrame.
        The DataFrame will have a MultiIndex with 'Category', 'Process', and 'Time'.
        """
        impacts = {}
        cat_scales = getattr(self.m, "scales", {}).get("characterization", 1.0)
        fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)
        impacts = {
            (c, p, t): pyo.value(self.m.specific_impact[c, p, t])
            * cat_scales[c]
            for c in self.m.CATEGORY
            for p in self.m.PROCESS
            for t in self.m.SYSTEM_TIME
        }
        df = pd.DataFrame.from_dict(impacts, orient="index", columns=["Value"])
        df.index = pd.MultiIndex.from_tuples(
            df.index, names=["Category", "Process", "Time"]
        )
        df = df.reset_index()
        df_pivot = df.pivot(
            index="Time", columns=["Category", "Process"], values="Value"
        )

        if resolve_names:
            mapping = self._get_process_name_map()
            df_pivot.columns = pd.MultiIndex.from_tuples(
                [
                    (category, mapping.get(process, process))
                    for category, process in df_pivot.columns
                ],
                names=df_pivot.columns.names,
            )
        self.df_impacts = df_pivot
        return self.df_impacts

    def get_radiative_forcing(self) -> pd.DataFrame:
        fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)
        inventory = {
            (p, e, t): pyo.value(self.m.scaled_inventory[p, e, t]) * fg_scale
            for p in self.m.PROCESS
            for e in self.m.ELEMENTARY_FLOW
            for t in self.m.SYSTEM_TIME
        }
        # do something with inventory
        return NotImplementedError(
            "Radiative forcing extraction is not implemented yet."
        )

    def get_installation(self, resolve_names: bool = True) -> pd.DataFrame:
        """
        Extracts the installation data from the model and returns it as a DataFrame.
        The DataFrame will have a MultiIndex with 'Time' and 'Process'.
        The values are the installed capacities for each process at each time step.
        """
        fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)
        installation_matrix = {
            (t, p): pyo.value(self.m.var_installation[p, t]) * fg_scale
            for p in self.m.PROCESS
            for t in self.m.SYSTEM_TIME
        }
        df = pd.DataFrame.from_dict(
            installation_matrix, orient="index", columns=["Value"]
        )
        df.index = pd.MultiIndex.from_tuples(df.index, names=["Time", "Process"])
        df = df.reset_index()
        df_pivot = df.pivot(index="Time", columns="Process", values="Value")

        if resolve_names:
            df_pivot = df_pivot.rename(columns=self._get_process_name_map())
        self.df_installation = df_pivot
        return self.df_installation

    def get_operation(self, resolve_names: bool = True) -> pd.DataFrame:
        """
        Extracts the operation data from the model and returns it as a DataFrame.
        The DataFrame will have a MultiIndex with 'Time' and 'Process'.
        The values are the operational levels for each process at each time step.
        """
        fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)
        operation_matrix = {
            (t, p): pyo.value(self.m.var_operation[p, t]) * fg_scale
            for p in self.m.PROCESS
            for t in self.m.SYSTEM_TIME
        }
        df = pd.DataFrame.from_dict(operation_matrix, orient="index", columns=["Value"])
        df.index = pd.MultiIndex.from_tuples(df.index, names=["Time", "Process"])
        df = df.reset_index()
        df_pivot = df.pivot(index="Time", columns="Process", values="Value")

        if resolve_names:
            df_pivot = df_pivot.rename(columns=self._get_process_name_map())
        self.df_operation = df_pivot
        return self.df_operation

    def get_production(self, resolve_names: bool = True) -> pd.DataFrame:
        """
        Extracts the production data from the model and returns it as a DataFrame.
        The DataFrame will have a MultiIndex with 'Process', 'Flow', and
        'Time'. The values are the total production for each process and flow
        at each time step.
        """
        production_tensor = {}
        fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)

        flows_with_production = {k[1] for k in self.m.foreground_production}
        for p in self.m.PROCESS:
            for f in flows_with_production:
                for t in self.m.SYSTEM_TIME:
                    if not self.m.flexible_operation:
                        total_production = sum(
                            self.m.foreground_production[p, f, tau]
                            * pyo.value(self.m.var_installation[p, t - tau])
                            for tau in self.m.PROCESS_TIME
                            if (t - tau in self.m.SYSTEM_TIME)
                        )
                    else:
                        tau0 = self.m.process_operation_start[p]
                        total_production = self.m.foreground_production[
                            p, f, tau0
                        ] * pyo.value(self.m.var_operation[p, t])
                    production_tensor[(p, f, t)] = total_production * fg_scale

        df = pd.DataFrame.from_dict(
            production_tensor, orient="index", columns=["Value"]
        )
        df.index = pd.MultiIndex.from_tuples(
            df.index, names=["Process", "Flow", "Time"]
        )
        df = df.reset_index()
        df_pivot = df.pivot(
            index="Time", columns=["Process", "Flow"], values="Value"
        )

        if resolve_names:
            mapping = self._get_process_name_map()
            df_pivot.columns = pd.MultiIndex.from_tuples(
                [
                    (mapping.get(process, process), flow)
                    for process, flow in df_pivot.columns
                ],
                names=df_pivot.columns.names,
            )
        self.df_production = df_pivot
        return self.df_production

    def get_demand(self) -> pd.DataFrame:
        """
        Extracts the demand data from the model and returns it as a DataFrame.
        The DataFrame will have a MultiIndex with 'Flow' and 'Time'.
        The values are the demand for each flow at each time step.
        """
        fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)
        demand_flows = {k[0] for k in self.m.demand}
        demand_matrix = {
            (f, t): self.m.demand[f, t] * fg_scale
            for f in demand_flows
            for t in self.m.SYSTEM_TIME
        }
        df = pd.DataFrame.from_dict(demand_matrix, orient="index", columns=["Value"])
        df.index = pd.MultiIndex.from_tuples(
            df.index, names=["Flow", "Time"]
        )
        df = df.reset_index()
        df_pivot = df.pivot(index="Time", columns="Flow", values="Value")
        self.df_demand = df_pivot
        return self.df_demand

    def plot_impacts(self, df_impacts=None, resolve_names: bool = True):
        """
        Plot a stacked bar chart for impacts.
        df_impacts: DataFrame with Time as index, Categories and Processes as columns.
        Columns must be a MultiIndex: (Category, Process)
        """
        if df_impacts is None:
            df_impacts = self.get_impacts(resolve_names=resolve_names)

        categories = df_impacts.columns.get_level_values(0).unique()
        ncols = 2
        nrows = math.ceil(len(categories) / ncols)

        fig, axes = self._create_clean_axes(nrows=nrows, ncols=ncols)

        for i, category in enumerate(categories):
            ax = axes[i]
            sub_df = df_impacts[category]
            self._apply_bar_styles(
                sub_df,
                ax,
                title=category,
                legend_title="Process",
                resolve_names=resolve_names,
            )

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        plt.show()
        return fig, axes

    def plot_production_and_demand(
        self, prod_df=None, demand_df=None, resolve_names: bool = True
    ):
        """
        Plot a stacked bar chart for production and line(s) for demand.

        Parameters:
            prod_df: DataFrame with Time as index, Processes as columns.
            demand_df: DataFrame with Time as index, Reference Products as columns.
        """

        if prod_df is None:
            prod_df = self.get_production(resolve_names=resolve_names)
        if demand_df is None:
            demand_df = self.get_demand()

        # Convert indices to strings for consistent tick labeling
        prod_df = prod_df.copy()
        demand_df = demand_df.copy()
        prod_df.index = prod_df.index.astype(str)
        demand_df.index = demand_df.index.astype(str)

        fig, axes = self._create_clean_axes()
        ax = axes[0]

        # Define x positions for line plotting
        x_positions = np.arange(len(prod_df.index))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            prod_df.index,
            rotation=self._plot_config["rotation"],
            ha="right",
            fontsize=self._plot_config["fontsize"],
        )

        # Plot production (stacked bars) with consistent colors per process
        color_map = self._get_process_color_map(resolve_names=resolve_names)
        colors = [
            color_map.get(self._column_process_label(col), self._plot_config["line_color"])
            for col in prod_df.columns
        ]
        prod_df.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            edgecolor="black",
            width=self._plot_config["bar_width"],
            color=colors,
            legend=False,  # We'll handle legend separately
        )

        # Plot demand (lines)
        demand_handles = []
        demand_labels = []
        for col in demand_df.columns:
            ax.plot(
                x_positions,
                demand_df[col].values,
                marker=self._plot_config["line_marker"],
                linewidth=self._plot_config["line_width"],
                label=f"Demand: {col}",
                color=self._plot_config["line_color"],
            )
            demand_labels.append(f"Demand: {col}")
            demand_handles.append(ax.lines[-1])

        # Create combined legend
        # Custom legend without zero-only series and no duplicates by process
        handles = []
        labels = []
        for col, color in zip(prod_df.columns, colors):
            if prod_df[col].abs().sum() == 0:
                continue
            proc_label = self._column_process_label(col)
            if proc_label in labels:
                continue
            labels.append(proc_label)
            handles.append(Patch(facecolor=color, edgecolor="black", label=proc_label))

        handles.extend(demand_handles)
        labels.extend(demand_labels)

        if handles:
            ax.legend(
                handles=handles,
                loc="upper left",
                fontsize=self._plot_config["fontsize"] - 2,
                title="Processes / Demand",
                title_fontsize=self._plot_config["fontsize"],
                frameon=False,
                bbox_to_anchor=(1.02, 1),
            )

        ax.set_title(
            "Production and Demand", fontsize=self._plot_config["fontsize"] + 2
        )
        ax.set_ylabel("Quantity", fontsize=self._plot_config["fontsize"])

        fig.tight_layout()
        plt.show()
        return fig, ax

    def plot_installation(self, df_installation=None, resolve_names: bool = True):
        """
        Plot a stacked bar chart for installation data.
        df_installation: DataFrame with Time as index, Processes as columns.
        """
        if df_installation is None:
            df_installation = self.get_installation(resolve_names=resolve_names)

        fig, axes = self._create_clean_axes()
        ax = axes[0]
        self._apply_bar_styles(
            df_installation,
            ax,
            title="Installed Capacity",
            legend_title="Processes",
            resolve_names=resolve_names,
        )
        ax.set_ylabel("Installation", fontsize=self._plot_config["fontsize"])
        fig.tight_layout()
        plt.show()
        return fig, ax

    def plot_operation(self, df_operation=None, resolve_names: bool = True):
        """
        Plot a stacked bar chart for operation data.
        df_operation: DataFrame with Time as index, Processes as columns.
        """
        if df_operation is None:
            df_operation = self.get_operation(resolve_names=resolve_names)

        fig, axes = self._create_clean_axes()
        ax = axes[0]
        self._apply_bar_styles(
            df_operation,
            ax,
            title="Operational Levels",
            legend_title="Processes",
            resolve_names=resolve_names,
        )
        ax.set_ylabel("Operational Levels", fontsize=self._plot_config["fontsize"])
        fig.tight_layout()
        plt.show()
        return fig, ax
