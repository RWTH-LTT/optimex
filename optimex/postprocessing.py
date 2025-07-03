import math

import matplotlib.pyplot as plt
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

    def _apply_bar_styles(self, df, ax, title=None, legend_title=None):
        """
        Apply standard bar plot styling.
        """
        df.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            width=self._plot_config["bar_width"],
            color=self._plot_config["colormap"][: len(df.columns)],
            edgecolor="black",
            legend=False,
        )
        ax.set_title(title or "", fontsize=self._plot_config["fontsize"] + 2)
        ax.set_xticklabels(
            df.index.astype(str),
            rotation=self._plot_config["rotation"],
            ha="right",
        )
        ax.legend(
            title=legend_title or "",
            fontsize=self._plot_config["fontsize"] - 2,
            loc="upper right",
            frameon=False,
            bbox_to_anchor=(1.15, 1),
        )

    def get_impacts(self) -> pd.DataFrame:
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
            * fg_scale
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

    def get_installation(self) -> pd.DataFrame:
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
        self.df_installation = df_pivot
        return self.df_installation

    def get_operation(self) -> pd.DataFrame:
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
        self.df_operation = df_pivot
        return self.df_operation

    def get_production(self) -> pd.DataFrame:
        """
        Extracts the production data from the model and returns it as a DataFrame.
        The DataFrame will have a MultiIndex with 'Process', 'Reference Product', and
        'Time'. The values are the total production for each process and reference
        product at each time step.
        """
        production_tensor = {}
        fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)

        for p in self.m.PROCESS:
            for f in self.m.REFERENCE_PRODUCT:
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
            df.index, names=["Process", "Reference Product", "Time"]
        )
        df = df.reset_index()
        df_pivot = df.pivot(
            index="Time", columns=["Process", "Reference Product"], values="Value"
        )
        self.df_production = df_pivot
        return self.df_production

    def get_demand(self) -> pd.DataFrame:
        """
        Extracts the demand data from the model and returns it as a DataFrame.
        The DataFrame will have a MultiIndex with 'Reference Product' and 'Time'.
        The values are the demand for each Reference Product at each time step.
        """
        fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)
        demand_matrix = {
            (f, t): self.m.demand[f, t] * fg_scale
            for f in self.m.REFERENCE_PRODUCT
            for t in self.m.SYSTEM_TIME
        }
        df = pd.DataFrame.from_dict(demand_matrix, orient="index", columns=["Value"])
        df.index = pd.MultiIndex.from_tuples(
            df.index, names=["Reference Product", "Time"]
        )
        df = df.reset_index()
        df_pivot = df.pivot(index="Time", columns="Reference Product", values="Value")
        self.df_demand = df_pivot
        return self.df_demand

    def plot_impacts(self, df_impacts=None):
        """
        Plot a stacked bar chart for impacts.
        df_impacts: DataFrame with Time as index, Categories and Processes as columns.
        Columns must be a MultiIndex: (Category, Process)
        """
        if df_impacts is None:
            df_impacts = self.get_impacts()

        categories = df_impacts.columns.get_level_values(0).unique()
        ncols = 2
        nrows = math.ceil(len(categories) / ncols)

        fig, axes = self._create_clean_axes(nrows=nrows, ncols=ncols)

        for i, category in enumerate(categories):
            ax = axes[i]
            sub_df = df_impacts[category]
            self._apply_bar_styles(sub_df, ax, title=category, legend_title="Process")

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        plt.show()
        return fig, axes

    def plot_production_and_demand(self, prod_df=None, demand_df=None):
        """
        Plot a stacked bar chart for production and line(s) for demand.

        Parameters:
            prod_df: DataFrame with Time as index, Processes as columns.
            demand_df: DataFrame with Time as index, Reference Products as columns.
        """

        if prod_df is None:
            prod_df = self.get_production()
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

        # Plot production (stacked bars)
        prod_df.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            edgecolor="black",
            width=self._plot_config["bar_width"],
            color=self._plot_config["colormap"][: len(prod_df.columns)],
            legend=False,  # We'll handle legend separately
        )

        # Plot demand (lines)
        for col in demand_df.columns:
            ax.plot(
                x_positions,
                demand_df[col].values,
                marker=self._plot_config["line_marker"],
                linewidth=self._plot_config["line_width"],
                label=f"Demand: {col}",
                color=self._plot_config["line_color"],
            )

        # Create combined legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles,
            loc="upper left",
            fontsize=self._plot_config["fontsize"] - 2,
            title="Processes / Demand",
            title_fontsize=self._plot_config["fontsize"],
            frameon=False,
        )

        ax.set_title(
            "Production and Demand", fontsize=self._plot_config["fontsize"] + 2
        )
        ax.set_ylabel("Quantity", fontsize=self._plot_config["fontsize"])

        fig.tight_layout()
        plt.show()
        return fig, ax

    def plot_installation(self, df_installation=None):
        """
        Plot a stacked bar chart for installation data.
        df_installation: DataFrame with Time as index, Processes as columns.
        """
        if df_installation is None:
            df_installation = self.get_installation()

        fig, axes = self._create_clean_axes()
        ax = axes[0]
        self._apply_bar_styles(
            df_installation, ax, title="Installed Capacity", legend_title="Processes"
        )
        ax.set_ylabel("Installation", fontsize=self._plot_config["fontsize"])
        fig.tight_layout()
        plt.show()
        return fig, ax

    def plot_operation(self, df_operation=None):
        """
        Plot a stacked bar chart for operation data.
        df_operation: DataFrame with Time as index, Processes as columns.
        """
        if df_operation is None:
            df_operation = self.get_operation()

        fig, axes = self._create_clean_axes()
        ax = axes[0]
        self._apply_bar_styles(
            df_operation, ax, title="Installed Capacity", legend_title="Processes"
        )
        ax.set_ylabel("Installation", fontsize=self._plot_config["fontsize"])
        fig.tight_layout()
        plt.show()
        return fig, ax
