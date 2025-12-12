import math

import bw2data as bd
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

        # Create consistent color mapping for all processes and products
        self._color_map = self._create_color_map()

        # Cache for code -> name lookups
        self._name_cache = {}

    def _create_color_map(self):
        """
        Create a consistent color mapping for all processes and products.
        Returns a dict mapping item names to colors.
        """
        # Collect all unique processes and products
        all_items = set()
        all_items.update(self.m.PROCESS)
        all_items.update(self.m.PRODUCT)

        # Sort for consistency
        all_items = sorted(all_items)

        # Map to colors
        colors = self._plot_config["colormap"]
        color_map = {item: colors[i % len(colors)] for i, item in enumerate(all_items)}
        return color_map

    def _get_name(self, code: str) -> str:
        """
        Get the human-readable name for a code with caching.
        Tries foreground database first, then falls back to code if not found.
        """
        if code in self._name_cache:
            return self._name_cache[code]

        try:
            # Try to get from foreground database
            node = bd.get_node(database="foreground", code=code)
            name = node.get("name", code)
        except Exception:
            # If not found or error, use code as name
            name = code

        self._name_cache[code] = name
        return name

    def _annotate_dataframe(self, df, annotated: bool):
        """
        Annotate DataFrame columns with human-readable names if requested.
        Handles both single-level and multi-level column indices.
        """
        if not annotated:
            return df

        if isinstance(df.columns, pd.MultiIndex):
            # Multi-level columns (e.g., (Process, Product))
            new_columns = pd.MultiIndex.from_tuples(
                [tuple(self._get_name(col) for col in cols) for cols in df.columns],
                names=df.columns.names,
            )
            df.columns = new_columns
        else:
            # Single-level columns
            df.columns = [self._get_name(col) for col in df.columns]

        return df

    def _get_colors_for_dataframe(self, df):
        """
        Get consistent colors for DataFrame columns.
        Handles both single-level and multi-level column indices.
        """
        colors = []
        if isinstance(df.columns, pd.MultiIndex):
            # For multi-level, use the first level (Process) for color
            for col in df.columns:
                # Use the first element of the tuple for color lookup
                key = col[0] if isinstance(col, tuple) else col
                colors.append(self._color_map.get(key, self._plot_config["colormap"][0]))
        else:
            # Single-level columns
            for col in df.columns:
                colors.append(self._color_map.get(col, self._plot_config["colormap"][0]))

        return colors

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

    def _apply_bar_styles(self, df, ax, colors, title=None, legend_title=None):
        """
        Apply standard bar plot styling with consistent colors.

        Parameters
        ----------
        df : DataFrame
            Data to plot
        ax : matplotlib axis
            Axis to plot on
        colors : list
            List of colors for each column
        title : str, optional
            Plot title
        legend_title : str, optional
            Legend title
        """
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
        ax.legend(
            title=legend_title or "",
            fontsize=self._plot_config["fontsize"] - 2,
            loc="center left",
            frameon=False,
            bbox_to_anchor=(1.02, 0.5),
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
            * fg_scale  # Unscale foreground impacts
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

        Note: var_operation is not scaled because when both demand and
        foreground_production are scaled by the same factor, the scaling
        cancels out in the constraint: demand = production * operation.
        """
        operation_matrix = {
            (t, p): pyo.value(self.m.var_operation[p, t])
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
        The DataFrame will have a MultiIndex with 'Process', 'Product', and
        'Time'. The values are the total production for each process and product
        at each time step.
        """
        production_tensor = {}
        fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)

        for p in self.m.PROCESS:
            for f in self.m.PRODUCT:
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
            df.index, names=["Process", "Product", "Time"]
        )
        df = df.reset_index()
        df_pivot = df.pivot(
            index="Time", columns=["Process", "Product"], values="Value"
        )
        self.df_production = df_pivot
        return self.df_production

    def get_demand(self) -> pd.DataFrame:
        """
        Extracts the demand data from the model and returns it as a DataFrame.
        The DataFrame will have a MultiIndex with 'Product' and 'Time'.
        The values are the demand for each Product at each time step.
        """
        fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)
        demand_matrix = {
            (f, t): self.m.demand[f, t] * fg_scale
            for f in self.m.PRODUCT
            for t in self.m.SYSTEM_TIME
        }
        df = pd.DataFrame.from_dict(demand_matrix, orient="index", columns=["Value"])
        df.index = pd.MultiIndex.from_tuples(
            df.index, names=["Product", "Time"]
        )
        df = df.reset_index()
        df_pivot = df.pivot(index="Time", columns="Product", values="Value")
        self.df_demand = df_pivot
        return self.df_demand

    def plot_impacts(self, df_impacts=None, annotated=True):
        """
        Plot a stacked bar chart for impacts.

        Parameters
        ----------
        df_impacts : DataFrame, optional
            DataFrame with Time as index, Categories and Processes as columns.
            Columns must be a MultiIndex: (Category, Process)
        annotated : bool, default=True
            If True, show human-readable names instead of codes
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
            # Filter out processes with all zero values in this category
            sub_df = sub_df.loc[:, (sub_df != 0).any(axis=0)]
            # Get colors BEFORE annotation (using codes)
            colors = self._get_colors_for_dataframe(sub_df)
            # Annotate if requested
            sub_df = self._annotate_dataframe(sub_df, annotated)
            self._apply_bar_styles(sub_df, ax, colors, title=category, legend_title="Process")

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        plt.show()
        return fig, axes

    def plot_production_and_demand(self, prod_df=None, demand_df=None, annotated=True):
        """
        Plot a stacked bar chart for production and line(s) for demand.

        Parameters
        ----------
        prod_df : DataFrame, optional
            DataFrame with Time as index, (Process, Product) as columns
        demand_df : DataFrame, optional
            DataFrame with Time as index, Products as columns
        annotated : bool, default=True
            If True, show human-readable names instead of codes
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

        # Filter out columns with all zero values
        prod_df = prod_df.loc[:, (prod_df != 0).any(axis=0)]
        demand_df = demand_df.loc[:, (demand_df != 0).any(axis=0)]

        # Get colors BEFORE annotation (using codes)
        prod_colors = self._get_colors_for_dataframe(prod_df)

        # Annotate if requested
        prod_df = self._annotate_dataframe(prod_df, annotated)
        demand_df = self._annotate_dataframe(demand_df, annotated)

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
        if not prod_df.empty:
            prod_df.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                edgecolor="black",
                width=self._plot_config["bar_width"],
                color=prod_colors,
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
            loc="center left",
            fontsize=self._plot_config["fontsize"] - 2,
            title="Processes / Demand",
            title_fontsize=self._plot_config["fontsize"],
            frameon=False,
            bbox_to_anchor=(1.02, 0.5),
        )

        ax.set_title(
            "Production and Demand", fontsize=self._plot_config["fontsize"] + 2
        )
        ax.set_ylabel("Quantity", fontsize=self._plot_config["fontsize"])

        fig.tight_layout()
        plt.show()
        return fig, ax

    def plot_installation(self, df_installation=None, annotated=True):
        """
        Plot a stacked bar chart for installation data.

        Parameters
        ----------
        df_installation : DataFrame, optional
            DataFrame with Time as index, Processes as columns
        annotated : bool, default=True
            If True, show human-readable names instead of codes
        """
        if df_installation is None:
            df_installation = self.get_installation()

        # Filter out columns with all zero values
        df_installation = df_installation.loc[:, (df_installation != 0).any(axis=0)]

        # Get colors BEFORE annotation (using codes)
        colors = self._get_colors_for_dataframe(df_installation)

        # Annotate if requested
        df_installation = self._annotate_dataframe(df_installation, annotated)

        fig, axes = self._create_clean_axes()
        ax = axes[0]
        self._apply_bar_styles(
            df_installation, ax, colors, title="Installed Capacity", legend_title="Processes"
        )
        ax.set_ylabel("Installation", fontsize=self._plot_config["fontsize"])
        fig.tight_layout()
        plt.show()
        return fig, ax

    def plot_operation(self, df_operation=None, annotated=True):
        """
        Plot a stacked bar chart for operation data.

        Parameters
        ----------
        df_operation : DataFrame, optional
            DataFrame with Time as index, Processes as columns
        annotated : bool, default=True
            If True, show human-readable names instead of codes
        """
        if df_operation is None:
            df_operation = self.get_operation()

        # Filter out columns with all zero values
        df_operation = df_operation.loc[:, (df_operation != 0).any(axis=0)]

        # Get colors BEFORE annotation (using codes)
        colors = self._get_colors_for_dataframe(df_operation)

        # Annotate if requested
        df_operation = self._annotate_dataframe(df_operation, annotated)

        fig, axes = self._create_clean_axes()
        ax = axes[0]
        self._apply_bar_styles(
            df_operation, ax, colors, title="Operational Level", legend_title="Processes"
        )
        ax.set_ylabel("Operation", fontsize=self._plot_config["fontsize"])
        fig.tight_layout()
        plt.show()
        return fig, ax
