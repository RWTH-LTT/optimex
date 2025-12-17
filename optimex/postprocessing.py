"""
Post-processing and visualization of optimization results.

This module provides tools to extract, process, and visualize results from solved
optimization models. The PostProcessor class handles denormalization of scaled
results, data extraction into DataFrames, and creation of publication-quality plots
for impacts, installation schedules, production, and operation profiles.

Key classes:
    - PostProcessor: Extract and visualize optimization results
"""
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
            - "max_xticks" : int, maximum number of x-axis ticks to display

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
            "max_xticks": 10,
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

    def _format_label(self, label):
        """Convert column labels (including MultiIndex tuples) to readable strings."""
        if isinstance(label, tuple):
            return " / ".join(str(part) for part in label)
        return str(label)

    def _set_smart_xticks(self, ax, labels):
        """
        Downsample x-axis tick labels to avoid clutter.

        Parameters
        ----------
        ax : matplotlib axis
            Axis on which ticks will be set.
        labels : iterable
            Original labels corresponding to each position along the x-axis.
        """
        labels = [str(lbl) for lbl in labels]
        if not labels:
            return

        max_ticks = self._plot_config.get("max_xticks", 10)
        total = len(labels)
        step = max(1, math.ceil(total / max_ticks))
        tick_positions = list(range(0, total, step))
        if tick_positions[-1] != total - 1:
            tick_positions.append(total - 1)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [labels[i] for i in tick_positions],
            rotation=self._plot_config["rotation"],
            ha="right",
            fontsize=self._plot_config["fontsize"],
        )

    def _create_clean_axes(self, nrows=1, ncols=1, figsize=None):
        """
        Create a grid of clean axes with consistent formatting.
        Returns fig, flattened list of axes.
        """
        fig_size = figsize or self._plot_config["figsize"]
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=fig_size, sharex=True
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
            legend=True,
        )
        ax.set_title(title or "", fontsize=self._plot_config["fontsize"] + 2)
        self._set_smart_xticks(ax, df.index)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            labels = [self._format_label(col) for col in df.columns]
            ax.legend(
                handles,
                labels,
                title=legend_title or "",
                fontsize=self._plot_config["fontsize"] - 2,
                loc="upper center",
                ncol=max(1, min(len(labels), 4)),
                frameon=False,
                bbox_to_anchor=(0.5, -0.25),
                title_fontsize=self._plot_config["fontsize"],
            )

    def get_impacts(self) -> pd.DataFrame:
        """
        Extract environmental impacts by category, process, and time.

        Returns denormalized impact values from the solved optimization model,
        organized as a pivoted DataFrame with time as rows and (category, process)
        as column MultiIndex.

        Returns
        -------
        pd.DataFrame
            Pivoted DataFrame with 'Time' as index and MultiIndex columns for
            (Category, Process) combinations. Values represent environmental
            impacts in the units of the characterization method.
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
        """
        Extract radiative forcing time series from model results.

        This method is currently not implemented. It will extract the scaled inventory
        and compute radiative forcing profiles over time for climate impact assessment.

        Returns
        -------
        pd.DataFrame
            DataFrame with radiative forcing values over time.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
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
        # var_installation is already in real units, no scaling needed
        installation_matrix = {
            (t, p): pyo.value(self.m.var_installation[p, t])
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
                        # Flexible operation: total_production_per_installation × o_t
                        # Sum of production across operation phase × operation level
                        total_production_per_installation = sum(
                            self.m.foreground_production[p, f, tau]
                            for tau in self.m.PROCESS_TIME
                            if self.m.process_operation_start[p] <= tau <= self.m.process_operation_end[p]
                        )

                        # Production = total_production × o_t
                        total_production = (
                            total_production_per_installation
                            * pyo.value(self.m.var_operation[p, t])
                        )
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
        Plot a stacked bar chart for impacts by category and process over time.

        Creates a figure with one subplot per impact category, showing process
        contributions as stacked bars. Automatically denormalizes scaled values
        and optionally displays human-readable process names.

        Parameters
        ----------
        df_impacts : DataFrame, optional
            DataFrame with Time as index, Categories and Processes as columns.
            Columns must be a MultiIndex: (Category, Process). If not provided,
            automatically extracted via get_impacts().
        annotated : bool, default=True
            If True, show human-readable names from Brightway database instead
            of process codes.
        """
        if df_impacts is None:
            df_impacts = self.get_impacts()

        categories = df_impacts.columns.get_level_values(0).unique()
        ncols = 2
        nrows = math.ceil(len(categories) / ncols)

        # Widen figure for multiple columns to avoid cramped subplots
        base_w, base_h = self._plot_config["figsize"]
        fig_w = base_w * ncols
        fig_h = base_h * max(1, nrows * 0.9)

        fig, axes = self._create_clean_axes(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))

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
        fig.subplots_adjust(bottom=0.25)
        plt.show()

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

        # Plot production (stacked bars)
        if not prod_df.empty:
            prod_df.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                edgecolor="black",
                width=self._plot_config["bar_width"],
                color=prod_colors,
                legend=True,  # We'll handle legend separately
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

        self._set_smart_xticks(ax, prod_df.index)

        # Create combined legend
        handles, labels = ax.get_legend_handles_labels()
        bar_count = prod_df.shape[1] if not prod_df.empty else 0
        if bar_count:
            bar_labels = [self._format_label(col) for col in prod_df.columns]
            labels = bar_labels + labels[bar_count:]
        ax.legend(
            handles=handles,
            loc="upper center",
            fontsize=self._plot_config["fontsize"] - 2,
            title="Processes / Demand",
            title_fontsize=self._plot_config["fontsize"],
            frameon=False,
            bbox_to_anchor=(0.5, -0.25),
            ncol=max(1, min(len(labels), 4)) if labels else 1,
        )

        ax.set_title(
            "Production and Demand", fontsize=self._plot_config["fontsize"] + 2
        )
        ax.set_ylabel("Quantity", fontsize=self._plot_config["fontsize"])

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25)
        plt.show()

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
        fig.subplots_adjust(bottom=0.25)
        plt.show()

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
        fig.subplots_adjust(bottom=0.25)
        plt.show()

    def get_production_capacity(self) -> pd.DataFrame:
        """
        Calculate maximum available production capacity for each product at each time step.

        Capacity is determined by counting installations in their operation phase and
        multiplying by their production coefficients.

        Returns
        -------
        pd.DataFrame
            DataFrame with Time as index and Products as columns.
            Values represent maximum production capacity (not actual production).
        """
        capacity_tensor = {}
        fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)

        for f in self.m.PRODUCT:
            for t in self.m.SYSTEM_TIME:
                # Calculate total capacity across all processes
                total_capacity = 0

                for p in self.m.PROCESS:
                    # Count installations in operation phase at time t
                    installations_operating = sum(
                        pyo.value(self.m.var_installation[p, t - tau])
                        for tau in self.m.PROCESS_TIME
                        if (t - tau in self.m.SYSTEM_TIME)
                        and pyo.value(self.m.process_operation_start[p]) <= tau <= pyo.value(self.m.process_operation_end[p])
                    )

                    # Production capacity per installation (sum over operation phase)
                    production_per_installation = sum(
                        self.m.foreground_production[p, f, tau]
                        for tau in self.m.PROCESS_TIME
                        if pyo.value(self.m.process_operation_start[p]) <= tau <= pyo.value(self.m.process_operation_end[p])
                    )

                    # Total capacity for this process
                    total_capacity += installations_operating * production_per_installation

                # Store denormalized capacity
                capacity_tensor[(f, t)] = total_capacity * fg_scale

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(capacity_tensor, orient="index", columns=["Value"])
        df.index = pd.MultiIndex.from_tuples(df.index, names=["Product", "Time"])
        df = df.reset_index()
        df_pivot = df.pivot(index="Time", columns="Product", values="Value")

        return df_pivot

    def plot_production_vs_capacity(self, product=None, prod_df=None, capacity_df=None, demand_df=None, annotated=True, show_capacity_changes=False, show_process_utilization=False):
        """
        Plot actual production vs maximum available capacity for a specific product.

        Shows three lines:
        - Actual production (solid line)
        - Maximum available capacity (dashed line)
        - Demand (dotted line)

        Optionally shows capacity dynamics as bars:
        - Capacity additions (positive bars, green)
        - Capacity removals (negative bars, red)

        Optionally shows per-process utilization breakdown:
        - Stacked bars showing capacity by process
        - Hatched overlay showing actual operation by process
        - Visual distinction between utilized and idle capacity

        Parameters
        ----------
        product : str, optional
            Product to plot. If None, uses the first product with non-zero demand.
        prod_df : pd.DataFrame, optional
            Production DataFrame from get_production()
        capacity_df : pd.DataFrame, optional
            Capacity DataFrame from get_production_capacity()
        demand_df : pd.DataFrame, optional
            Demand DataFrame from get_demand()
        annotated : bool, default=True
            If True, show human-readable names instead of codes
        show_capacity_changes : bool, default=False
            If True, show capacity additions and removals as background bars
        show_process_utilization : bool, default=False
            If True, show per-process capacity and operation breakdown
        """
        # Get data if not provided
        if prod_df is None:
            prod_df = self.get_production()
        if capacity_df is None:
            capacity_df = self.get_production_capacity()
        if demand_df is None:
            demand_df = self.get_demand()

        # Determine which product to plot
        if product is None:
            # Find first product with non-zero demand
            products_with_demand = demand_df.columns[(demand_df != 0).any(axis=0)]
            if len(products_with_demand) == 0:
                raise ValueError("No products with non-zero demand found")
            product = products_with_demand[0]

        # Extract data for the selected product
        # Production: sum across all processes for this product
        if isinstance(prod_df.columns, pd.MultiIndex):
            # Production has MultiIndex (Process, Product)
            production_cols = [col for col in prod_df.columns if col[1] == product]
            actual_production = prod_df[production_cols].sum(axis=1)
        else:
            # Single product case
            actual_production = prod_df[product] if product in prod_df.columns else pd.Series(0, index=prod_df.index)

        # Capacity for this product
        max_capacity = capacity_df[product] if product in capacity_df.columns else pd.Series(0, index=capacity_df.index)

        # Demand for this product
        demand = demand_df[product] if product in demand_df.columns else pd.Series(0, index=demand_df.index)

        # Convert indices to strings for consistent plotting
        actual_production.index = actual_production.index.astype(str)
        max_capacity.index = max_capacity.index.astype(str)
        demand.index = demand.index.astype(str)

        # Calculate capacity changes if requested
        capacity_additions_by_process = None
        capacity_removals_by_process = None
        if show_capacity_changes:
            # Calculate gross capacity additions and removals by process for this product
            fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)

            additions = {p: {} for p in self.m.PROCESS}
            removals = {p: {} for p in self.m.PROCESS}

            for t in self.m.SYSTEM_TIME:
                for p in self.m.PROCESS:
                    # Production capacity per installation for this product
                    prod_per_inst = sum(
                        self.m.foreground_production[p, product, tau]
                        for tau in self.m.PROCESS_TIME
                        if pyo.value(self.m.process_operation_start[p]) <= tau <= pyo.value(self.m.process_operation_end[p])
                    )

                    # Additions: new installations at time t
                    installation = pyo.value(self.m.var_installation[p, t])
                    additions[p][t] = installation * prod_per_inst * fg_scale

                    # Removals: capacity that aged out
                    op_end = pyo.value(self.m.process_operation_end[p])
                    t_aged_out = t - op_end - 1

                    if t_aged_out in self.m.SYSTEM_TIME:
                        installation_aged = pyo.value(self.m.var_installation[p, t_aged_out])
                        removals[p][t] = installation_aged * prod_per_inst * fg_scale
                    else:
                        removals[p][t] = 0

            # Convert to DataFrames with string index
            capacity_additions_by_process = pd.DataFrame(additions)
            capacity_additions_by_process.index = capacity_additions_by_process.index.astype(str)

            capacity_removals_by_process = pd.DataFrame(removals)
            capacity_removals_by_process.index = capacity_removals_by_process.index.astype(str)

            # Filter out processes with no contributions
            capacity_additions_by_process = capacity_additions_by_process.loc[:, (capacity_additions_by_process != 0).any(axis=0)]
            capacity_removals_by_process = capacity_removals_by_process.loc[:, (capacity_removals_by_process != 0).any(axis=0)]

            # Get colors BEFORE annotation (using codes)
            addition_colors = self._get_colors_for_dataframe(capacity_additions_by_process)
            removal_colors = self._get_colors_for_dataframe(capacity_removals_by_process)

            # Annotate if requested
            capacity_additions_by_process = self._annotate_dataframe(capacity_additions_by_process, annotated)
            capacity_removals_by_process = self._annotate_dataframe(capacity_removals_by_process, annotated)

        # Calculate per-process capacity and operation if requested
        process_capacity_df = None
        process_operation_df = None
        process_utilization_colors = None
        if show_process_utilization:
            fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)

            # Calculate capacity and operation for each process
            capacity_by_process = {p: {} for p in self.m.PROCESS}
            operation_by_process = {p: {} for p in self.m.PROCESS}

            for t in self.m.SYSTEM_TIME:
                for p in self.m.PROCESS:
                    # Production capacity per installation for this product
                    prod_per_inst = sum(
                        self.m.foreground_production[p, product, tau]
                        for tau in self.m.PROCESS_TIME
                        if pyo.value(self.m.process_operation_start[p]) <= tau <= pyo.value(self.m.process_operation_end[p])
                    )

                    # Skip processes that don't produce this product
                    if prod_per_inst == 0:
                        capacity_by_process[p][t] = 0
                        operation_by_process[p][t] = 0
                        continue

                    # Count installations in operation phase at time t
                    installations_operating = sum(
                        pyo.value(self.m.var_installation[p, t - tau])
                        for tau in self.m.PROCESS_TIME
                        if (t - tau in self.m.SYSTEM_TIME)
                        and pyo.value(self.m.process_operation_start[p]) <= tau <= pyo.value(self.m.process_operation_end[p])
                    )

                    # Capacity = production per installation × installations
                    capacity_by_process[p][t] = installations_operating * prod_per_inst * fg_scale

                    # Operation = var_operation × production per installation
                    var_op = pyo.value(self.m.var_operation[p, t])
                    operation_by_process[p][t] = var_op * prod_per_inst * fg_scale

            # Convert to DataFrames
            process_capacity_df = pd.DataFrame(capacity_by_process)
            process_capacity_df.index = process_capacity_df.index.astype(str)

            process_operation_df = pd.DataFrame(operation_by_process)
            process_operation_df.index = process_operation_df.index.astype(str)

            # Filter to only processes with non-zero capacity at some point
            has_capacity = (process_capacity_df != 0).any(axis=0)
            process_capacity_df = process_capacity_df.loc[:, has_capacity]
            process_operation_df = process_operation_df.loc[:, has_capacity]

            # Get colors BEFORE annotation
            process_utilization_colors = self._get_colors_for_dataframe(process_capacity_df)

            # Annotate if requested
            process_capacity_df = self._annotate_dataframe(process_capacity_df, annotated)
            process_operation_df = self._annotate_dataframe(process_operation_df, annotated)

        # Get product name for annotation
        product_name = self._get_name(product) if annotated else product

        # Create figure
        fig, axes = self._create_clean_axes()
        ax = axes[0]

        # Define x positions
        x_positions = np.arange(len(actual_production.index))

        # Plot capacity changes as stacked bars if requested
        if show_capacity_changes and capacity_additions_by_process is not None:
            bar_width = 0.6

            # Plot additions as positive stacked bars (by process)
            if not capacity_additions_by_process.empty:
                bottom_additions = np.zeros(len(x_positions))
                for i, col in enumerate(capacity_additions_by_process.columns):
                    values = capacity_additions_by_process[col].values
                    ax.bar(
                        x_positions,
                        values,
                        width=bar_width,
                        bottom=bottom_additions,
                        alpha=0.5,
                        color=addition_colors[i] if i < len(addition_colors) else '#06A77D',
                        label=f'+ {col}',
                        edgecolor='white',
                        linewidth=0.5,
                        zorder=1
                    )
                    bottom_additions += values

            # Plot removals as negative stacked bars (by process)
            if not capacity_removals_by_process.empty:
                bottom_removals = np.zeros(len(x_positions))
                for i, col in enumerate(capacity_removals_by_process.columns):
                    values = capacity_removals_by_process[col].values
                    ax.bar(
                        x_positions,
                        -values,  # Negative for removals
                        width=bar_width,
                        bottom=bottom_removals,
                        alpha=0.5,
                        color=removal_colors[i] if i < len(removal_colors) else '#D62828',
                        label=f'− {col}',
                        edgecolor='white',
                        linewidth=0.5,
                        zorder=1
                    )
                    bottom_removals -= values

        # Plot process utilization breakdown if requested
        if show_process_utilization and process_capacity_df is not None and not process_capacity_df.empty:
            bar_width = 0.7

            # Plot capacity as stacked bars (light colors, shows installed capacity)
            bottom_capacity = np.zeros(len(x_positions))
            for i, col in enumerate(process_capacity_df.columns):
                capacity_values = process_capacity_df[col].values
                color = process_utilization_colors[i] if i < len(process_utilization_colors) else f'C{i}'

                # Capacity bar (light, represents installed capacity)
                ax.bar(
                    x_positions,
                    capacity_values,
                    width=bar_width,
                    bottom=bottom_capacity,
                    alpha=0.3,
                    color=color,
                    edgecolor=color,
                    linewidth=1,
                    label=f'{col} (capacity)',
                    zorder=1
                )
                bottom_capacity += capacity_values

            # Plot operation as overlaid stacked bars (solid colors with hatch, shows actual use)
            bottom_operation = np.zeros(len(x_positions))
            for i, col in enumerate(process_operation_df.columns):
                operation_values = process_operation_df[col].values
                color = process_utilization_colors[i] if i < len(process_utilization_colors) else f'C{i}'

                # Only plot if there's any operation
                if (operation_values > 0.001).any():
                    ax.bar(
                        x_positions,
                        operation_values,
                        width=bar_width,
                        bottom=bottom_operation,
                        alpha=0.8,
                        color=color,
                        edgecolor='white',
                        linewidth=0.5,
                        hatch='///',
                        label=f'{col} (operated)',
                        zorder=2
                    )
                bottom_operation += operation_values

            # Calculate and display utilization summary
            total_capacity = process_capacity_df.sum().sum()
            total_operation = process_operation_df.sum().sum()
            if total_capacity > 0:
                utilization_pct = (total_operation / total_capacity) * 100
                ax.text(
                    0.02, 0.98,
                    f'Overall utilization: {utilization_pct:.1f}%',
                    transform=ax.transAxes,
                    fontsize=self._plot_config["fontsize"] - 2,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )

        # Plot actual production
        ax.plot(
            x_positions,
            actual_production.values,
            marker='o',
            linewidth=self._plot_config["line_width"],
            label='Actual Production',
            color='#2E86AB',
            linestyle='-',
            zorder=3
        )

        # Plot maximum capacity
        ax.plot(
            x_positions,
            max_capacity.values,
            marker='s',
            linewidth=self._plot_config["line_width"],
            label='Max Capacity',
            color='#A23B72',
            linestyle='--',
            zorder=3
        )

        # Plot demand
        ax.plot(
            x_positions,
            demand.values,
            marker='^',
            linewidth=self._plot_config["line_width"],
            label='Demand',
            color='#F18F01',
            linestyle=':',
            zorder=3
        )

        # Fill area between production and capacity (only if not showing bars or utilization)
        if not show_capacity_changes and not show_process_utilization:
            ax.fill_between(
                x_positions,
                actual_production.values,
                max_capacity.values,
                alpha=0.2,
                color='#A23B72',
                label='Unused Capacity',
                zorder=2
            )

        # Set labels and title
        self._set_smart_xticks(ax, actual_production.index)
        ax.set_ylabel(f"Quantity of {product_name}", fontsize=self._plot_config["fontsize"])

        # Update title based on mode
        if show_process_utilization:
            ax.set_title(
                f"Production vs Capacity: {product_name}\n(solid = capacity, hatched = operated)",
                fontsize=self._plot_config["fontsize"] + 2,
                pad=20
            )
        else:
            ax.set_title(
                f"Production vs Capacity: {product_name}",
                fontsize=self._plot_config["fontsize"] + 2,
                pad=20
            )

        # Add grid
        ax.grid(
            True,
            alpha=self._plot_config["grid_alpha"],
            linestyle=self._plot_config["grid_linestyle"]
        )

        # Add legend below the plot
        # Adjust ncol based on number of legend entries
        handles, labels = ax.get_legend_handles_labels()
        ncol = min(4, max(2, len(handles) // 3 + 1)) if show_process_utilization else 3
        ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=ncol,
            fontsize=self._plot_config["fontsize"] - 2,
            frameon=False
        )

        fig.tight_layout()
        # Adjust bottom margin based on legend size
        bottom_margin = 0.30 if show_process_utilization and len(handles) > 6 else 0.25
        fig.subplots_adjust(bottom=bottom_margin)
        plt.show()

    def plot_utilization_heatmap(self, product=None, annotated=True, show_values=True):
        """
        Plot a heatmap showing capacity utilization by process over time.

        This provides a clean, dedicated view of which processes are being
        operated vs sitting idle at each time step.

        Parameters
        ----------
        product : str, optional
            Product to analyze. If None, uses the first product with non-zero demand.
        annotated : bool, default=True
            If True, show human-readable process names instead of codes.
        show_values : bool, default=True
            If True, show utilization percentages in cells.
        """
        # Get demand to determine product
        demand_df = self.get_demand()

        if product is None:
            products_with_demand = demand_df.columns[(demand_df != 0).any(axis=0)]
            if len(products_with_demand) == 0:
                raise ValueError("No products with non-zero demand found")
            product = products_with_demand[0]

        fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)

        # Calculate utilization for each process at each time
        utilization_data = {}
        capacity_data = {}
        operation_data = {}

        for p in self.m.PROCESS:
            # Check if process produces this product
            prod_per_inst = sum(
                self.m.foreground_production[p, product, tau]
                for tau in self.m.PROCESS_TIME
                if pyo.value(self.m.process_operation_start[p]) <= tau <= pyo.value(self.m.process_operation_end[p])
            )

            if prod_per_inst == 0:
                continue  # Skip processes that don't produce this product

            utilization_data[p] = {}
            capacity_data[p] = {}
            operation_data[p] = {}

            for t in self.m.SYSTEM_TIME:
                # Calculate capacity
                installations_operating = sum(
                    pyo.value(self.m.var_installation[p, t - tau])
                    for tau in self.m.PROCESS_TIME
                    if (t - tau in self.m.SYSTEM_TIME)
                    and pyo.value(self.m.process_operation_start[p]) <= tau <= pyo.value(self.m.process_operation_end[p])
                )
                capacity = installations_operating * prod_per_inst * fg_scale

                # Calculate operation
                var_op = pyo.value(self.m.var_operation[p, t])
                operation = var_op * prod_per_inst * fg_scale

                capacity_data[p][t] = capacity
                operation_data[p][t] = operation

                # Calculate utilization %
                if capacity > 0.001:
                    utilization_data[p][t] = (operation / capacity) * 100
                else:
                    utilization_data[p][t] = np.nan  # No capacity = no utilization possible

        if not utilization_data:
            raise ValueError(f"No processes produce {product}")

        # Convert to DataFrame
        util_df = pd.DataFrame(utilization_data).T
        cap_df = pd.DataFrame(capacity_data).T
        op_df = pd.DataFrame(operation_data).T

        # Filter to only times with some capacity
        has_capacity = (cap_df.sum(axis=0) > 0.001)
        util_df = util_df.loc[:, has_capacity]
        cap_df = cap_df.loc[:, has_capacity]
        op_df = op_df.loc[:, has_capacity]

        if util_df.empty:
            raise ValueError(f"No capacity found for {product}")

        # Annotate process names
        if annotated:
            util_df.index = [self._get_name(p) for p in util_df.index]
            cap_df.index = [self._get_name(p) for p in cap_df.index]
            op_df.index = [self._get_name(p) for p in op_df.index]

        product_name = self._get_name(product) if annotated else product

        # Create figure
        fig, ax = plt.subplots(figsize=(max(10, len(util_df.columns) * 0.4), max(4, len(util_df) * 0.8)))

        # Create heatmap
        # Use a diverging colormap: red (0%) -> yellow (50%) -> green (100%)
        cmap = plt.cm.RdYlGn
        im = ax.imshow(util_df.values, aspect='auto', cmap=cmap, vmin=0, vmax=100)

        # Set ticks
        ax.set_xticks(np.arange(len(util_df.columns)))
        ax.set_yticks(np.arange(len(util_df.index)))
        ax.set_xticklabels(util_df.columns, fontsize=self._plot_config["fontsize"] - 2)
        ax.set_yticklabels(util_df.index, fontsize=self._plot_config["fontsize"] - 1)

        # Rotate x labels if many years
        if len(util_df.columns) > 15:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        # Add value annotations
        if show_values:
            for i in range(len(util_df.index)):
                for j in range(len(util_df.columns)):
                    val = util_df.iloc[i, j]
                    if not np.isnan(val):
                        # Choose text color based on background
                        text_color = 'white' if val < 30 or val > 70 else 'black'
                        ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                               fontsize=self._plot_config["fontsize"] - 3, color=text_color)
                    else:
                        ax.text(j, i, '-', ha='center', va='center',
                               fontsize=self._plot_config["fontsize"] - 3, color='gray')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Utilization %', fontsize=self._plot_config["fontsize"])

        # Labels and title
        ax.set_xlabel('Year', fontsize=self._plot_config["fontsize"])
        ax.set_ylabel('Process', fontsize=self._plot_config["fontsize"])
        ax.set_title(f'Capacity Utilization: {product_name}', fontsize=self._plot_config["fontsize"] + 2)

        # Add summary statistics
        mean_util = np.nanmean(util_df.values)
        total_cap = cap_df.sum().sum()
        total_op = op_df.sum().sum()
        overall_util = (total_op / total_cap * 100) if total_cap > 0 else 0

        ax.text(1.02, 0.98, f'Overall: {overall_util:.0f}%\nMean: {mean_util:.0f}%',
               transform=ax.transAxes, fontsize=self._plot_config["fontsize"] - 2,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        fig.tight_layout()
        plt.show()
