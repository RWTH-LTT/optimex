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

        # Pre-populate cache for code -> name lookups (batch load for performance)
        self._name_cache = self._build_name_cache()

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

    def _build_name_cache(self) -> dict:
        """
        Build a cache of code -> name mappings by batch loading from the database.
        This is much faster than querying one node at a time.
        """
        cache = {}
        try:
            # Batch load all activities from the foreground database
            foreground_db = bd.Database("foreground")
            for activity in foreground_db:
                code = activity.get("code", "")
                name = activity.get("name", code)
                if code:
                    cache[code] = name
        except Exception:
            # If database access fails, start with empty cache
            pass
        return cache

    def _get_name(self, code: str) -> str:
        """
        Get the human-readable name for a code.
        Uses pre-populated cache, falls back to code if not found.
        """
        return self._name_cache.get(code, code)

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
        if hasattr(self, "df_impacts"):
            return self.df_impacts
        
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

    def get_dynamic_inventory(self, biosphere_database: str = "ecoinvent-3.12-biosphere") -> pd.DataFrame:
        """
        Extract the dynamic inventory from the solved model.

        Returns a DataFrame with elementary flows over time, formatted for use
        with dynamic_characterization.

        Parameters
        ----------
        biosphere_database : str, default="ecoinvent-3.12-biosphere"
            Name of the biosphere database to look up flow IDs.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: activity, flow, date, amount.
            - activity: process code (str)
            - flow: biosphere flow ID (int)
            - date: datetime of emission
            - amount: flow amount (float)
        """
        fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)
        inventory = {
            (p, e, t): pyo.value(self.m.scaled_inventory[p, e, t]) * fg_scale
            for p in self.m.PROCESS
            for e in self.m.ELEMENTARY_FLOW
            for t in self.m.SYSTEM_TIME
        }

        df = pd.DataFrame.from_records(
            [(p, e, t, v) for (p, e, t), v in inventory.items()],
            columns=["activity", "flow", "date", "amount"]
        ).astype({
            "activity": "str",
            "flow": "str",
            "amount": "float64"
        })

        # Convert year integers to datetime
        df["date"] = pd.to_datetime(df["date"].astype(int), format="%Y")

        # Convert flow codes to database IDs
        biosphere_db = bd.Database(biosphere_database)
        df["flow"] = df["flow"].apply(
            lambda x: biosphere_db.get(code=x).id
        )

        self.df_dynamic_inventory = df
        return self.df_dynamic_inventory

    def get_characterized_dynamic_inventory(
        self,
        base_lcia_method: tuple,
        metric: str = "radiative_forcing",
        time_horizon: int = 100,
        fixed_time_horizon: bool = True,
        biosphere_database: str = "ecoinvent-3.12-biosphere",
        df_inventory: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Characterize the dynamic inventory using dynamic_characterization.

        Parameters
        ----------
        base_lcia_method : tuple
            The LCIA method tuple for characterization (e.g., ('IPCC', 'GWP100')).
        metric : str, default="radiative_forcing"
            Characterization metric. Options: "radiative_forcing", "GWP".
        time_horizon : int, default=100
            Time horizon for characterization in years.
        fixed_time_horizon : bool, default=True
            If True, use fixed time horizon; if False, use dynamic time horizon.
        biosphere_database : str, default="ecoinvent-3.12-biosphere"
            Name of the biosphere database (used if df_inventory not provided).
        df_inventory : pd.DataFrame, optional
            Pre-computed inventory DataFrame. If not provided, calls get_dynamic_inventory().

        Returns
        -------
        pd.DataFrame
            Characterized inventory DataFrame with columns: date, amount.
        """
        from dynamic_characterization import characterize

        if df_inventory is None:
            df_inventory = self.get_dynamic_inventory(biosphere_database=biosphere_database)

        df_characterized = characterize(
            df_inventory,
            metric=metric,
            base_lcia_method=base_lcia_method,
            time_horizon=time_horizon,
            fixed_time_horizon=fixed_time_horizon,
        )

        self.df_characterized_inventory = df_characterized
        return self.df_characterized_inventory

    def plot_characterized_dynamic_inventory(
        self,
        base_lcia_method: tuple = None,
        metric: str = "radiative_forcing",
        time_horizon: int = 100,
        fixed_time_horizon: bool = True,
        biosphere_database: str = "ecoinvent-3.12-biosphere",
        df_characterized: pd.DataFrame = None,
    ):
        """
        Plot the characterized dynamic inventory aggregated by year.

        Parameters
        ----------
        base_lcia_method : tuple, optional
            The LCIA method tuple for characterization. Required if df_characterized
            is not provided.
        metric : str, default="radiative_forcing"
            Characterization metric (used if df_characterized not provided).
        time_horizon : int, default=100
            Time horizon for characterization (used if df_characterized not provided).
        fixed_time_horizon : bool, default=True
            If True, use fixed time horizon (used if df_characterized not provided).
        biosphere_database : str, default="ecoinvent-3.12-biosphere"
            Name of the biosphere database (used if df_characterized not provided).
        df_characterized : pd.DataFrame, optional
            Pre-computed characterized inventory. If not provided, calls
            get_characterized_dynamic_inventory().
        """
        if df_characterized is None:
            if base_lcia_method is None:
                raise ValueError("base_lcia_method is required when df_characterized is not provided")
            df_characterized = self.get_characterized_dynamic_inventory(
                base_lcia_method=base_lcia_method,
                metric=metric,
                time_horizon=time_horizon,
                fixed_time_horizon=fixed_time_horizon,
                biosphere_database=biosphere_database,
            )

        # Ensure date column is datetime
        df_plot = df_characterized.copy()
        df_plot["date"] = pd.to_datetime(df_plot["date"])

        # Round to nearest year and aggregate
        df_grouped = (
            df_plot
            .assign(date_rounded=(df_plot["date"] + pd.offsets.MonthBegin(6)).dt.to_period("Y").dt.to_timestamp())
            .groupby("date_rounded")["amount"]
            .sum()
            .reset_index()
        )

        # Create plot
        fig, axes = self._create_clean_axes()
        ax = axes[0]

        ax.plot(
            df_grouped["date_rounded"],
            df_grouped["amount"],
            marker=self._plot_config["line_marker"],
            linewidth=self._plot_config["line_width"],
            color=self._plot_config["line_color"],
        )

        ax.set_xlabel("Year", fontsize=self._plot_config["fontsize"])
        ax.set_ylabel(f"{metric.replace('_', ' ').title()}", fontsize=self._plot_config["fontsize"])
        ax.set_title(f"Dynamic {metric.replace('_', ' ').title()}", fontsize=self._plot_config["fontsize"] + 2)

        fig.tight_layout()
        plt.show()

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

    def get_operation(self, aggregate_vintages: bool = True) -> pd.DataFrame:
        """
        Extracts the operation data from the model and returns it as a DataFrame.

        With 3D var_operation[p, v, t], this method can either aggregate across
        vintages (backward compatible) or return per-vintage data.

        Parameters
        ----------
        aggregate_vintages : bool, default=True
            If True (default), sum operation across vintages for each (process, time)
            to provide backward-compatible 2D output.
            If False, return full 3D data with (Process, Vintage) as MultiIndex columns.

        Returns
        -------
        pd.DataFrame
            If aggregate_vintages=True: DataFrame with Time as index, Process as columns.
            If aggregate_vintages=False: DataFrame with Time as index,
                (Process, Vintage) MultiIndex columns.

        Note: var_operation is not scaled because when both demand and
        foreground_production are scaled by the same factor, the scaling
        cancels out in the constraint: demand = production * operation.
        """
        if aggregate_vintages:
            # Aggregate across vintages (backward compatible)
            operation_matrix = {}
            for p in self.m.PROCESS:
                for t in self.m.SYSTEM_TIME:
                    total_op = sum(
                        pyo.value(self.m.var_operation[proc, v, time])
                        for (proc, v, time) in self.m.ACTIVE_VINTAGE_TIME
                        if proc == p and time == t
                    )
                    operation_matrix[(t, p)] = total_op
            df = pd.DataFrame.from_dict(operation_matrix, orient="index", columns=["Value"])
            df.index = pd.MultiIndex.from_tuples(df.index, names=["Time", "Process"])
            df = df.reset_index()
            df_pivot = df.pivot(index="Time", columns="Process", values="Value")
            self.df_operation = df_pivot
            return self.df_operation
        else:
            # Return full 3D data with (Process, Vintage) columns
            operation_matrix = {
                (t, p, v): pyo.value(self.m.var_operation[p, v, t])
                for (p, v, t) in self.m.ACTIVE_VINTAGE_TIME
            }
            df = pd.DataFrame.from_dict(operation_matrix, orient="index", columns=["Value"])
            df.index = pd.MultiIndex.from_tuples(df.index, names=["Time", "Process", "Vintage"])
            df = df.reset_index()
            df_pivot = df.pivot(index="Time", columns=["Process", "Vintage"], values="Value")
            return df_pivot

    def get_operation_by_vintage(self) -> pd.DataFrame:
        """
        Get operation data broken down by vintage for detailed merit-order analysis.

        This is a convenience method equivalent to get_operation(aggregate_vintages=False).

        Returns
        -------
        pd.DataFrame
            DataFrame with Time as index and (Process, Vintage) MultiIndex columns.
            Values are the operational levels for each process-vintage combination.

        Example
        -------
        >>> pp = PostProcessor(solved_model)
        >>> op_by_vintage = pp.get_operation_by_vintage()
        >>> # See how much each vintage contributes to operation
        >>> op_by_vintage.loc[2030, ('solar_pv', 2025)]  # Operation of 2025-vintage solar at 2030
        """
        return self.get_operation(aggregate_vintages=False)

    def get_production(self) -> pd.DataFrame:
        """
        Extracts the production data from the model and returns it as a DataFrame.
        The DataFrame will have a MultiIndex with 'Process', 'Product', and
        'Time'. The values are the total production for each process and product
        at each time step.

        With 3D var_operation[p, v, t], production is summed across all active
        vintages at each time step.
        """
        production_tensor = {}
        fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)

        # Get production overrides data from model (if exists)
        production_overrides = getattr(self.m, "_production_vintage_overrides", {})
        production_overrides_index = getattr(self.m, "_production_overrides_index", frozenset())

        def get_production_value(p, r, tau, vintage):
            """Get production value, checking sparse overrides first."""
            key = (p, r, tau, vintage)
            if key in production_overrides:
                return production_overrides[key]
            return pyo.value(self.m.foreground_production[p, r, tau])

        def has_production_overrides(p, r):
            """Check if any vintage overrides exist for this process/product."""
            return (p, r) in production_overrides_index

        for p in self.m.PROCESS:
            for f in self.m.PRODUCT:
                op_start = pyo.value(self.m.process_operation_start[p])
                op_end = pyo.value(self.m.process_operation_end[p])

                for t in self.m.SYSTEM_TIME:
                    # Sum production across all active vintages at time t
                    total_production = 0
                    for (proc, v, time) in self.m.ACTIVE_VINTAGE_TIME:
                        if proc != p or time != t:
                            continue

                        # Get production rate for this vintage
                        if has_production_overrides(p, f):
                            production_rate = sum(
                                pyo.value(get_production_value(p, f, tau_op, v))
                                for tau_op in self.m.PROCESS_TIME
                                if op_start <= tau_op <= op_end
                            )
                        else:
                            production_rate = sum(
                                pyo.value(self.m.foreground_production[p, f, tau_op])
                                for tau_op in self.m.PROCESS_TIME
                                if op_start <= tau_op <= op_end
                            )

                        # Production from this vintage
                        total_production += production_rate * pyo.value(self.m.var_operation[p, v, t])

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
            if self.df_impacts is not None:
                df_impacts = self.df_impacts
            else:
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

    def get_existing_capacity(self) -> pd.DataFrame:
        """
        Extract existing (brownfield) capacity data from the model.

        Returns a DataFrame showing which processes have existing capacity,
        when they were installed, and their operational status at each time step.

        Returns
        -------
        pd.DataFrame
            DataFrame with Time as index and (Process, Type) as MultiIndex columns.
            Type can be 'existing_capacity' (total existing) or 'existing_operating'
            (existing capacity in operation phase at that time).
        """
        existing_cap_dict = getattr(self.m, "_existing_capacity_dict", {})

        if not existing_cap_dict:
            # Return empty DataFrame if no existing capacity
            return pd.DataFrame()

        data = {}
        for t in self.m.SYSTEM_TIME:
            for p in self.m.PROCESS:
                # Calculate existing capacity in operation at time t
                op_start = pyo.value(self.m.process_operation_start[p])
                op_end = pyo.value(self.m.process_operation_end[p])

                existing_operating = 0
                existing_total = 0

                for (proc, inst_year), capacity in existing_cap_dict.items():
                    if proc == p:
                        existing_total += capacity
                        tau_existing = t - inst_year
                        if op_start <= tau_existing <= op_end:
                            existing_operating += capacity

                if existing_total > 0:
                    data[(t, p, "existing_capacity")] = existing_total
                    data[(t, p, "existing_operating")] = existing_operating

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data, orient="index", columns=["Value"])
        df.index = pd.MultiIndex.from_tuples(df.index, names=["Time", "Process", "Type"])
        df = df.reset_index()
        df_pivot = df.pivot(index="Time", columns=["Process", "Type"], values="Value")

        return df_pivot

    def get_production_capacity(self) -> pd.DataFrame:
        """
        Calculate maximum available production capacity for each product at each time step.

        Capacity is determined by counting installations in their operation phase and
        multiplying by their production coefficients. This includes both new installations
        (from var_installation) and existing (brownfield) capacity.

        Note: Uses vintage-aware 4D calculation when production overrides exist,
        matching the optimizer's capacity constraint calculation.

        Returns
        -------
        pd.DataFrame
            DataFrame with Time as index and Products as columns.
            Values represent maximum production capacity (not actual production).
        """
        capacity_tensor = {}
        fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)
        existing_cap_dict = getattr(self.m, "_existing_capacity_dict", {})

        # Get production overrides data from model (if exists)
        production_overrides = getattr(self.m, "_production_vintage_overrides", {})
        production_overrides_index = getattr(self.m, "_production_overrides_index", frozenset())

        def get_production_value(p, r, tau, vintage):
            """Get production value, checking sparse overrides first."""
            key = (p, r, tau, vintage)
            if key in production_overrides:
                return production_overrides[key]
            return pyo.value(self.m.foreground_production[p, r, tau])

        def has_production_overrides(p, r):
            """Check if any vintage overrides exist for this process/product."""
            return (p, r) in production_overrides_index

        for f in self.m.PRODUCT:
            for t in self.m.SYSTEM_TIME:
                # Calculate total capacity across all processes
                total_capacity = 0

                for p in self.m.PROCESS:
                    op_start = pyo.value(self.m.process_operation_start[p])
                    op_end = pyo.value(self.m.process_operation_end[p])

                    if has_production_overrides(p, f):
                        # 4D vintage-aware capacity calculation
                        # Each vintage may have different production rates
                        process_capacity = 0

                        # New installations: sum capacity by vintage
                        for tau in self.m.PROCESS_TIME:
                            vintage = t - tau
                            if vintage in self.m.SYSTEM_TIME and op_start <= tau <= op_end:
                                # Production rate for this vintage (sum over all operating taus)
                                production_per_unit = sum(
                                    get_production_value(p, f, tau_op, vintage)
                                    for tau_op in self.m.PROCESS_TIME
                                    if op_start <= tau_op <= op_end
                                )
                                installation = pyo.value(self.m.var_installation[p, vintage])
                                process_capacity += production_per_unit * installation

                        # Existing (brownfield) capacity
                        for (proc, inst_year), capacity in existing_cap_dict.items():
                            if proc == p:
                                tau_existing = t - inst_year
                                if op_start <= tau_existing <= op_end:
                                    nearest_vintage = min(self.m.SYSTEM_TIME)
                                    production_per_unit = sum(
                                        get_production_value(p, f, tau_op, nearest_vintage)
                                        for tau_op in self.m.PROCESS_TIME
                                        if op_start <= tau_op <= op_end
                                    )
                                    process_capacity += production_per_unit * capacity

                        total_capacity += process_capacity
                    else:
                        # 3D calculation: no overrides, all vintages have same production rate
                        # Count new installations in operation phase at time t
                        installations_operating = sum(
                            pyo.value(self.m.var_installation[p, t - tau])
                            for tau in self.m.PROCESS_TIME
                            if (t - tau in self.m.SYSTEM_TIME)
                            and op_start <= tau <= op_end
                        )

                        # Add existing (brownfield) capacity in operation phase
                        for (proc, inst_year), capacity in existing_cap_dict.items():
                            if proc == p:
                                tau_existing = t - inst_year
                                if op_start <= tau_existing <= op_end:
                                    installations_operating += capacity

                        # Production capacity per installation (sum over operation phase)
                        production_per_installation = sum(
                            pyo.value(self.m.foreground_production[p, f, tau])
                            for tau in self.m.PROCESS_TIME
                            if op_start <= tau <= op_end
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

    def _extract_product_data(self, product, prod_df, capacity_df):
        """
        Extract production and capacity series for a single product.

        Parameters
        ----------
        product : str
            Product code to extract.
        prod_df : pd.DataFrame
            Production DataFrame from get_production().
        capacity_df : pd.DataFrame
            Capacity DataFrame from get_production_capacity().

        Returns
        -------
        tuple[pd.Series, pd.Series]
            (actual_production, max_capacity) series with string indices.
        """
        # Production: sum across all processes for this product
        if isinstance(prod_df.columns, pd.MultiIndex):
            production_cols = [col for col in prod_df.columns if col[1] == product]
            actual_production = prod_df[production_cols].sum(axis=1)
        else:
            actual_production = prod_df[product] if product in prod_df.columns else pd.Series(0, index=prod_df.index)

        # Capacity for this product
        max_capacity = capacity_df[product] if product in capacity_df.columns else pd.Series(0, index=capacity_df.index)

        # Convert indices to strings for consistent plotting
        actual_production = actual_production.copy()
        max_capacity = max_capacity.copy()
        actual_production.index = actual_production.index.astype(str)
        max_capacity.index = max_capacity.index.astype(str)

        return actual_production, max_capacity

    def _plot_capacity_balance_on_ax(
        self,
        ax,
        product,
        prod_df,
        capacity_df,
        annotated=True,
        show_legend=True,
        show_fill=True,
        show_title=True,
    ):
        """
        Plot production vs capacity lines on a given axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on.
        product : str
            Product code to plot.
        prod_df : pd.DataFrame
            Production DataFrame from get_production().
        capacity_df : pd.DataFrame
            Capacity DataFrame from get_production_capacity().
        annotated : bool, default=True
            If True, show human-readable names instead of codes.
        show_legend : bool, default=True
            If True, show legend on the axis.
        show_fill : bool, default=True
            If True, fill area between production and capacity.
        show_title : bool, default=True
            If True, show title with product name.
        """
        actual_production, max_capacity = self._extract_product_data(product, prod_df, capacity_df)
        product_name = self._get_name(product) if annotated else product

        x_positions = np.arange(len(actual_production.index))

        # Plot production line
        ax.plot(
            x_positions,
            actual_production.values,
            marker='o',
            linewidth=self._plot_config["line_width"],
            label='Production / Demand',
            color='#2E86AB',
            linestyle='-',
            zorder=3
        )

        # Plot capacity line
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

        # Fill area between production and capacity
        if show_fill:
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
        ax.set_ylabel(f"Quantity", fontsize=self._plot_config["fontsize"])

        if show_title:
            ax.set_title(
                f"{product_name}",
                fontsize=self._plot_config["fontsize"] + 2,
                pad=10
            )

        # Add grid
        ax.grid(
            True,
            alpha=self._plot_config["grid_alpha"],
            linestyle=self._plot_config["grid_linestyle"]
        )

        if show_legend:
            ax.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
                ncol=3,
                fontsize=self._plot_config["fontsize"] - 2,
                frameon=False
            )

    def _compute_capacity_breakdown(self, product):
        """
        Compute capacity additions, removals, and operation breakdown by process.

        Parameters
        ----------
        product : str
            Product code to compute breakdown for.

        Returns
        -------
        dict
            Dictionary with keys: capacity_additions_df, capacity_removals_df,
            existing_additions_df, existing_removals_df, operation_df.
            All DataFrames have process columns and time index.

        Note: Uses vintage-aware 4D calculation when production overrides exist.
        """
        fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)
        existing_cap_dict = getattr(self.m, "_existing_capacity_dict", {})

        # Get production overrides data from model (if exists)
        production_overrides = getattr(self.m, "_production_vintage_overrides", {})
        production_overrides_index = getattr(self.m, "_production_overrides_index", frozenset())

        def get_production_value(p, r, tau, vintage):
            """Get production value, checking sparse overrides first."""
            key = (p, r, tau, vintage)
            if key in production_overrides:
                return production_overrides[key]
            return pyo.value(self.m.foreground_production[p, r, tau])

        def has_production_overrides(p, r):
            """Check if any vintage overrides exist for this process/product."""
            return (p, r) in production_overrides_index

        capacity_additions = {p: {} for p in self.m.PROCESS}
        capacity_removals = {p: {} for p in self.m.PROCESS}
        existing_additions = {p: {} for p in self.m.PROCESS}
        existing_removals = {p: {} for p in self.m.PROCESS}
        operation = {p: {} for p in self.m.PROCESS}

        for t in self.m.SYSTEM_TIME:
            for p in self.m.PROCESS:
                op_start = pyo.value(self.m.process_operation_start[p])
                op_end = pyo.value(self.m.process_operation_end[p])

                # Base production per installation (3D, for processes without overrides)
                prod_per_inst_3d = sum(
                    pyo.value(self.m.foreground_production[p, product, tau])
                    for tau in self.m.PROCESS_TIME
                    if op_start <= tau <= op_end
                )

                if prod_per_inst_3d == 0:
                    capacity_additions[p][t] = 0
                    capacity_removals[p][t] = 0
                    existing_additions[p][t] = 0
                    existing_removals[p][t] = 0
                    operation[p][t] = 0
                    continue

                if has_production_overrides(p, product):
                    # 4D vintage-aware calculations

                    # New capacity entering operation (vintage = t - op_start)
                    t_entering = t - op_start
                    if t_entering in self.m.SYSTEM_TIME:
                        installation_entering = pyo.value(self.m.var_installation[p, t_entering])
                        prod_per_inst_vintage = sum(
                            get_production_value(p, product, tau_op, t_entering)
                            for tau_op in self.m.PROCESS_TIME
                            if op_start <= tau_op <= op_end
                        )
                        capacity_additions[p][t] = installation_entering * prod_per_inst_vintage * fg_scale
                    else:
                        capacity_additions[p][t] = 0

                    # Capacity exiting operation (vintage = t - op_end - 1)
                    t_exiting = t - op_end - 1
                    if t_exiting in self.m.SYSTEM_TIME:
                        installation_exiting = pyo.value(self.m.var_installation[p, t_exiting])
                        prod_per_inst_vintage = sum(
                            get_production_value(p, product, tau_op, t_exiting)
                            for tau_op in self.m.PROCESS_TIME
                            if op_start <= tau_op <= op_end
                        )
                        capacity_removals[p][t] = installation_exiting * prod_per_inst_vintage * fg_scale
                    else:
                        capacity_removals[p][t] = 0

                    # Existing capacity changes (use nearest vintage for rate)
                    existing_add = 0
                    existing_rem = 0
                    nearest_vintage = min(self.m.SYSTEM_TIME)
                    prod_per_inst_existing = sum(
                        get_production_value(p, product, tau_op, nearest_vintage)
                        for tau_op in self.m.PROCESS_TIME
                        if op_start <= tau_op <= op_end
                    )
                    for (proc, inst_year), capacity in existing_cap_dict.items():
                        if proc == p:
                            tau_existing = t - inst_year
                            tau_existing_prev = (t - 1) - inst_year
                            if op_start <= tau_existing <= op_end:
                                if tau_existing_prev < op_start:
                                    existing_add += capacity * prod_per_inst_existing * fg_scale
                            if tau_existing > op_end:
                                if op_start <= tau_existing_prev <= op_end:
                                    existing_rem += capacity * prod_per_inst_existing * fg_scale
                    existing_additions[p][t] = existing_add
                    existing_removals[p][t] = existing_rem

                    # Operation level - sum production across all active vintages
                    total_operation = 0
                    for (proc, v, time) in self.m.ACTIVE_VINTAGE_TIME:
                        if proc != p or time != t:
                            continue
                        # Get production rate for this vintage
                        production_rate = sum(
                            pyo.value(get_production_value(p, product, tau_op, v))
                            for tau_op in self.m.PROCESS_TIME
                            if op_start <= tau_op <= op_end
                        )
                        total_operation += production_rate * pyo.value(self.m.var_operation[p, v, t])

                    operation[p][t] = total_operation * fg_scale
                else:
                    # 3D calculation: no overrides
                    prod_per_inst = prod_per_inst_3d

                    # New capacity entering operation
                    t_entering = t - op_start
                    if t_entering in self.m.SYSTEM_TIME:
                        installation_entering = pyo.value(self.m.var_installation[p, t_entering])
                        capacity_additions[p][t] = installation_entering * prod_per_inst * fg_scale
                    else:
                        capacity_additions[p][t] = 0

                    # Capacity exiting operation
                    t_exiting = t - op_end - 1
                    if t_exiting in self.m.SYSTEM_TIME:
                        installation_exiting = pyo.value(self.m.var_installation[p, t_exiting])
                        capacity_removals[p][t] = installation_exiting * prod_per_inst * fg_scale
                    else:
                        capacity_removals[p][t] = 0

                    # Existing capacity changes
                    existing_add = 0
                    existing_rem = 0
                    for (proc, inst_year), capacity in existing_cap_dict.items():
                        if proc == p:
                            tau_existing = t - inst_year
                            tau_existing_prev = (t - 1) - inst_year
                            if op_start <= tau_existing <= op_end:
                                if tau_existing_prev < op_start:
                                    existing_add += capacity * prod_per_inst * fg_scale
                            if tau_existing > op_end:
                                if op_start <= tau_existing_prev <= op_end:
                                    existing_rem += capacity * prod_per_inst * fg_scale
                    existing_additions[p][t] = existing_add
                    existing_removals[p][t] = existing_rem

                    # Operation level - sum production across all active vintages
                    total_operation = 0
                    for (proc, v, time) in self.m.ACTIVE_VINTAGE_TIME:
                        if proc != p or time != t:
                            continue
                        total_operation += prod_per_inst * pyo.value(self.m.var_operation[p, v, t])

                    operation[p][t] = total_operation * fg_scale

        # Convert to DataFrames
        capacity_additions_df = pd.DataFrame(capacity_additions)
        capacity_additions_df.index = capacity_additions_df.index.astype(str)

        capacity_removals_df = pd.DataFrame(capacity_removals)
        capacity_removals_df.index = capacity_removals_df.index.astype(str)

        existing_additions_df = pd.DataFrame(existing_additions)
        existing_additions_df.index = existing_additions_df.index.astype(str)

        existing_removals_df = pd.DataFrame(existing_removals)
        existing_removals_df.index = existing_removals_df.index.astype(str)

        operation_df = pd.DataFrame(operation)
        operation_df.index = operation_df.index.astype(str)

        # Filter to only processes with non-zero values
        has_values = ((capacity_additions_df != 0).any(axis=0) |
                     (capacity_removals_df != 0).any(axis=0) |
                     (existing_additions_df != 0).any(axis=0) |
                     (existing_removals_df != 0).any(axis=0) |
                     (operation_df != 0).any(axis=0))
        capacity_additions_df = capacity_additions_df.loc[:, has_values]
        capacity_removals_df = capacity_removals_df.loc[:, has_values]
        existing_additions_df = existing_additions_df.loc[:, has_values]
        existing_removals_df = existing_removals_df.loc[:, has_values]
        operation_df = operation_df.loc[:, has_values]

        return {
            "capacity_additions_df": capacity_additions_df,
            "capacity_removals_df": capacity_removals_df,
            "existing_additions_df": existing_additions_df,
            "existing_removals_df": existing_removals_df,
            "operation_df": operation_df,
        }

    def _plot_capacity_balance_detailed_on_ax(
        self,
        ax,
        product,
        prod_df,
        capacity_df,
        annotated=True,
        show_legend=True,
        show_title=True,
    ):
        """
        Plot detailed capacity balance with grouped bars on a given axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on.
        product : str
            Product code to plot.
        prod_df : pd.DataFrame
            Production DataFrame.
        capacity_df : pd.DataFrame
            Capacity DataFrame.
        annotated : bool, default=True
            If True, show human-readable names.
        show_legend : bool, default=True
            If True, show legend.
        show_title : bool, default=True
            If True, show title.

        Returns
        -------
        tuple
            (process_legend, type_legend) for creating shared legends.
        """
        from matplotlib.patches import Patch

        actual_production, max_capacity = self._extract_product_data(product, prod_df, capacity_df)
        product_name = self._get_name(product) if annotated else product

        # Compute breakdown data
        breakdown = self._compute_capacity_breakdown(product)
        capacity_additions_df = breakdown["capacity_additions_df"]
        capacity_removals_df = breakdown["capacity_removals_df"]
        existing_additions_df = breakdown["existing_additions_df"]
        existing_removals_df = breakdown["existing_removals_df"]
        operation_df = breakdown["operation_df"]

        # Get colors from the shared color map (uses original process codes before annotation)
        process_codes = list(capacity_additions_df.columns)

        # Annotate DataFrames
        capacity_additions_df = self._annotate_dataframe(capacity_additions_df.copy(), annotated)
        capacity_removals_df = self._annotate_dataframe(capacity_removals_df.copy(), annotated)
        existing_additions_df = self._annotate_dataframe(existing_additions_df.copy(), annotated)
        existing_removals_df = self._annotate_dataframe(existing_removals_df.copy(), annotated)
        operation_df = self._annotate_dataframe(operation_df.copy(), annotated)

        x_positions = np.arange(len(actual_production.index))

        process_legend = []
        type_legend = []

        if not capacity_additions_df.empty:
            bar_width = 0.35
            offset = 0.20
            cap_positions = x_positions - offset
            op_positions = x_positions + offset

            # Plot capacity additions (positive, dark green border)
            bottom_additions = np.zeros(len(x_positions))
            for i, col in enumerate(capacity_additions_df.columns):
                add_values = capacity_additions_df[col].values
                clr = self._color_map.get(process_codes[i], 'black')
                ax.bar(cap_positions, add_values, width=bar_width, bottom=bottom_additions,
                       color=clr, hatch="///", edgecolor="#30A834FF", linewidth=1.5, zorder=1)
                bottom_additions += add_values

            # Plot existing capacity additions (light green border)
            for i, col in enumerate(existing_additions_df.columns):
                add_values = existing_additions_df[col].values
                clr = self._color_map.get(process_codes[i], 'black')
                ax.bar(cap_positions, add_values, width=bar_width, bottom=bottom_additions,
                       color=clr, hatch="///", edgecolor="#81C784", linewidth=1.5, zorder=1)
                bottom_additions += add_values

            # Plot capacity removals (negative, dark red border)
            bottom_removals = np.zeros(len(x_positions))
            for i, col in enumerate(capacity_removals_df.columns):
                rem_values = capacity_removals_df[col].values
                clr = self._color_map.get(process_codes[i], 'black')
                ax.bar(cap_positions, -rem_values, width=bar_width, bottom=bottom_removals,
                       color=clr, hatch="///", edgecolor="#CD221FFF", linewidth=1.5, zorder=1)
                bottom_removals -= rem_values

            # Plot existing capacity removals (light red border)
            for i, col in enumerate(existing_removals_df.columns):
                rem_values = existing_removals_df[col].values
                clr = self._color_map.get(process_codes[i], 'black')
                ax.bar(cap_positions, -rem_values, width=bar_width, bottom=bottom_removals,
                       color=clr, hatch="///", edgecolor="#E57373", linewidth=1.5, zorder=1)
                bottom_removals -= rem_values

            # Plot operation (solid bars)
            bottom_operation = np.zeros(len(x_positions))
            for i, col in enumerate(operation_df.columns):
                operation_values = operation_df[col].values
                clr = self._color_map.get(process_codes[i], 'black')
                ax.bar(op_positions, operation_values, width=bar_width, bottom=bottom_operation,
                       alpha=0.9, color=clr, edgecolor='black', linewidth=1, zorder=2)
                bottom_operation += operation_values

            ax.axhline(0, color='gray', linewidth=0.5, zorder=0)

            # Bar group labels
            ax.text(cap_positions[0], -0.08, 'Δ Cap', transform=ax.get_xaxis_transform(),
                   ha='center', va='top', fontsize=self._plot_config["fontsize"] - 3, color='gray')
            ax.text(op_positions[0], -0.08, 'Op', transform=ax.get_xaxis_transform(),
                   ha='center', va='top', fontsize=self._plot_config["fontsize"] - 3, color='gray')

            # Build legend handles
            process_legend = [Patch(facecolor=self._color_map.get(process_codes[i], 'black'), edgecolor='black', linewidth=0.5, label=col)
                            for i, col in enumerate(capacity_additions_df.columns)]
            type_legend = [
                Patch(facecolor="white", edgecolor='#30A834', linewidth=2, label='+ New Cap'),
                Patch(facecolor="white", edgecolor='#81C784', linewidth=2, label='+ Existing Cap'),
                Patch(facecolor="white", edgecolor='#CD221F', linewidth=2, label='− New Cap'),
                Patch(facecolor="white", edgecolor='#E57373', linewidth=2, label='− Existing Cap'),
            ]

        # Plot production and capacity lines
        ax.plot(x_positions, actual_production.values, marker='o',
                linewidth=self._plot_config["line_width"], label='Production / Demand',
                color='#2E86AB', linestyle='-', zorder=3)
        ax.plot(x_positions, max_capacity.values, marker='s',
                linewidth=self._plot_config["line_width"], label='Max Capacity',
                color='#A23B72', linestyle='--', zorder=3)

        self._set_smart_xticks(ax, actual_production.index)
        ax.set_ylabel(f"Quantity", fontsize=self._plot_config["fontsize"])

        if show_title:
            ax.set_title(f"{product_name}", fontsize=self._plot_config["fontsize"] + 2, pad=10)

        ax.grid(True, alpha=self._plot_config["grid_alpha"], linestyle=self._plot_config["grid_linestyle"])

        if show_legend and process_legend:
            all_handles = process_legend + type_legend
            ncol = min(6, len(all_handles))
            ax.legend(handles=all_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                     ncol=ncol, fontsize=self._plot_config["fontsize"] - 2, frameon=False)

        return process_legend, type_legend

    def plot_capacity_balance(self, product=None, prod_df=None, capacity_df=None, demand_df=None, annotated=True, detailed=False):
        """
        Plot actual production vs maximum available capacity for a specific product.

        Shows two lines:
        - Production (demand is assumed equal and overlaid)
        - Maximum available capacity (dashed line)

        When detailed=True, also shows grouped bars per time step:
        - Left bar: Capacity changes (additions/removals stacked by process)
        - Right bar: Operation level (stacked by process)

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
        detailed : bool, default=False
            If True, show grouped bars for capacity changes and operation by process
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
            products_with_demand = demand_df.columns[(demand_df != 0).any(axis=0)]
            if len(products_with_demand) == 0:
                raise ValueError("No products with non-zero demand found")
            product = products_with_demand[0]

        fig, axes = self._create_clean_axes(nrows=1, ncols=1)
        ax = axes[0]

        if detailed:
            self._plot_capacity_balance_detailed_on_ax(
                ax, product, prod_df, capacity_df,
                annotated=annotated, show_legend=True, show_title=True
            )
        else:
            self._plot_capacity_balance_on_ax(
                ax, product, prod_df, capacity_df,
                annotated=annotated, show_legend=True, show_fill=True, show_title=True
            )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25)
        plt.show()

    # Alias for backward compatibility
    plot_production_vs_capacity = plot_capacity_balance

    def plot_capacity_balance_all(self, prod_df=None, capacity_df=None, demand_df=None, annotated=True, detailed=False):
        """
        Plot production vs capacity for all products in the foreground system.

        Creates a grid of subplots, one for each product with non-zero production
        or capacity. Each subplot shows the production line and capacity line
        over time.

        When detailed=True, also shows grouped bars per time step for each product:
        - Left bar: Capacity changes (additions/removals stacked by process)
        - Right bar: Operation level (stacked by process)

        Parameters
        ----------
        prod_df : pd.DataFrame, optional
            Production DataFrame from get_production()
        capacity_df : pd.DataFrame, optional
            Capacity DataFrame from get_production_capacity()
        demand_df : pd.DataFrame, optional
            Demand DataFrame from get_demand()
        annotated : bool, default=True
            If True, show human-readable names instead of codes
        detailed : bool, default=False
            If True, show grouped bars for capacity changes and operation by process
        """
        # Get data if not provided
        if prod_df is None:
            prod_df = self.get_production()
        if capacity_df is None:
            capacity_df = self.get_production_capacity()
        if demand_df is None:
            demand_df = self.get_demand()

        # Collect all products that have either production or capacity
        all_products = set()

        # Products from production (MultiIndex columns)
        if isinstance(prod_df.columns, pd.MultiIndex):
            prod_products = set(col[1] for col in prod_df.columns)
            # Filter to products with non-zero production
            for product in prod_products:
                production_cols = [col for col in prod_df.columns if col[1] == product]
                if (prod_df[production_cols].sum(axis=1) != 0).any():
                    all_products.add(product)
        else:
            for product in prod_df.columns:
                if (prod_df[product] != 0).any():
                    all_products.add(product)

        # Products from capacity
        for product in capacity_df.columns:
            if (capacity_df[product] != 0).any():
                all_products.add(product)

        if not all_products:
            raise ValueError("No products with production or capacity found")

        # Sort products for consistent ordering
        products = sorted(all_products)

        # Determine grid dimensions
        n_products = len(products)
        ncols = min(2, n_products)
        nrows = math.ceil(n_products / ncols)

        # Calculate figure size
        base_w, base_h = self._plot_config["figsize"]
        fig_w = base_w * ncols
        fig_h = base_h * max(1, nrows * 0.8)

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharex=True
        )

        # Handle single subplot case
        if n_products == 1:
            axes = np.array([axes])
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        # Plot each product with individual legends (colors are synced via self._color_map)
        for i, product in enumerate(products):
            ax = axes[i]
            if detailed:
                self._plot_capacity_balance_detailed_on_ax(
                    ax, product, prod_df, capacity_df,
                    annotated=annotated,
                    show_legend=True,
                    show_title=True
                )
            else:
                self._plot_capacity_balance_on_ax(
                    ax, product, prod_df, capacity_df,
                    annotated=annotated,
                    show_legend=True,
                    show_fill=True,
                    show_title=True
                )

        # Hide unused axes
        for j in range(len(products), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(
            "Production vs Capacity by Product",
            fontsize=self._plot_config["fontsize"] + 4,
            y=1.02
        )

        fig.tight_layout()
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

        Note: Uses vintage-aware 4D calculation when production overrides exist.
        """
        # Get demand to determine product
        demand_df = self.get_demand()

        if product is None:
            products_with_demand = demand_df.columns[(demand_df != 0).any(axis=0)]
            if len(products_with_demand) == 0:
                raise ValueError("No products with non-zero demand found")
            product = products_with_demand[0]

        fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)
        existing_cap_dict = getattr(self.m, "_existing_capacity_dict", {})

        # Get production overrides data from model (if exists)
        production_overrides = getattr(self.m, "_production_vintage_overrides", {})
        production_overrides_index = getattr(self.m, "_production_overrides_index", frozenset())

        def get_production_value(p, r, tau, vintage):
            """Get production value, checking sparse overrides first."""
            key = (p, r, tau, vintage)
            if key in production_overrides:
                return production_overrides[key]
            return pyo.value(self.m.foreground_production[p, r, tau])

        def has_production_overrides(p, r):
            """Check if any vintage overrides exist for this process/product."""
            return (p, r) in production_overrides_index

        # Calculate utilization for each process at each time
        utilization_data = {}
        capacity_data = {}
        operation_data = {}

        for p in self.m.PROCESS:
            op_start = pyo.value(self.m.process_operation_start[p])
            op_end = pyo.value(self.m.process_operation_end[p])

            # Check if process produces this product (3D base rate)
            prod_per_inst_3d = sum(
                pyo.value(self.m.foreground_production[p, product, tau])
                for tau in self.m.PROCESS_TIME
                if op_start <= tau <= op_end
            )

            if prod_per_inst_3d == 0:
                continue  # Skip processes that don't produce this product

            utilization_data[p] = {}
            capacity_data[p] = {}
            operation_data[p] = {}

            for t in self.m.SYSTEM_TIME:
                if has_production_overrides(p, product):
                    # 4D vintage-aware calculations
                    # Capacity from new installations (sum by vintage)
                    capacity = 0
                    for tau in self.m.PROCESS_TIME:
                        vintage = t - tau
                        if vintage in self.m.SYSTEM_TIME and op_start <= tau <= op_end:
                            production_per_unit = sum(
                                get_production_value(p, product, tau_op, vintage)
                                for tau_op in self.m.PROCESS_TIME
                                if op_start <= tau_op <= op_end
                            )
                            installation = pyo.value(self.m.var_installation[p, vintage])
                            capacity += production_per_unit * installation

                    # Add existing (brownfield) capacity
                    nearest_vintage = min(self.m.SYSTEM_TIME)
                    prod_per_inst_existing = sum(
                        get_production_value(p, product, tau_op, nearest_vintage)
                        for tau_op in self.m.PROCESS_TIME
                        if op_start <= tau_op <= op_end
                    )
                    for (proc, inst_year), cap in existing_cap_dict.items():
                        if proc == p:
                            tau_existing = t - inst_year
                            if op_start <= tau_existing <= op_end:
                                capacity += prod_per_inst_existing * cap

                    capacity *= fg_scale

                    # Operation - sum production across all active vintages
                    operation = 0
                    for (proc, v, time) in self.m.ACTIVE_VINTAGE_TIME:
                        if proc != p or time != t:
                            continue
                        # Get production rate for this vintage
                        production_rate = sum(
                            pyo.value(get_production_value(p, product, tau_op, v))
                            for tau_op in self.m.PROCESS_TIME
                            if op_start <= tau_op <= op_end
                        )
                        operation += production_rate * pyo.value(self.m.var_operation[p, v, t])
                    operation *= fg_scale
                else:
                    # 3D calculation: no overrides
                    prod_per_inst = prod_per_inst_3d

                    # Calculate capacity from new installations
                    installations_operating = sum(
                        pyo.value(self.m.var_installation[p, t - tau])
                        for tau in self.m.PROCESS_TIME
                        if (t - tau in self.m.SYSTEM_TIME)
                        and op_start <= tau <= op_end
                    )

                    # Add existing (brownfield) capacity in operation phase
                    for (proc, inst_year), cap in existing_cap_dict.items():
                        if proc == p:
                            tau_existing = t - inst_year
                            if op_start <= tau_existing <= op_end:
                                installations_operating += cap

                    capacity = installations_operating * prod_per_inst * fg_scale

                    # Calculate operation - sum across all active vintages
                    operation = 0
                    for (proc, v, time) in self.m.ACTIVE_VINTAGE_TIME:
                        if proc != p or time != t:
                            continue
                        operation += prod_per_inst * pyo.value(self.m.var_operation[p, v, t])
                    operation *= fg_scale

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
