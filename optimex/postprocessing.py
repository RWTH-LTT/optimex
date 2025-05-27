import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo


class PostProcessor:
    def __init__(self, solved_model: pyo.ConcreteModel):
        self.m = solved_model
        self._plot_config = {
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

    def _create_clean_ax(self):
        fig, ax = plt.subplots(figsize=self._plot_config["figsize"])
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
        return fig, ax

    def get_impacts(self) -> pd.DataFrame:
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

    def get_installation(self) -> pd.DataFrame:
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
        production_tensor = {}
        fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)

        for p in self.m.PROCESS:
            for f in self.m.FUNCTIONAL_FLOW:
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
            df.index, names=["Process", "Functional Flow", "Time"]
        )
        df = df.reset_index()
        df_pivot = df.pivot(
            index="Time", columns=["Process", "Functional Flow"], values="Value"
        )
        self.df_production = df_pivot
        return self.df_production

    def get_demand(self) -> pd.DataFrame:
        fg_scale = getattr(self.m, "scales", {}).get("foreground", 1.0)
        demand_matrix = {
            (f, t): self.m.demand[f, t] * fg_scale
            for f in self.m.FUNCTIONAL_FLOW
            for t in self.m.SYSTEM_TIME
        }
        df = pd.DataFrame.from_dict(demand_matrix, orient="index", columns=["Value"])
        df.index = pd.MultiIndex.from_tuples(
            df.index, names=["Functional Flow", "Time"]
        )
        df = df.reset_index()
        df_pivot = df.pivot(index="Time", columns="Functional Flow", values="Value")
        self.df_demand = df_pivot
        return self.df_demand

    def plot_production_and_demand(self, prod_df=None, demand_df=None):
        fig, ax = self._create_clean_ax()
        if prod_df is None:
            prod_df = self.get_production()
        if demand_df is None:
            demand_df = self.get_demand()
        prod_df = prod_df.copy()
        demand_df = demand_df.copy()

        # Ensure string index for x-axis labels
        prod_df.index = prod_df.index.astype(str)
        demand_df.index = demand_df.index.astype(str)

        import numpy as np

        x_positions = np.arange(len(prod_df.index))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            prod_df.index,
            rotation=self._plot_config["rotation"],
            ha="right",
            fontsize=10,
        )

        # Plot stacked bar chart for production
        prod_df.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            edgecolor="black",
            width=self._plot_config["bar_width"],
            color=self._plot_config["colormap"][: len(prod_df.columns)],
            legend=False,  # We'll handle legend below
        )

        # Plot demand as line(s)
        for col in demand_df.columns:
            ax.plot(
                x_positions,
                demand_df[col].values,
                marker=self._plot_config["line_marker"],
                linewidth=self._plot_config["line_width"],
                label=f"Demand {col}",
                color=self._plot_config["line_color"],
            )

        # Custom legend combining both
        handles1, labels1 = ax.get_legend_handles_labels()

        ax.legend(
            handles=handles1,
            loc="upper left",
            fontsize=10,
            title="Processes / Demand",
            title_fontsize=12,
            frameon=False,
        )

        ax.set_title("Production and Demand", fontsize=16)
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
        fig, ax = self._create_clean_ax()
        df_installation.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            width=self._plot_config["bar_width"],
            color=self._plot_config["colormap"][: len(df_installation.columns)],
            edgecolor="black",
            legend=False,
        )
        ax.set_title("Installed Capacity", fontsize=self._plot_config["fontsize"] + 2)
        ax.set_ylabel("Installation", fontsize=self._plot_config["fontsize"])
        ax.grid(
            axis="y",
            linestyle=self._plot_config["grid_linestyle"],
            alpha=self._plot_config["grid_alpha"],
        )
        ax.set_xticklabels(
            df_installation.index.astype(str),
            rotation=self._plot_config["rotation"],
            ha="right",
        )

        # Legend outside top-right
        ax.legend(
            title="Processes",
            fontsize=self._plot_config["fontsize"] - 2,
            loc="upper right",
            frameon=False,
            bbox_to_anchor=(1.15, 1),
        )
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
        fig, ax = self._create_clean_ax()
        df_operation.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            width=self._plot_config["bar_width"],
            color=self._plot_config["colormap"][: len(df_operation.columns)],
            edgecolor="black",
            legend=False,
        )
        ax.set_title("Operational Level", fontsize=self._plot_config["fontsize"] + 2)
        ax.set_ylabel("Operation", fontsize=self._plot_config["fontsize"])
        ax.grid(
            axis="y",
            linestyle=self._plot_config["grid_linestyle"],
            alpha=self._plot_config["grid_alpha"],
        )
        ax.set_xticklabels(
            df_operation.index.astype(str),
            rotation=self._plot_config["rotation"],
            ha="right",
        )

        # Legend outside top-right
        ax.legend(
            title="Processes",
            fontsize=self._plot_config["fontsize"] - 2,
            loc="upper right",
            frameon=False,
            bbox_to_anchor=(1.15, 1),
        )
        fig.tight_layout()
        plt.show()
        return fig, ax
