import matplotlib.pyplot as plt
import pandas as pd
import pyomo.environ as pyo


class PostProcessor:
    def __init__(self, solved_model: pyo.ConcreteModel):
        raise NotImplementedError(
            "PostProcessor is currently disabled due to major refactoring. "
            "This class will be updated in a future version."
        )
        self.m = solved_model
        self.df_scaling = None
        self.df_production = None
        self.df_demand = None

    def get_impact(self) -> float:
        return self.m.OBJ()

    def get_scaling(self) -> pd.DataFrame:
        scaling_matrix = {
            (t, p): self.m.scaling[p, t].value
            for p in self.m.PROCESS
            for t in self.m.SYSTEM_TIME
        }
        df = pd.DataFrame.from_dict(scaling_matrix, orient="index", columns=["Value"])
        df.index = pd.MultiIndex.from_tuples(df.index, names=["Time", "Process"])
        df = df.reset_index()
        df_pivot = df.pivot(index="Time", columns="Process", values="Value")
        self.df_scaling = df_pivot
        return self.df_scaling

    def plot_scaling(self, plot_name: str, df_scaling: pd.DataFrame = None):
        if df_scaling is None:
            df_scaling = self.df_scaling

        core_processes = {
            col.split("_")[2]: col for col in df_scaling.columns
        }  # Mapping core name to column
        unique_processes = sorted(
            set(core_processes.keys())
        )  # Unique process names (without prefixes)
        colormap = plt.colormaps["tab20"]  # A palette with more unique colors
        color_map = {
            process: colormap(i / len(unique_processes))
            for i, process in enumerate(unique_processes)
        }
        categories = [
            f[:4] for f in self.m.FUNCTIONAL_FLOW
        ]  # Extract the first four letters of each functional flow

        # Function to get the color for a process based on its core name
        def get_color(process):
            core_name = process.split("_")[2]  # Extract the core name (without prefix)
            return color_map[core_name]

        # Create horizontal subplots
        fig, axes = plt.subplots(1, len(categories), figsize=(14, 6), sharey=True)

        for ax, category in zip(axes, categories):
            # Identify corresponding processes
            columns = [
                col for col in df_scaling.columns if col.startswith(f"{category}_prod_")
            ]

            # Plot each category
            df_scaling[columns].plot(
                kind="bar",
                stacked=True,
                ax=ax,
                color=[get_color(col) for col in columns],
                legend=False,  # Suppress legend to add custom ones later
            )
            # Apply best practices
            ax.set_title(
                f"Scaling Factors for {category.capitalize()} Production",
                fontsize=14,
                pad=10,
            )
            ax.set_xlabel("Time", fontsize=12)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

            # Add simplified legend
            handles, labels = ax.get_legend_handles_labels()
            # Remove first two prefixes from the process names for the legend
            simplified_labels = [
                process.split("_", 2)[-1].replace("_", " ").capitalize()
                for process in columns
            ]
            ax.legend(
                handles,
                simplified_labels,
                title="Process Type",
                fontsize=10,
                loc="upper right",
                frameon=False,
            )

        # Final layout adjustments
        axes[0].set_ylabel("Scaling Factor", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()

        # Save and show the plot
        plt.savefig(plot_name, dpi=300, bbox_inches="tight")
        return fig

    def get_production(self) -> pd.DataFrame:
        production_matrices = {}
        self.dfs_production = {}
        for f in self.m.FUNCTIONAL_FLOW:  # Loop over factors f
            production_matrices[f] = {}

            for p in self.m.PROCESS:
                for t in self.m.SYSTEM_TIME:
                    total_production = 0
                    for tau in self.m.PROCESS_TIME:
                        if t - tau in self.m.SYSTEM_TIME:
                            total_production += (
                                self.m.foreground_production[p, f, tau]
                                * self.m.scaling[p, t - tau].value
                            )
                    production_matrices[f][(p, t)] = total_production

            df = pd.DataFrame.from_dict(
                production_matrices[f], orient="index", columns=["Value"]
            )
            df.index = pd.MultiIndex.from_tuples(df.index, names=["Process", "Time"])
            df = df.reset_index()
            df_pivot = df.pivot(index="Time", columns="Process", values="Value")
            self.dfs_production[f] = df_pivot
        return self.dfs_production

    def get_demand(self) -> pd.DataFrame:
        demand_matrix = {
            (f, t): self.m.demand[f, t]
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

    def plot_production(self, plot_name: str, dfs_production: pd.DataFrame = None):
        if dfs_production is None:
            dfs_production = self.dfs_production
        if self.df_demand is None:
            self.df_demand = self.get_demand()

        core_processes = {
            col.split("_")[2]: col for col in self.m.PROCESS
        }  # Mapping core name to process
        unique_processes = sorted(
            set(core_processes.keys())
        )  # Unique process names (without prefixes)
        colormap = plt.colormaps["tab20"]  # A palette with more unique colors
        color_map = {
            process: colormap(i / len(unique_processes))
            for i, process in enumerate(unique_processes)
        }
        categories = {
            f: f[:4] for f in self.m.FUNCTIONAL_FLOW
        }  # Map first four letters to functional flows

        # Function to get the color for a process based on its core name
        def get_color(process):
            core_name = process.split("_")[2]  # Extract the core name (without prefix)
            return color_map[core_name]

        # Create horizontal subplots
        fig, axes = plt.subplots(1, len(categories), figsize=(14, 6), sharey=True)

        for ax, f in zip(axes, categories):
            # Identify corresponding processes
            df_production = dfs_production[f]
            columns = [
                col
                for col in df_production.columns
                if col.startswith(f"{categories[f]}_prod_")
            ]

            # Plot each category
            df_production[columns].plot(
                kind="bar",
                stacked=True,
                ax=ax,
                color=[get_color(col) for col in columns],
                legend=False,  # Suppress legend to add custom ones later
                zorder=1,
            )
            line_plot = self.df_demand[f].plot(
                kind="line",
                ax=ax,
                color="black",
                linewidth=2,
                legend=False,
                zorder=2,
            )
            # Apply best practices
            ax.set_title(f"Production for {f.capitalize()}", fontsize=14, pad=10)
            ax.set_xticks(ax.get_xticks()[::5])
            ax.set_xlabel("Time", fontsize=12)
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

            # Add simplified legend
            handles, labels = ax.get_legend_handles_labels()
            handles += [line_plot]
            # Remove first two prefixes from the process names for the legend
            simplified_labels = ["Demand"]
            simplified_labels += [
                process.split("_", 2)[-1].replace("_", " ").capitalize()
                for process in columns
            ]
            ax.legend(
                handles,
                simplified_labels,
                title="Process Type",
                fontsize=10,
                loc="upper right",
                frameon=False,
            )

        # Final layout adjustments
        axes[0].set_ylabel("Production", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()

        # Save and show the plot
        plt.savefig(plot_name, dpi=300, bbox_inches="tight")
        return fig

    def get_specific_impacts(self) -> pd.DataFrame:
        # precompute expression values
        self.scaled_technosphere_values = {
            (p, i, t): pyo.value(self.m.scaled_technosphere[p, i, t])
            for p in self.m.PROCESS
            for i in self.m.INTERMEDIATE_FLOW
            for t in self.m.SYSTEM_TIME
        }
        self.scaled_biosphere_values = {
            (p, e, t): pyo.value(self.m.scaled_biosphere[p, e, t])
            for p in self.m.PROCESS
            for e in self.m.ELEMENTARY_FLOW
            for t in self.m.SYSTEM_TIME
        }
        self.time_process_specific_impact_values = {
            (p, t): pyo.value(self.m.time_process_specific_impact(p, t))
            for p in self.m.PROCESS
            for t in self.m.SYSTEM_TIME
        }
        impact_matrix = {
            (t, self.m.process_names[p]): self.time_process_specific_impact_values(
                self.m, p, t
            )
            for p in self.m.PROCESS
            for t in self.m.SYSTEM_TIME
        }
        df = pd.DataFrame.from_dict(impact_matrix, orient="index", columns=["Value"])
        df.index = pd.MultiIndex.from_tuples(df.index, names=["Time", "Process"])
        df = df.reset_index()
        df_pivot = df.pivot(index="Time", columns="Process", values="Value")
        return df_pivot

    # TODO: calculate oversupply of functional flows
