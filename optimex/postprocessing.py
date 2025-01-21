import pandas as pd
import pyomo.environ as pyo


class PostProcessor:
    def __init__(self, solved_model: pyo.ConcreteModel):
        self.m = solved_model
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

    def get_scaling(self) -> pd.DataFrame:
        scaling_matrix = {
            (t, self.m.process_names[p]): self.m.scaling[p, t].value
            for p in self.m.PROCESS
            for t in self.m.SYSTEM_TIME
        }
        df = pd.DataFrame.from_dict(scaling_matrix, orient="index", columns=["Value"])
        df.index = pd.MultiIndex.from_tuples(df.index, names=["Time", "Process"])
        df = df.reset_index()
        df_pivot = df.pivot(index="Time", columns="Process", values="Value")
        df_pivot = df_pivot[(df_pivot.T != 0).any()]
        self.df_scaling = df_pivot
        return self.df_scaling

    def plot_scaling(self):
        self.df_scaling.plot(
            kind="bar",
            stacked=True,
            title="Scaling Factors",
            ylabel="Scaling Factor",
            xlabel="Time",
        )

    def get_specific_impacts(self) -> pd.DataFrame:
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
