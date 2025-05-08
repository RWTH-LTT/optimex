import logging
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import bw2calc as bc
import bw2data as bd
import numpy as np
import pandas as pd
from dynamic_characterization import characterize
from tqdm import tqdm


class LCADataProcessor:
    """
    Class to perform time-explicit Life Cycle Assessment (LCA)
    computations and gather necessary data for building an optimization model.

    This class is primarily responsible for executing the LCA-based computations
    required to collect all the data needed for building `ModelInputs`. It is reliant on
    Brightway2, an open-source framework for Life Cycle Assessment, to perform the
    calculations and retrieve LCA results.
    """

    def __init__(
        self,
        demand: dict,
        start_date: datetime,
        method: tuple,
        database_date_dict: dict,
        temporal_resolution: str = "year",
        timehorizon: int = 100,
    ) -> None:
        """
        Initialize the LCADataProcessor class to compute and collect LCA data
        for creating input to an optimization model. This class is designed to
        gather the necessary LCA data based on processes and their intermediate
        and elementary flows that produce outputs matching the required demand.

        For unique identification of processes, the class uses the `code`
        attribute of a `bw2data.parameters.ActivityParameter` instance.

        Parameters
        ----------
        demand : dict
            A dictionary containing time-explicit demands for each flow.
        start_date : datetime
            The start date for the time horizon, used to calculate future years.
        method : tuple
            A tuple defining the LCIA method (e.g.,
            `("EF v3.1", "climate change", "global warming potential (GWP100)")`).
        database_date_dict : dict
            A dictionary mapping database names to their respective dates.
        temporal_resolution : str, optional
            The temporal resolution for the optimization model (e.g., "year").
            Default is "year".
        timehorizon : int, optional
            The length of the time horizon in years. Default is 100 years.
        """
        # Store the provided values as instance variables
        self.demand_raw = demand
        self.timehorizon = timehorizon
        self.method = method

        if temporal_resolution != "year":
            raise NotImplementedError("Only 'year' is currently supported.")
        self.start_date = start_date

        # Extract dynamic database(s)
        dynamic_dbs = {
            db for db, date in database_date_dict.items() if date == "dynamic"
        }

        if len(dynamic_dbs) != 1:
            raise ValueError("There should be exactly one dynamic database.")
        self.foreground_db = bd.Database(dynamic_dbs.pop())
        self.biosphere_db = bd.Database(bd.config.biosphere)

        # Remove foreground DB from background DBs dictionary
        self.background_dbs = {
            db: date
            for db, date in database_date_dict.items()
            if db != self.foreground_db.name
        }

        # Dictionaries for optimization model

        self._processes = {}  # dict: {process code: process name}
        self._intermediate_flows = (
            {}
        )  # dict: {intermediate flow code: intermediate flow name}
        self._elementary_flows = (
            {}
        )  # dict: {elementary flow code: elementary flow name}
        self._functional_flows = set()  # set: functional flow names
        self._system_time = set(
            range(self.start_date.year, self.start_date.year + self.timehorizon)
        )  # set: absolute time points
        self._process_time = set()  # set: relative process time points

        # Tensors/matrices for optimization model
        self._demand = {}
        self._foreground_technosphere = {}
        self._foreground_biosphere = {}
        self._foreground_production = {}
        self._background_inventory = {}
        self._mapping = {}
        self._characterization = {}

        self._process_operation_time = {}  # dict: {process code: (start, end)}

    @property
    def processes(self) -> dict:
        """Read-only access to the processes dictionary."""
        return self._processes

    @property
    def intermediate_flows(self) -> dict:
        """Read-only access to the intermediate flows dictionary."""
        return self._intermediate_flows

    @property
    def elementary_flows(self) -> dict:
        """Read-only access to the elementary flows dictionary."""
        return self._elementary_flows

    @property
    def functional_flows(self) -> set:
        """Read-only access to the functional flows list."""
        return self._functional_flows

    @property
    def system_time(self) -> set:
        """Read-only access to the system time list."""
        return self._system_time

    @property
    def process_time(self) -> set:
        """Read-only access to the process time list."""
        return self._process_time

    @property
    def foreground_technosphere(self) -> dict:
        """Read-only access to the foreground technosphere tensor."""
        return self._foreground_technosphere

    @property
    def foreground_biosphere(self) -> dict:
        """Read-only access to the foreground biosphere tensor."""
        return self._foreground_biosphere

    @property
    def foreground_production(self) -> dict:
        """Read-only access to the foreground production tensor."""
        return self._foreground_production

    @property
    def background_inventory(self) -> dict:
        """Read-only access to the inventory tensor."""
        return self._background_inventory

    @property
    def mapping(self) -> dict:
        """Read-only access to the mapping matrix."""
        return self._mapping

    @property
    def characterization(self) -> dict:
        """Read-only access to the characterization matrix."""
        return self._characterization

    @property
    def demand(self) -> dict:
        """Read-only access to the parsed demand dictionary."""
        return self._demand

    @property
    def process_operation_time(self) -> dict:
        """Read-only access to the process operation time dictionary."""
        return self._process_operation_time

    def parse_demand(self) -> dict:
        """
        Parse the demand dictionary into a format that can be used by the optimization
        model.

        This method creates a new dictionary that maps each flow and year to the
        corresponding amount from the demand data. The year is extracted from the 'date'
        field, and the amount is associated with the corresponding flow and year.

        Returns
        -------
        dict
            Dictionary containing the parsed demand with (flow, year) as the key and
            amount as the value.
        """

        for flow, td in self.demand_raw.items():
            years = td.date.astype("datetime64[Y]").astype(int) + 1970
            amounts = td.amount

            # Create a dictionary of (flow, year) -> amount
            self._demand.update(
                {(flow, year): amount for year, amount in zip(years, amounts)}
            )
            self._functional_flows.add(flow)

        return self._demand

    def construct_foreground_tensors(self) -> tuple[dict, dict, dict]:
        """
        Constructs the foreground technosphere, biosphere, and production tensors by
        expanding the standard techno- and biosphere matrices with a new dimension:
        process time (tau). These tensors represent the amounts of flows associated with
        processes in the system, distributed over time according to the temporal
        distributions of the exchanges.

        The method iterates through the processes in the foreground database and, for
        each exchange, collects temporal distribution data (i.e., years and amounts).
        This data is then aggregated into three separate tensors that represent
        different types of flows.

        Parameters
        ----------
        None

        Returns
        -------
        tuple of dict
            A tuple containing:
            - Foreground Technosphere Tensor: A dictionary where the keys are tuples
            of (process_code, flow_code, year) and the values are the corresponding
            amounts for each flow in the technosphere.
            - Foreground Biosphere Tensor: A dictionary where the keys are tuples
            of (process_code, flow_code, year) and the values are the corresponding
            amounts for each elementary flow to the biosphere.
            - Foreground Production Tensor: A dictionary where the keys are tuples
            of (process_code, functional_flow, year) and the values are the
            corresponding amounts for each produced functional flow.

        Notes
        -----
        Conventionally, functional flows are a subset of intermediate flows and are
        included within the technosphere matrix. However, for clarity and easier
        downstream processing, we explicitly extract functional flows and store them in
        a separate production tensor.
        """

        technosphere_tensor = {}
        production_tensor = {}
        biosphere_tensor = {}

        for act in self.foreground_db:
            # Only process activities present in the demand
            if act["functional flow"] not in self.demand_raw.keys():
                continue

            # Store process information
            self._processes.setdefault(act["code"], act["name"])

            for exc in act.exchanges():
                # Extract temporal distribution
                temporal_dist = exc.get("temporal_distribution", {})
                years = temporal_dist.date.astype("timedelta64[Y]").astype(int)
                # Ensure all years are included in process time
                self._process_time.update(
                    year for year in years if year not in self._process_time
                )
                temporal_factor = temporal_dist.amount

                # Skip if temporal distribution is missing or invalid
                if not years.any() or not temporal_factor.any():
                    continue

                # Determine the tensor type (technosphere or biosphere)
                type = exc["type"]
                input_code = exc.input["code"]
                input_name = exc.input["name"]

                # Update tensors and flow dictionaries
                if type == "technosphere":
                    technosphere_tensor.update(
                        {
                            (act["code"], input_code, year): exc["amount"] * factor
                            for year, factor in zip(years, temporal_factor)
                        }
                    )
                    self._intermediate_flows.setdefault(input_code, input_name)
                elif type == "biosphere":
                    biosphere_tensor.update(
                        {
                            (act["code"], input_code, year): exc["amount"] * factor
                            for year, factor in zip(years, temporal_factor)
                        }
                    )
                    self._elementary_flows.setdefault(input_code, input_name)
                elif type == "production":
                    production_tensor.update(
                        {
                            (act["code"], act["functional flow"], year): exc["amount"]
                            * factor
                            for year, factor in zip(years, temporal_factor)
                        }
                    )

        # Store the tensors as protected variables
        self._foreground_technosphere = technosphere_tensor
        self._foreground_biosphere = biosphere_tensor
        self._foreground_production = production_tensor

        return technosphere_tensor, biosphere_tensor, production_tensor

    def set_process_operation_time(self, operation_time: dict[str, tuple[int, int]]):
        """
        Set the user-defined operation time for processes.

        Parameters
        ----------
        operation_time : dict of str to (int, int)
            A dictionary mapping process codes to a tuple representing
            their (start_time, end_time). Both start and end must be integers.

        Raises
        ------
        ValueError
            If the input is not properly structured or contains invalid values.
        """
        for proc_code, period in operation_time.items():
            if not isinstance(proc_code, str):
                raise ValueError("Process code must be a string.")
            if (
                not isinstance(period, tuple)
                or len(period) != 2
                or not all(isinstance(t, int) for t in period)
            ):
                raise ValueError(
                    "Each value must be a tuple of two integers (start, end)."
                )
            if period[0] > period[1]:
                raise ValueError(
                    f"Start time must not exceed end time for process '{proc_code}'."
                )
        self._process_operation_time = operation_time

    def _calculate_inventory_of_db(self, db_name, intermediate_flows, method, cutoff):
        """
        Calculate the life cycle inventory for a specified background database.

        Performs an LCA for each intermediate flow exchanged with the given database
        using the specified LCIA method. Intermediate flows are mapped to resulting
        elementary flows to construct an inventory tensor. A cutoff threshold is
        applied to filter insignificant results.

        Parameters
        ----------
        db_name : str
            Name of the background database to analyze.
        intermediate_flows : dict
            Dictionary mapping intermediate flow codes to flow names.
        method : tuple
            LCIA method used to compute environmental impacts (e.g.,
            ("EF v3.1", "climate change", "GWP100")).
        cutoff : float
            Number of top elementary flows (per intermediate flow) to retain based on
            impact magnitude. Used to reduce computational complexity.

        Returns
        -------
        inventory_tensor : dict
            Dictionary with keys as (db_name, intermediate_flow_code,
            elementary_flow_code) and values as flow amounts.
        elementary_flows : dict
            Dictionary mapping elementary flow codes to their names.
        """

        logging.info(f"Calculating inventory for database: {db_name}")
        db = bd.Database(name=db_name)
        inventory_tensor = {}
        elementary_flows = {}
        activity_cache = {}

        # Cache activity objects by looking up intermediate flows in the database
        for key in intermediate_flows.keys():
            try:
                activity_cache[key] = db.get(code=key)
            except Exception as e:  # Catch exceptions (e.g., if key is not valid)
                logging.warning(f"Failed to get activity for key '{key}': {e}")
        function_unit_dict = {activity: 1 for activity in activity_cache.values()}

        lca = bc.LCA(function_unit_dict, method)
        lca.lci(factorize=len(function_unit_dict) > 10)  # factorize if many activities
        logging.info(f"Factorized LCI for database: {db_name}")
        for intermediate_flow_code, activity in tqdm(activity_cache.items()):
            # logging.info(f"Calculating inventory for activity: {activity}")
            lca.redo_lci({activity.id: 1})
            if lca.inventory.nnz == 0:
                logging.warning(
                    f"Skipping activity {activity} as it has no non-zero inventory."
                )
                continue
            raw_inventory_df = lca.to_dataframe(matrix_label="inventory", cutoff=cutoff)

            inventory_df = (
                raw_inventory_df.groupby("row_code", as_index=False)
                .agg({"amount": "sum"})
                .merge(
                    raw_inventory_df[["row_code", "row_name"]].drop_duplicates(
                        "row_code"
                    ),
                    on="row_code",
                )
            )

            # Vectorized updates to `inventory_tensor`
            inventory_tensor.update(
                {
                    (db_name, intermediate_flow_code, elementary_flow_code): amount
                    for elementary_flow_code, amount in zip(
                        inventory_df["row_code"], inventory_df["amount"]
                    )
                }
            )

            # Vectorized updates to `elementary_flows`
            elementary_flows.update(
                dict(zip(inventory_df["row_code"], inventory_df["row_name"]))
            )
        logging.info(f"Finished calculating inventory for database: {db_name}")
        return inventory_tensor, elementary_flows

    def parallel_inventory_tensor_calculation(self, cutoff=1e4, n_jobs=None) -> dict:
        """
        Not yet implemented. Could improve performance significantly by parallelizing
        """
        raise NotImplementedError("This method is not yet functionally implemented.")

        # Define a function to wrap the call to _calculate_inventory_of_db
        def process_db(db_name):
            try:
                # Call the _calculate_inventory_of_db method for each db
                inventory_tensor, elementary_flows = self._calculate_inventory_of_db(
                    db_name, self._intermediate_flows, self.method, cutoff
                )
                return inventory_tensor, elementary_flows
            except Exception as e:
                logging.error(
                    f"Error occurred while processing database {db_name}: {str(e)}"
                )
                return None, None  # Return None for failed jobs

        results = []

        # Use ProcessPoolExecutor to parallelize the processing of databases
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit tasks to the executor
            future_to_db = {
                executor.submit(process_db, db_name): db_name
                for db_name in self.background_dbs
            }

            # Wait for results to be completed
            for future in as_completed(future_to_db):
                db_name = future_to_db[future]
                try:
                    inventory_tensor, elementary_flows = future.result()
                    if inventory_tensor is not None and elementary_flows is not None:
                        results.append((inventory_tensor, elementary_flows))
                except Exception as e:
                    logging.error(f"Error processing database {db_name}: {e}")

        # Combine results from all databases
        for inventory_tensor, elementary_flows in results:
            self._background_inventory.update(inventory_tensor)
            self._elementary_flows.update(elementary_flows)

        return self._background_inventory

    def sequential_inventory_tensor_calculation(self, cutoff=1e4) -> dict:
        """
        Calculate the inventory tensor for all background databases sequentially.

        For each background database, this method performs LCA calculations and
        constructs an inventory matrix. The results are aggregated into a single
        dictionary representing the background inventory tensor.

        See also `parallel_inventory_tensor_calculation` for a parallelized version.

        Parameters
        ----------
        cutoff : float, optional
            Maximum number of elementary flows to retain per activity based on
            total impact (from `lca.to_dataframe()`). Defaults to 1e4, meaning
            only the top 10,000 flows ordered by impact score will be included.

        Returns
        -------
        dict
            A dictionary containing the combined inventory tensor for all
            background databases.
        """
        results = []

        # Iterate over each database in self.background_dbs sequentially
        for db_name in self.background_dbs:
            try:
                # Directly call the _calculate_inventory_of_db method for each db
                inventory_tensor, elementary_flows = self._calculate_inventory_of_db(
                    db_name, self._intermediate_flows, self.method, cutoff
                )
                # Store the result in the results list
                results.append((inventory_tensor, elementary_flows))

            except Exception as e:
                logging.error(
                    f"Error occurred while processing database {db_name}: {str(e)}"
                )
                continue  # Continue with the next database if one fails

        # Combine results from all databases
        for inventory_tensor, elementary_flows in results:
            self._background_inventory.update(inventory_tensor)
            self._elementary_flows.update(elementary_flows)

        return self._background_inventory

    def load_inventory_tensors(self, file_path: str) -> None:
        """
        Load inventory tensors from a pickle file.

        This method unpickles pre-computed inventory tensors and updates the
        internal background inventory and elementary flow mappings.

        .. warning::
            Only unpickle data you trust. Loading pickle files from untrusted
            sources can be insecure.

        Parameters
        ----------
        file_path : str
            Path to the pickle file containing the saved inventory tensors.
        """

        # Load the inventory tensor from the pickle file
        with open(file_path, "rb") as file:
            inventory_tensor = pickle.load(file)

        # Update the background inventory with the loaded tensor
        self._background_inventory = inventory_tensor

        # Update the elementary flows dictionary with names from the biosphere database
        for key in inventory_tensor.keys():
            _, _, elementary_flow_code = key
            if elementary_flow_code not in self._elementary_flows:
                self._elementary_flows[elementary_flow_code] = self.biosphere_db.get(
                    code=elementary_flow_code
                )["name"]
        return self._background_inventory

    def save_inventory_tensors(self, file_path: str) -> None:
        """
        Save the inventory tensors to a pickle file for later use.

        Parameters
        ----------
        file_path : str
            The path to the pickle file where the inventory tensors will be saved.
        """
        with open(file_path, "wb") as file:
            pickle.dump(self._background_inventory, file)

    def construct_mapping_matrix(self) -> dict:
        """
        Construct the mapping matrix that links the background databases to the system
        time points. The matrix is constructed with linear interpolation between
        background databases.

        Returns
        -------
        dict
            Dictionary containing the mapping matrix with keys as tuples (year, db_key).
        """
        # Generate a list of datetime objects for the years in the range
        years = [self.start_date.year + i for i in range(self.timehorizon)]

        # Sort database entries by their year (extract year from datetime)
        db_names = sorted(
            self.background_dbs.keys(), key=lambda k: self.background_dbs[k].year
        )
        db_years = [self.background_dbs[k].year for k in db_names]

        # Initialize an empty list to hold matrix rows
        matrix = []

        for year in years:
            row = np.zeros(len(db_names))

            # Handle edge cases for the first and last database years
            if year <= db_years[0]:
                row[0] = 1.0  # All weight to the first DB
            elif year >= db_years[-1]:
                row[-1] = 1.0  # All weight to the last DB
            else:
                # Linear interpolation between databases
                for k in range(len(db_years) - 1):
                    if db_years[k] <= year <= db_years[k + 1]:
                        t0, t1 = db_years[k], db_years[k + 1]
                        w1 = (year - t0) / (t1 - t0)
                        w0 = 1 - w1
                        row[k] = w0
                        row[k + 1] = w1
                        break
            matrix.append(row)

        # Create a dictionary with tupled keys (year, db_name)
        mapping_matrix = {
            (db_names[j], years[i]): matrix[i][j]
            for i in range(len(years))
            for j in range(len(db_names))
        }

        self._mapping = mapping_matrix

        return mapping_matrix

    def construct_characterization_matrix(self, dynamic=True, metric="GWP") -> dict:
        """
        Construct the dynamic characterization matrix.

        The characterization matrix maps elementary flows to impact factors for each
        system time point. This implementation currently supports only the Global
        Warming Potential (GWP) metric.

        Parameters
        ----------
        dynamic : bool, optional
            Whether to compute a dynamic characterization matrix over time. If False,
            a static (time-invariant) matrix is used. Default is True.
        metric : str, optional
            The impact assessment metric to use. Currently, only "GWP" is supported.
            Default is "GWP".

        Returns
        -------
        dict
            A dictionary where keys are tuples of (elementary_flow_code, year) and
            values are the corresponding characterization factors.
        """

        # TODO: Add support for radiative forcing

        # Create a DataFrame based on self.elementary_flows
        df = pd.DataFrame({"code": list(self.elementary_flows.keys())})

        dates = pd.date_range(
            start=self.start_date, periods=self.timehorizon, freq="YE"
        )

        # Get the id of each flow and add it to the DataFrame
        df["flow"] = df["code"].map(lambda code: self.biosphere_db.get(code=code).id)
        if dynamic:
            df["amount"] = 1
            df["activity"] = np.nan

            df = df.loc[np.repeat(df.index, len(dates))].reset_index(drop=True)

            # Tile the dates and ensure 'date' is in datetime format
            df["date"] = np.tile(dates, len(df) // len(dates))
            df["date"] = df["date"].astype("datetime64[s]")

            # dynamic characterization needs dates in datetime64 format
            df_characterized = characterize(
                df,
                metric=metric,
                fixed_time_horizon=True,
                base_lcia_method=self.method,
                time_horizon=self.timehorizon,
            )

            df_characterized["date"] = df_characterized["date"].dt.to_pydatetime()
            characterization_matrix = {
                (
                    df.loc[df["flow"] == row["flow"], "code"].values[0],
                    row["date"].year,
                ): row["amount"]
                for _, row in df_characterized.iterrows()
            }
        else:
            method_data = bd.Method(self.method).load()
            # Preprocess method_data into a dictionary for fast lookups
            method_data_dict = {
                flow: value for flow, value in method_data if value != 0
            }
            characterization_matrix = {
                (row["code"], year): method_data_dict[row["flow"]]
                for _, row in df.iterrows()
                for year in dates.year
                if row["flow"] in method_data_dict
            }

        self._characterization = characterization_matrix

        return characterization_matrix
