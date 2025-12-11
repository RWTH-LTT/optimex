from bw2data.backends.proxies import Exchange
from bw2data.backends.schema import ExchangeDataset
from bw2data.errors import MultipleResults, UnknownObject
from bw_temporalis import TemporalDistribution
from loguru import logger

def get_exchange(**kwargs) -> Exchange:
    """
    Get an exchange from the database.

    Parameters
    ----------
    **kwargs :
        Arguments to specify an exchange.
            - input_node: Input node object
            - input_code: Input node code
            - input_database: Input node database
            - output_node: Output node object
            - output_code: Output node code
            - output_database: Output node database

    Returns
    -------
    Exchange
        The exchange object matching the criteria.

    Raises
    ------
    MultipleResults
        If multiple exchanges match the criteria.
    UnknownObject
        If no exchange matches the criteria.
    """

    # Process input_node if present
    input_node = kwargs.pop("input_node", None)
    if input_node:
        kwargs["input_code"] = input_node["code"]
        kwargs["input_database"] = input_node["database"]

    # Process output_node if present
    output_node = kwargs.pop("output_node", None)
    if output_node:
        kwargs["output_code"] = output_node["code"]
        kwargs["output_database"] = output_node["database"]

    # Map kwargs to database fields
    mapping = {
        "input_code": ExchangeDataset.input_code,
        "input_database": ExchangeDataset.input_database,
        "output_code": ExchangeDataset.output_code,
        "output_database": ExchangeDataset.output_database,
    }

    # Build query filters
    filters = []
    for key, value in kwargs.items():
        field = mapping.get(key)
        if field is not None:
            filters.append(field == value)

    # Execute query with filters
    qs = ExchangeDataset.select().where(*filters)
    candidates = [Exchange(obj) for obj in qs]
    num_candidates = len(candidates)

    if num_candidates > 1:
        raise MultipleResults(
            f"Found {num_candidates} results for the given search. "
            "Please be more specific or double-check your system model for duplicates."
        )
    if num_candidates == 0:
        raise UnknownObject("No exchange found matching the criteria.")

    return candidates[0]

def add_temporal_distribution_to_exchange(
    temporal_distribution: TemporalDistribution, **kwargs
):
    """
    Adds a temporal distribution to an exchange specified by kwargs.

    Parameters
    ----------
    temporal_distribution : TemporalDistribution
        TemporalDistribution to be added to the exchange.
    **kwargs :
        Arguments to specify an exchange.
            - input_node: Input node object
            - input_id: Input node database ID
            - input_code: Input node code
            - input_database: Input node database
            - output_node: Output node object
            - output_id: Output node database ID
            - output_code: Output node code
            - output_database: Output node database

    Returns
    -------
    None
        The exchange is saved with the temporal distribution.
    """
    exchange = get_exchange(**kwargs)
    exchange["temporal_distribution"] = temporal_distribution
    exchange.save()
    logger.info(f"Added temporal distribution to exchange {exchange}.")