from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from optimex import lca_processor


def _minimal_config(
    foreground_db_name: str = "foreground",
) -> lca_processor.LCAConfig:
    temporal = lca_processor.TemporalConfig(
        start_date=datetime(2025, 1, 1),
        database_dates={},
    )
    return lca_processor.LCAConfig.model_construct(
        demand={},
        temporal=temporal,
        characterization_methods=[],
        background_inventory=lca_processor.BackgroundInventoryConfig(),
        foreground_db_name=foreground_db_name,
    )


def _patch_processor_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        lca_processor.LCADataProcessor,
        "_parse_demand",
        lambda _: None,
    )
    monkeypatch.setattr(
        lca_processor.LCADataProcessor, "_construct_foreground_tensors", lambda _: None
    )
    monkeypatch.setattr(
        lca_processor.LCADataProcessor, "_prepare_background_inventory", lambda _: None
    )
    monkeypatch.setattr(
        lca_processor.LCADataProcessor,
        "_construct_characterization_tensor",
        lambda _: None,
    )
    monkeypatch.setattr(
        lca_processor.LCADataProcessor,
        "_construct_mapping_matrix",
        lambda _: None,
    )


def _patch_brightway(monkeypatch: pytest.MonkeyPatch, *databases: str) -> MagicMock:
    """Stand in for the Brightway registry with the given database names."""
    monkeypatch.setattr(lca_processor.bd, "databases", {name: {} for name in databases})
    db_factory = MagicMock(
        side_effect=lambda name: SimpleNamespace(name=name, metadata={})
    )
    monkeypatch.setattr(lca_processor.bd, "Database", db_factory)
    monkeypatch.setattr(
        lca_processor.bd, "config", SimpleNamespace(biosphere="biosphere3")
    )
    return db_factory


def test_custom_foreground_db_name_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_processor_methods(monkeypatch)
    config = _minimal_config()

    db_factory = _patch_brightway(monkeypatch, "custom_foreground")

    processor = lca_processor.LCADataProcessor(
        config, foreground_db_name="custom_foreground"
    )

    assert processor.foreground_db.name == "custom_foreground"
    assert any(
        call.args == ("custom_foreground",) for call in db_factory.call_args_list
    )


def test_missing_custom_foreground_db_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_processor_methods(monkeypatch)
    config = _minimal_config()

    monkeypatch.setattr(lca_processor.bd, "databases", {"foreground": {}})

    with pytest.raises(
        ValueError, match="Foreground database 'missing_db' is not defined."
    ):
        lca_processor.LCADataProcessor(config, foreground_db_name="missing_db")


def test_config_foreground_db_name_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """The name on the config must be honoured when no explicit name is passed.

    Without this, a config pointing at a non-default foreground database is
    silently ignored and the processor reads whatever "foreground" happens to be.
    """
    _patch_processor_methods(monkeypatch)
    config = _minimal_config(foreground_db_name="custom_foreground")

    _patch_brightway(monkeypatch, "custom_foreground")

    processor = lca_processor.LCADataProcessor(config)

    assert processor.foreground_db.name == "custom_foreground"


def test_explicit_foreground_db_name_overrides_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicitly passed name still wins over the one on the config."""
    _patch_processor_methods(monkeypatch)
    config = _minimal_config(foreground_db_name="config_foreground")

    _patch_brightway(monkeypatch, "config_foreground", "explicit_foreground")

    processor = lca_processor.LCADataProcessor(
        config, foreground_db_name="explicit_foreground"
    )

    assert processor.foreground_db.name == "explicit_foreground"


def test_missing_config_foreground_db_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config naming an absent database must fail loudly, not fall back."""
    _patch_processor_methods(monkeypatch)
    config = _minimal_config(foreground_db_name="missing_db")

    monkeypatch.setattr(lca_processor.bd, "databases", {"foreground": {}})

    with pytest.raises(
        ValueError, match="Foreground database 'missing_db' is not defined."
    ):
        lca_processor.LCADataProcessor(config)
