import pytest

from ct_pcd_gui.features.lesion_insertion.domain import thresholds as mod


@pytest.mark.parametrize("source,targets,expected", [
    (0, (1, 2), {1: 0, 2: 0}),
    (1, (1, 2), {1: 1, 2: 1}),
    (0, (2,), {2: 0}),
    (1, (1,), {1: 1}),
    (0, (7,), {7: 0}),
])
def test_one_source_to_requested_targets(source, targets, expected):
    assert mod.threshold_mapping(source, targets, 2, (1, 2, 7)) == expected


@pytest.mark.parametrize("source,targets,count,available", [
    (None, (1, 2), 2, (1, 2)),
    (-1, (1,), 2, (1, 2)),
    (2, (1,), 2, (1, 2)),
    (0, (), 1, (1, 2)),
    (1, (1,), 1, (1, 2)),
    (0, (3,), 2, (1, 2)),
    (0, (1, 1), 2, (1, 2)),
    (True, (1,), 2, (1,)),
    (0, (True,), 2, (1,)),
    (0, (1.0,), 2, (1,)),
    (0, (0,), 2, (1,)),
    (0.0, (1,), 2, (1,)),
])
def test_invalid_or_incomplete_selection_is_rejected(source, targets, count, available):
    with pytest.raises(ValueError):
        mod.threshold_mapping(source, targets, count, available)


def test_uniform_old_session_is_migrated_without_changing_routing():
    migrated = mod.migrate_threshold_fields({"background_hu": [10, 20]}, {1: 1, 2: 1})
    assert migrated["source_channel"] == 1
    assert migrated["target_spectrum_indices"] == (1, 2)


def test_mixed_source_old_session_requires_explicit_source_choice():
    migrated = mod.migrate_threshold_fields({"background_hu": [10, 20]}, {1: 0, 2: 1})
    assert migrated["source_channel"] is None
    assert migrated["target_spectrum_indices"] == (1, 2)


def test_new_session_selection_overrides_legacy_global_mapping():
    migrated = mod.migrate_threshold_fields(
        {"source_channel": 0, "target_spectrum_indices": [2], "background_hu": [1, 2]},
        {1: 0, 2: 1},
    )
    assert migrated["source_channel"] == 0
    assert migrated["target_spectrum_indices"] == (2,)


def test_single_source_file_without_old_mapping_uses_only_source():
    assert mod.migrate_threshold_fields({"background_hu": [10]}, {})["source_channel"] == 0


def test_no_destination_is_invented_during_migration():
    assert mod.migrate_threshold_fields({"background_hu": [10]}, {})["target_spectrum_indices"] == ()


def test_summary_explains_same_source_for_both_targets():
    text = mod.threshold_summary(1, (1, 2), (1, 2))
    assert "Lesion threshold 2" in text
    assert "same lesion source threshold" in text


def test_summary_identifies_excluded_and_unknown_targets():
    assert "not inserted into target threshold(s) 1" in mod.threshold_summary(0, (2,), (1, 2))
    assert "WARNING" in mod.threshold_summary(0, (3,), (1, 2))
