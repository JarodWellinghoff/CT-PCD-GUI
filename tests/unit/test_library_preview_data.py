"""Source-HU and geometry checks; no Qt or OpenGL is required."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from ct_pcd_gui.features.lesion_insertion.application.library_preview import prepare_library_preview
from ct_pcd_gui.features.lesion_viewer.domain.inspection import to_zyx


def example(channels=2, shape=(3, 4, 5)):
    values = (
        np.arange(np.prod(shape) * channels, dtype=np.float64).reshape(*shape, channels)
        - 1000.123456789
    )
    mask = np.zeros_like(values, dtype=bool)
    mask[1:3, 1:3, 1:4, :] = True
    model = SimpleNamespace(voi_hu=values, mask=mask, row_spacing_mm=0.8,
                            column_spacing_mm=0.6, slice_spacing_mm=1.5)
    item = SimpleNamespace(path='/research/lesion.npz', lesion_id='Lesion01',
                           display_name='Lesion', metadata={
                               'old_background_hu': [-900] * channels,
                               'contrast_hu_by_channel': [50] * channels,
                               'kvp': 120, 'old_background_method': 'inverse segmentation',
                               'old_background_voxel_count': 48,
                           })
    return model, item


@pytest.mark.parametrize('channels', [1, 2, 3])
def test_values_are_lossless_read_only_views(channels):
    model, item = example(channels)
    original = model.voi_hu.copy()
    data = prepare_library_preview(model, item)
    assert len(data.channels) == channels
    for i, channel in enumerate(data.channels):
        assert channel.units == 'HU'
        assert channel.name == f'Source threshold {i + 1}'
        assert np.shares_memory(channel.values, model.voi_hu)
        assert channel.values.dtype == np.float64
        np.testing.assert_array_equal(channel.values, original[..., i])
        with pytest.raises(ValueError):
            channel.values[0, 0, 0] = 0
    assert model.voi_hu.flags.writeable
    np.testing.assert_array_equal(model.voi_hu, original)


def test_axis_order_spacing_and_per_channel_mask_are_not_swapped():
    model, item = example()
    model.mask[..., 1] = False
    model.mask[0, 0, 0, 1] = True
    data = prepare_library_preview(model, item)
    assert data.spacing_xyz == (0.6, 0.8, 1.5)
    source = data.channels[1]
    image = to_zyx(source.values, 'YXZ')
    assert image.shape == (5, 3, 4)
    assert image[4, 1, 2] == model.voi_hu[1, 2, 4, 1]
    assert np.count_nonzero(source.mask) == 1
    request = data.surface_request(1)
    assert request.axis_order == 'YXZ'
    assert request.spacing_xyz == data.spacing_xyz
    assert request.volume is source.mask
    assert request.threshold == 0.5 and request.pad_border
    assert not request.largest_component_only
    assert request.downsample == 1 and request.smoothing_iterations == 0


def test_summary_reports_exact_mask_measurements_and_background():
    model, item = example()
    data = prepare_library_preview(model, item)
    stats = json.loads(data.metadata)['computed_mask_statistics'][0]
    expected = model.voi_hu[..., 0][model.mask[..., 0]]
    assert stats['segmented_voxels'] == 12
    assert stats['mask_volume_mm3'] == pytest.approx(12 * 0.8 * 0.6 * 1.5)
    assert stats['mean_hu'] == pytest.approx(expected.mean())
    assert stats['std_hu'] == pytest.approx(expected.std())
    assert 'Stored background: -900 HU' in data.summary
    assert 'Stored lesion/background contrast: 50 HU' in data.summary
    assert 'VOI extent' in data.summary
    assert '120' in data.summary


def test_nonfinite_values_are_preserved_but_excluded_from_hu_statistics():
    model, item = example()
    model.voi_hu[1, 1, 1, 0] = np.nan
    model.voi_hu[1, 1, 2, 0] = np.inf
    data = prepare_library_preview(model, item)
    stats = json.loads(data.metadata)['computed_mask_statistics'][0]
    assert stats['segmented_voxels'] == 12
    assert stats['nonfinite_segmented_voxels'] == 2
    assert np.isnan(data.channels[0].values[1, 1, 1])
    assert np.isfinite(stats['mean_hu'])


@pytest.mark.parametrize('mask_kind', ['3d', 'singleton_channel'])
def test_shared_mask_is_applied_to_each_threshold(mask_kind):
    model, item = example()
    model.mask = model.mask[..., 0] if mask_kind == '3d' else model.mask[..., :1]
    data = prepare_library_preview(model, item)
    np.testing.assert_array_equal(data.channels[0].mask, data.channels[1].mask)


def test_empty_mask_keeps_hu_but_disables_surface():
    model, item = example()
    model.mask[:] = False
    data = prepare_library_preview(model, item)
    assert len(data.channels) == 2
    with pytest.raises(ValueError, match='no segmented voxels'):
        data.surface_request(0)
    assert 'no finite segmented voxels' in data.summary


def test_single_slice_still_has_a_2d_preview():
    model, item = example(1, (3, 4, 1))
    model.mask[:] = True
    data = prepare_library_preview(model, item)
    assert data.channels[0].values.shape == (3, 4, 1)
    with pytest.raises(ValueError, match='at least two voxels'):
        data.surface_request(0)


@pytest.mark.parametrize('spacing', [0, -1, np.nan, np.inf])
def test_invalid_geometry_is_rejected(spacing):
    model, item = example()
    model.row_spacing_mm = spacing
    with pytest.raises(ValueError, match='positive finite'):
        prepare_library_preview(model, item)


@pytest.mark.parametrize('bad_shape', [(2, 4, 5, 2), (3, 4, 5, 3)])
def test_mismatched_masks_are_rejected(bad_shape):
    model, item = example()
    model.mask = np.ones(bad_shape, dtype=bool)
    with pytest.raises(ValueError, match='mask'):
        prepare_library_preview(model, item)


def test_cancellation_checked_between_thresholds():
    model, item = example()
    calls = 0
    def cancel():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        prepare_library_preview(model, item, check_cancelled=cancel)


@pytest.mark.parametrize('index', [-1, 2])
def test_out_of_range_surface_channel(index):
    model, item = example()
    with pytest.raises(ValueError, match='available source threshold'):
        prepare_library_preview(model, item).surface_request(index)
