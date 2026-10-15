"""Regression tests for target filtering, using real pipeline code and synthetic geometry."""
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

pytest.importorskip("pydicom")
from ct_pcd_gui.features.lesion_insertion._vendor.lesion_pipeline import pipeline


class _PipelineProxy:
    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch

    def update(self, **values):
        for key, value in values.items():
            self.monkeypatch.setattr(pipeline, key, value)

    def __getitem__(self, key):
        return getattr(pipeline, key)


def insertion(mapping, count=2, label='test'):
    return NS(spectrum_channel_map=mapping, model=NS(channel_count=count), background_hu=(0,) * count, label=label)

def execute_synthetic_frames(tmp_path, maps, monkeypatch):
    p = _PipelineProxy(monkeypatch)
    frames = np.array([[[100, 200]], [[300, 400]]], dtype='<u2')
    header = b'synthetic-header'
    trailer = b'unchanged-trailer'
    original = header + frames.tobytes() + trailer
    source = tmp_path / 'source.bin'
    source.write_bytes(original)
    output = tmp_path / 'out'
    geometry = [NS(rows=1, columns=2, stored_rows=1, stored_columns=2, pixel_data_transposed=False, spectrum_index=s) for s in (1, 2)]
    decoded = []
    projected = []

    def decode(ds, stored, frame, geom):
        decoded.append(frame)
        return np.array(stored[frame], dtype=float)

    def project(src, detector, volume):
        projected.append(volume)
        return np.full((1, 2), 10 * (volume + 1), dtype=float)
    p.update(_WORKER_OUTPUT_ROOT=output, _WORKER_OVERWRITE=False, _WORKER_INSERTIONS=[insertion(m) for m in maps], _WORKER_SPECTRUM_MAP={}, pydicom=NS(dcmread=lambda *a, **kw: NS()), _pixel_data_span=lambda ds: (len(header), frames.nbytes), _metadata_signature=lambda ds: 'fixed-header', read_geometry=lambda ds, frame: geometry[frame], projection_frame_count=lambda ds: 2, stored_projection_shape=lambda ds, geom: frames.shape, projection_pixel_dtype=lambda ds: frames.dtype, source_and_detector_positions=lambda geom: (None, None), _volume_for=lambda ins, channel: channel, project_volume=project, decode_projection_frame=decode, encode_projection_frame=lambda ds, value, frame, geom: (value.astype(frames.dtype), 0))
    record = NS(path=source, relative_path=Path('frames.bin'), spectrum_index=1, spectrum_indices=(1, 2), source_indices=(1,), instance_number=1)
    result = p['_process_record'](record)
    data = (output / record.relative_path).read_bytes()
    assert source.read_bytes() == original
    assert data[:len(header)] == header
    assert data[-len(trailer):] == trailer
    pixels = np.frombuffer(data[len(header):len(header) + frames.nbytes], dtype=frames.dtype).reshape(frames.shape)
    return (pixels, frames, decoded, projected, result, data, original)

def test_target_one_only_preserves_second_frame_stored_bytes(tmp_path, monkeypatch):
    pixels, frames, decoded, projected, result, data, original = execute_synthetic_frames(tmp_path, [{1: 1}], monkeypatch)
    np.testing.assert_array_equal(pixels[0], frames[0] + 20)
    np.testing.assert_array_equal(pixels[1], frames[1])
    assert decoded == [0] and projected == [1]
    assert result.changed_frames == 1
    assert data[-len(b'unchanged-trailer') - 4:] == original[-len(b'unchanged-trailer') - 4:]

def test_target_two_only_preserves_first_frame(tmp_path, monkeypatch):
    pixels, frames, decoded, projected, result, *_ = execute_synthetic_frames(tmp_path, [{2: 0}], monkeypatch)
    np.testing.assert_array_equal(pixels[0], frames[0])
    np.testing.assert_array_equal(pixels[1], frames[1] + 10)
    assert decoded == [1] and projected == [0]

def test_both_targets_receive_same_selected_source(tmp_path, monkeypatch):
    pixels, frames, decoded, projected, result, *_ = execute_synthetic_frames(tmp_path, [{1: 1, 2: 1}], monkeypatch)
    np.testing.assert_array_equal(pixels, frames + 20)
    assert projected == [1, 1] and result.changed_frames == 2

def test_multiple_lesions_have_independent_routing(tmp_path, monkeypatch):
    pixels, frames, decoded, projected, result, *_ = execute_synthetic_frames(tmp_path, [{1: 0}, {2: 1}], monkeypatch)
    np.testing.assert_array_equal(pixels[0], frames[0] + 10)
    np.testing.assert_array_equal(pixels[1], frames[1] + 20)
    assert projected == [0, 1]

def test_all_excluded_frames_are_never_decoded_or_reencoded(tmp_path, monkeypatch):
    pixels, frames, decoded, projected, result, data, original = execute_synthetic_frames(tmp_path, [], monkeypatch)
    assert data == original
    assert decoded == projected == []
    assert result.changed_frames == result.changed_pixels == 0
