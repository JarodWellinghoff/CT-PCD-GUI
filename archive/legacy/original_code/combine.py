"""Memory-bounded single-frame to multi-frame DICOM-CT-PD converter.

The global pass stores only compact scalar metadata and path indices. Full
headers and decoded frames are loaded only for the output chunk being written.
"""

from __future__ import annotations
import io
import copy
import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydicom import Dataset, dcmread, dcmwrite
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.tag import Tag
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from pydicom.filebase import DicomIO
from pydicom.uid import RLELossless
from tqdm import tqdm

try:
    # pydicom >= 3.0 can decode directly from a path while reading only the
    # minimum pixel-related content required.
    from pydicom.pixels import pixel_array as _decode_pixel_array_from_path
except ImportError:  # Compatibility with older pydicom releases.
    _decode_pixel_array_from_path = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Conservative output ceiling retained from the original 4 GiB design.
# 0xFFFFFFFF is reserved for undefined element length, and DICOM value lengths
# must be even, so this uses the largest defined even 32-bit value.
OUTPUT_FILE_LIMIT_BYTES = 0xFFFFFFFF
FILE_LIMIT = OUTPUT_FILE_LIMIT_BYTES  # Backward-compatible constant name.

# Deliberately smaller than the DICOM maximum. During serialization, both the
# NumPy pixel array and the immutable PixelData bytes object coexist briefly.
DEFAULT_MAX_PIXEL_BYTES = 512 * 1024**2  # 512 MB
DEFAULT_METADATA_RESERVE_BYTES = 64 * 1024**2  # 64 MB
DEFAULT_MAX_FRAMES_PER_CHUNK = 10_000

DEFAULT_PARENT_DIR = (
    r"U:\PUBLIC\DICOM-CTPD\PCD_DICOMCTPD_Generator" r"\Full_dose_CTPD_Single_Frame"
)

_TUBE_PATTERN = re.compile(r"\bTube(\d+)", re.IGNORECASE)
_THRESHOLD_PATTERN = re.compile(r"\bThr(\d+)", re.IGNORECASE)
_GENERATED_FILE_SUFFIX = "_multiframe_ct_pd.dcm"
_PIXEL_DATA_TAG = Tag(0x7FE0, 0x0010)

# Only these tags are read during the global indexing pass. Full headers are
# read later, one output chunk at a time.
_INDEX_TAGS = [
    "SeriesDescription",
    "InstanceNumber",
    "Rows",
    "Columns",
    "SamplesPerPixel",
    "BitsAllocated",
    "NumberOfFrames",
]

_INDEX_DTYPE = np.dtype(
    [
        ("tube", np.int32),
        ("threshold", np.int32),
        ("instance", np.int64),
        ("pixel_bytes", np.uint64),
        ("path_index", np.int64),
    ]
)

GroupedIndex = dict[int, dict[int, np.ndarray]]

VARYING_PRIVATE_TAGS = (
    Tag(0x7031, 0x1001),  # DetectorFocalCenterAngularPosition
    Tag(0x7031, 0x1002),  # DetectorFocalCenterAxialPosition
    Tag(0x7031, 0x1003),  # DetectorFocalCenterRadialDistance
    Tag(0x7033, 0x100B),  # SourceAngularPositionShift
    Tag(0x7033, 0x1065),  # PhotonStatistics
    Tag(0x7033, 0x1067),  # Timestamp
)

VARYING_PUBLIC_TAGS = (
    Tag(0x0020, 0x0013),  # InstanceNumber
    Tag(0x0028, 0x0106),  # SmallestImagePixelValue
    Tag(0x0028, 0x0107),  # LargestImagePixelValue
)


_FLOAT32_PRIVATE_TAGS = frozenset(
    {
        Tag(0x7031, 0x1001),
        Tag(0x7031, 0x1002),
        Tag(0x7031, 0x1003),
        Tag(0x7033, 0x1065),
        Tag(0x7033, 0x1067),
    }
)


class TqdmWriter:
    def __init__(
        self,
        fileobj,
        progress,
        chunk_size: int = 16 * 1024 * 1024,  # 16 MiB
    ):
        self.fileobj = fileobj
        self.progress = progress
        self.chunk_size = chunk_size

    def write(self, data):
        view = memoryview(data)
        total_written = 0

        while total_written < len(view):
            chunk = view[total_written : total_written + self.chunk_size]

            n = self.fileobj.write(chunk)

            if n is None:
                # Some file-like objects may return None.
                n = len(chunk)

            if n == 0:
                raise OSError("write() returned 0 bytes")

            total_written += n
            self.progress.update(n)

        return total_written

    def seek(self, offset, whence=0):
        return self.fileobj.seek(offset, whence)

    def tell(self):
        return self.fileobj.tell()


@dataclass(frozen=True, slots=True)
class TubeSplitPlan:
    """Aligned chunk boundaries shared by every threshold in one tube."""

    starts: np.ndarray
    stops: np.ndarray
    sizes: np.ndarray
    first_instances: np.ndarray
    last_instances: np.ndarray
    bytes_by_threshold: dict[int, np.ndarray]

    @property
    def number_of_groups(self) -> int:
        return int(self.sizes.size)


@dataclass(frozen=True, slots=True)
class IndexedFiles:
    """Compact global index; no complete pydicom Dataset objects are retained."""

    paths: tuple[str, ...]
    records: np.ndarray
    grouped: GroupedIndex


# -----------------------------------------------------------------------------
# File discovery and compact indexing
# -----------------------------------------------------------------------------


def discover_dicom_files(
    parent_dir: str | os.PathLike[str],
    *,
    extensions: Iterable[str] = (".dcm", ".ima"),
    exclude_generated: bool = True,
) -> list[str]:
    """Return DICOM paths beneath *parent_dir* without reading the files."""

    root_path = os.fspath(parent_dir)
    normalized_extensions = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    }

    paths: list[str] = []

    for root, _, filenames in os.walk(root_path):
        for filename in filenames:
            lower_name = filename.lower()

            if exclude_generated and lower_name.endswith(_GENERATED_FILE_SUFFIX):
                continue

            if Path(lower_name).suffix not in normalized_extensions:
                continue

            paths.append(os.path.join(root, filename))

    paths.sort()
    return paths


def pixel_data_nbytes(ds: Dataset) -> float:
    """Estimate the uncompressed Pixel Data value length for one input file."""

    rows = int(ds.Rows)
    columns = int(ds.Columns)
    samples_per_pixel = int(ds.get("SamplesPerPixel", 1))
    number_of_frames = int(ds.get("NumberOfFrames", 1))
    bits_allocated = int(ds.BitsAllocated)

    if rows <= 0 or columns <= 0:
        raise ValueError("Rows and Columns must be positive")
    if samples_per_pixel <= 0:
        raise ValueError("SamplesPerPixel must be positive")
    if bits_allocated <= 0:
        raise ValueError("BitsAllocated must be positive")
    if number_of_frames <= 0:
        raise ValueError("NumberOfFrames must be positive")

    total_bits = rows * columns * samples_per_pixel * number_of_frames * bits_allocated
    return total_bits / 8


def _parse_series_identifiers(ds: Dataset, path: str) -> tuple[int, int, int]:
    series_description = str(ds.get("SeriesDescription", ""))

    tube_match = _TUBE_PATTERN.search(series_description)
    if tube_match is None:
        raise ValueError(
            f"No tube number was found in SeriesDescription={series_description!r} "
            f"for {path!r}"
        )

    threshold_match = _THRESHOLD_PATTERN.search(series_description)
    if threshold_match is None:
        raise ValueError(
            "No threshold number was found in "
            f"SeriesDescription={series_description!r} for {path!r}"
        )

    instance_value = ds.get("InstanceNumber")
    if instance_value is None:
        raise ValueError(f"InstanceNumber is missing from {path!r}")

    try:
        instance = int(instance_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid InstanceNumber={instance_value!r} in {path!r}"
        ) from exc

    return int(tube_match.group(1)), int(threshold_match.group(1)), instance


def parse_header(path_index, path):
    try:
        ds = dcmread(
            path,
            force=True,
            stop_before_pixels=True,
            specific_tags=_INDEX_TAGS,
        )
        tube, threshold, instance = _parse_series_identifiers(ds, path)
        pixel_bytes = pixel_data_nbytes(ds)
        return (
            tube,
            threshold,
            instance,
            pixel_bytes,
            path_index,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to index {path!r}") from exc


def build_file_index(paths: Sequence[str | os.PathLike[str]]) -> IndexedFiles:
    """
    Build a compact, sorted index without retaining complete DICOM headers.

    The returned records are sorted by tube, threshold, and InstanceNumber.
    Each record stores only scalar metadata and an integer into ``paths``.
    """

    normalized_paths = tuple(os.fspath(path) for path in paths)
    records = np.empty(len(normalized_paths), dtype=_INDEX_DTYPE)

    with tqdm(
        total=len(normalized_paths), desc="Indexing DICOM headers", unit="file"
    ) as pbar:
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(parse_header, path_index, path): path
                for path_index, path in enumerate(normalized_paths)
            }

            for future in as_completed(futures):
                path = futures[future]
                try:
                    data = future.result()
                    records[data[-1]] = data
                except Exception as e:
                    print(f"{path} generated an exception: {e}")
                finally:
                    pbar.update(1)

    if records.size == 0:
        return IndexedFiles(normalized_paths, records, {})

    order = np.lexsort(
        (
            records["instance"],
            records["threshold"],
            records["tube"],
        )
    )
    records = records[order]
    grouped = group_index_records(records)

    return IndexedFiles(normalized_paths, records, grouped)


def group_index_records(records: np.ndarray) -> GroupedIndex:
    """Create tube/threshold views into a sorted compact index array."""

    grouped: GroupedIndex = {}

    if records.size == 0:
        return grouped

    new_group = np.empty(records.size, dtype=bool)
    new_group[0] = True
    new_group[1:] = (records["tube"][1:] != records["tube"][:-1]) | (
        records["threshold"][1:] != records["threshold"][:-1]
    )

    starts = np.flatnonzero(new_group)
    stops = np.r_[starts[1:], records.size]

    for start, stop in zip(starts, stops):
        tube = int(records["tube"][start])
        threshold = int(records["threshold"][start])
        grouped.setdefault(tube, {})[threshold] = records[start:stop]

    return grouped


# -----------------------------------------------------------------------------
# Aligned chunk planning
# -----------------------------------------------------------------------------


def pixel_byte_prefix_sum(group: np.ndarray) -> np.ndarray:
    prefix = np.empty(group.size + 1, dtype=np.uint64)
    prefix[0] = 0
    np.cumsum(group["pixel_bytes"], dtype=np.uint64, out=prefix[1:])
    return prefix


def balanced_boundaries(
    number_of_instances: int,
    number_of_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return balanced [start, stop) boundaries."""

    if number_of_instances <= 0:
        raise ValueError("number_of_instances must be positive")
    if not 1 <= number_of_groups <= number_of_instances:
        raise ValueError("number_of_groups must be between 1 and number_of_instances")

    base_size, remainder = divmod(number_of_instances, number_of_groups)
    sizes = np.full(number_of_groups, base_size, dtype=np.intp)
    sizes[:remainder] += 1

    stops = np.cumsum(sizes, dtype=np.intp)
    starts = np.empty_like(stops)
    starts[0] = 0
    starts[1:] = stops[:-1]

    return starts, stops, sizes


def make_tube_split_plan(
    tube: int,
    threshold_groups: dict[int, np.ndarray],
    *,
    usable_pixel_limit: int,
    max_frames_per_chunk: int | None = DEFAULT_MAX_FRAMES_PER_CHUNK,
) -> TubeSplitPlan:
    """
    Create common, balanced chunk boundaries for all thresholds in one tube.
    """

    if not threshold_groups:
        raise ValueError(f"Tube {tube} has no threshold groups")
    if usable_pixel_limit <= 0:
        raise ValueError("usable_pixel_limit must be positive")
    if max_frames_per_chunk is not None and max_frames_per_chunk <= 0:
        raise ValueError("max_frames_per_chunk must be positive or None")

    threshold_items = sorted(threshold_groups.items())
    reference_threshold, reference_group = threshold_items[0]
    reference_instances = reference_group["instance"]
    number_of_instances = int(reference_group.size)

    if number_of_instances == 0:
        raise ValueError(f"Tube {tube}, threshold {reference_threshold} is empty")

    if number_of_instances > 1 and np.any(
        reference_instances[1:] <= reference_instances[:-1]
    ):
        raise ValueError(
            f"Tube {tube}, threshold {reference_threshold} has duplicate or "
            "non-increasing InstanceNumber values"
        )

    for threshold, group in threshold_items[1:]:
        if group.size != number_of_instances:
            raise ValueError(
                f"Tube {tube} thresholds are not aligned: threshold "
                f"{reference_threshold} has {number_of_instances:,} instances, "
                f"while threshold {threshold} has {group.size:,}"
            )

        if not np.array_equal(group["instance"], reference_instances):
            differing_positions = np.flatnonzero(
                group["instance"] != reference_instances
            )
            position = int(differing_positions[0])
            raise ValueError(
                f"Tube {tube}, threshold {threshold} is not aligned with "
                f"threshold {reference_threshold}. At sorted position "
                f"{position}, the InstanceNumber values are "
                f"{int(group['instance'][position])} and "
                f"{int(reference_instances[position])}."
            )

    prefix_by_threshold: dict[int, np.ndarray] = {}
    minimum_number_of_groups = 1

    if max_frames_per_chunk is not None:
        minimum_number_of_groups = max(
            minimum_number_of_groups,
            (number_of_instances + max_frames_per_chunk - 1) // max_frames_per_chunk,
        )

    for threshold, group in threshold_items:
        oversized = np.flatnonzero(group["pixel_bytes"] > usable_pixel_limit)
        if oversized.size:
            position = int(oversized[0])
            raise ValueError(
                f"Tube {tube}, threshold {threshold}, instance "
                f"{int(group['instance'][position])} requires "
                f"{int(group['pixel_bytes'][position]):,} pixel bytes, which "
                f"exceeds the configured limit of {usable_pixel_limit:,} bytes"
            )

        prefix = pixel_byte_prefix_sum(group)
        prefix_by_threshold[threshold] = prefix
        total_bytes = int(prefix[-1])
        required_groups = (total_bytes + usable_pixel_limit - 1) // usable_pixel_limit
        minimum_number_of_groups = max(
            minimum_number_of_groups,
            required_groups,
        )

    for number_of_groups in range(
        minimum_number_of_groups,
        number_of_instances + 1,
    ):
        starts, stops, sizes = balanced_boundaries(
            number_of_instances,
            number_of_groups,
        )

        bytes_by_threshold: dict[int, np.ndarray] = {}
        fits_all_thresholds = True

        for threshold, prefix in prefix_by_threshold.items():
            chunk_bytes = prefix[stops] - prefix[starts]
            bytes_by_threshold[threshold] = chunk_bytes

            if np.any(chunk_bytes > usable_pixel_limit):
                fits_all_thresholds = False
                break

        if fits_all_thresholds:
            return TubeSplitPlan(
                starts=starts,
                stops=stops,
                sizes=sizes,
                first_instances=reference_instances[starts].copy(),
                last_instances=reference_instances[stops - 1].copy(),
                bytes_by_threshold=bytes_by_threshold,
            )

    raise RuntimeError(f"Could not construct a valid split plan for tube {tube}")


def calculate_usable_pixel_limit(
    max_pixel_bytes: int,
    metadata_reserve_bytes: int,
) -> int:
    """Apply the memory cap and reserve room below the output ceiling."""

    if max_pixel_bytes <= 0:
        raise ValueError("max_pixel_bytes must be positive")
    if metadata_reserve_bytes < 0:
        raise ValueError("metadata_reserve_bytes must be nonnegative")
    if metadata_reserve_bytes >= OUTPUT_FILE_LIMIT_BYTES:
        raise ValueError("metadata_reserve_bytes is too large")

    dicom_limited_payload = OUTPUT_FILE_LIMIT_BYTES - metadata_reserve_bytes
    usable = min(max_pixel_bytes, dicom_limited_payload)

    if usable <= 0:
        raise ValueError("No usable Pixel Data space remains after reservation")

    return usable


# -----------------------------------------------------------------------------
# Per-chunk loading and DICOM construction
# -----------------------------------------------------------------------------


def _source_path(source: str | os.PathLike[str] | Dataset) -> str:
    if isinstance(source, (str, os.PathLike)):
        return os.fspath(source)

    filename = getattr(source, "filename", None)
    if filename:
        return os.fspath(filename)

    raise TypeError(
        "Each DICOM source must be a path or a Dataset with a valid filename"
    )


def _strip_pixel_data(ds: FileDataset) -> None:
    """
    Remove raw and cached pixel data from a Dataset after decoding.

    This keeps returned headers lightweight and prevents the Dataset from
    retaining another reference to the decoded NumPy array.
    """

    # Raw pixel payload
    for keyword in (
        "PixelData",
        "FloatPixelData",
        "DoubleFloatPixelData",
    ):
        if keyword in ds:
            del ds[keyword]

    # pydicom Dataset.pixel_array caches the decoded ndarray internally.
    # Remove the cache because the pixels are being stored separately.
    if hasattr(ds, "_pixel_array"):
        ds._pixel_array = None

    if hasattr(ds, "_pixel_id"):
        ds._pixel_id = {}


def _read_dicom(
    path: str,
) -> tuple[FileDataset, NDArray]:
    """
    Read one DICOM file once and return its header and decoded pixels.
    """

    try:
        ds = dcmread(
            path,
            force=True,
        )

        frame = ds.pixel_array

    except Exception as exc:
        raise RuntimeError(f"Failed to read/decode DICOM file {path!r}") from exc

    # The decoded frame remains alive through `frame`, so we can safely
    # remove both the raw Pixel Data and Dataset's cached ndarray.
    _strip_pixel_data(ds)

    return ds, frame


def load_headers_and_pixel_data(
    paths: Sequence[str],
    *,
    max_workers: int | None = 4,
) -> tuple[list[FileDataset], bytes]:
    """
    Read DICOM headers and pixel data concurrently with bounded memory usage.

    Each DICOM file is read only once.

    Returns
    -------
    headers
        DICOM datasets with Pixel Data removed.

    pixel_data
        Preallocated NumPy array with shape
        (num_files, *frame_shape).

    Peak image memory is approximately:

        destination volume
        + max_workers * (
            raw Pixel Data
            + decoded frame
            + decoder working memory
        )

    rather than storing all decoded frames before stacking.
    """

    if not paths:
        raise ValueError("Cannot load DICOM data from an empty path sequence")

    if max_workers is None:
        cc = os.cpu_count()
        cc = cc if cc is not None else 1
        max_workers = 5 * cc
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1 or None")

    # ------------------------------------------------------------
    # First file synchronously
    #
    # We need its metadata, frame shape, and dtype before allocating
    # the final destination array.
    # ------------------------------------------------------------

    first_header, first_frame = _read_dicom(paths[0])

    expected_shape = first_frame.shape
    target_dtype = _pixel_dtype_from_header(first_header)

    pixel_data = np.empty(
        (len(paths), *expected_shape),
        dtype=target_dtype,
        order="C",
    )

    headers: list[FileDataset | None] = [None] * len(paths)
    headers[0] = first_header

    try:
        np.copyto(
            pixel_data[0],
            first_frame,
            casting="unsafe",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not copy first frame with shape "
            f"{first_frame.shape} and dtype {first_frame.dtype} "
            f"into target dtype {target_dtype}"
        ) from exc
    finally:
        del first_frame

    # ------------------------------------------------------------
    # Remaining files
    # ------------------------------------------------------------

    def read_frame(
        frame_index: int,
    ) -> tuple[int, FileDataset, NDArray]:

        header, frame = _read_dicom(paths[frame_index])

        if frame.shape != expected_shape:
            raise ValueError(
                f"Frame {frame_index} has shape {frame.shape}; "
                f"expected {expected_shape}"
            )

        return frame_index, header, frame

    next_index = 1
    futures = set()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        # Keep only max_workers files in flight at once.
        while next_index < len(paths) and len(futures) < max_workers:
            futures.add(
                executor.submit(
                    read_frame,
                    next_index,
                )
            )
            next_index += 1

        with tqdm(
            total=len(paths),
            initial=1,
            desc="Loading DICOM data",
            unit="file",
        ) as progress:

            while futures:
                done, futures = wait(
                    futures,
                    return_when=FIRST_COMPLETED,
                )

                for future in done:
                    frame_index, header, frame = future.result()

                    headers[frame_index] = header

                    try:
                        np.copyto(
                            pixel_data[frame_index],
                            frame,
                            casting="unsafe",
                        )
                    finally:
                        del frame

                    progress.update(1)

                    # Submit exactly one replacement job for
                    # every completed job.
                    if next_index < len(paths):
                        futures.add(
                            executor.submit(
                                read_frame,
                                next_index,
                            )
                        )
                        next_index += 1

    # This should be guaranteed unless an exception occurred.
    result_headers = [header for header in headers if header is not None]

    _validate_pixel_layout(result_headers)

    return result_headers, pixel_data.tobytes(order="C")


def _read_chunk_headers(
    paths: Sequence[str],
    *,
    max_workers: int | None = 4,
) -> list[FileDataset]:
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be >= 1 or None")

    headers: list[FileDataset | None] = [None] * len(paths)

    def read_header(index: int) -> tuple[int, FileDataset]:
        path = paths[index]

        try:
            header = dcmread(
                path,
                force=True,
                stop_before_pixels=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to read DICOM header {path!r}") from exc

        return index, header

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(read_header, index) for index in range(len(paths))]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Loading chunk headers",
            unit="file",
        ):
            index, header = future.result()
            headers[index] = header

    return [header for header in headers if header is not None]


def _sort_chunk_by_instance(
    paths: Sequence[str],
    headers: Sequence[FileDataset],
) -> tuple[list[str], list[FileDataset]]:
    instances: list[int] = []

    for path, header in zip(paths, headers):
        value = header.get("InstanceNumber")
        if value is None:
            raise ValueError(f"InstanceNumber is missing from {path!r}")
        instances.append(int(value))

    order = np.argsort(np.asarray(instances, dtype=np.int64), kind="stable")
    sorted_paths = [paths[int(index)] for index in order]
    sorted_headers = [headers[int(index)] for index in order]
    sorted_instances = np.asarray(instances, dtype=np.int64)[order]

    if sorted_instances.size > 1 and np.any(
        sorted_instances[1:] <= sorted_instances[:-1]
    ):
        raise ValueError(
            "The chunk contains duplicate or non-increasing InstanceNumber values"
        )

    return sorted_paths, sorted_headers


def _pixel_dtype_from_header(ds: Dataset) -> np.dtype[Any]:
    bits_allocated = int(ds.BitsAllocated)
    pixel_representation = int(ds.get("PixelRepresentation", 0))

    if bits_allocated not in (8, 16, 32, 64):
        raise ValueError(
            f"Unsupported BitsAllocated={bits_allocated}; expected 8, 16, 32, or 64"
        )
    if pixel_representation not in (0, 1):
        raise ValueError(
            f"Unsupported PixelRepresentation={pixel_representation}; expected 0 or 1"
        )

    kind = "i" if pixel_representation else "u"
    byte_order = "|" if bits_allocated == 8 else "<"
    return np.dtype(f"{byte_order}{kind}{bits_allocated // 8}")


def _validate_pixel_layout(headers: Sequence[Dataset]) -> None:
    if not headers:
        raise ValueError("At least one DICOM header is required")

    keywords = (
        "Rows",
        "Columns",
        "SamplesPerPixel",
        "BitsAllocated",
        "BitsStored",
        "HighBit",
        "PixelRepresentation",
        "PhotometricInterpretation",
        "PlanarConfiguration",
    )

    first = headers[0]

    if int(first.get("NumberOfFrames", 1)) != 1:
        raise ValueError("This converter expects single-frame input DICOM files")

    for frame_index, header in enumerate(headers[1:], start=1):
        if int(header.get("NumberOfFrames", 1)) != 1:
            raise ValueError(
                f"Input frame {frame_index} is not a single-frame DICOM file"
            )

        for keyword in keywords:
            default: Any = None
            if keyword == "SamplesPerPixel":
                default = 1
            elif keyword == "PixelRepresentation":
                default = 0

            if first.get(keyword, default) != header.get(keyword, default):
                raise ValueError(
                    f"Input frame {frame_index} has a different {keyword}: "
                    f"{header.get(keyword, default)!r} != "
                    f"{first.get(keyword, default)!r}"
                )


def _read_pixel_array(path: str) -> np.ndarray:
    path_decoder_error: Exception | None = None

    if _decode_pixel_array_from_path is not None:
        try:
            return np.asarray(_decode_pixel_array_from_path(path))
        except Exception as exc:
            # Retain compatibility with files that require dcmread(force=True)
            # or an older pixel-data handler.
            path_decoder_error = exc

    try:
        ds = dcmread(path, force=True)
        array = np.asarray(ds.pixel_array)
        del ds
        return array
    except Exception as exc:
        message = f"Failed to decode Pixel Data from {path!r}"
        if path_decoder_error is not None:
            message += (
                "; both the path-based decoder and the compatibility " "fallback failed"
            )
        raise RuntimeError(message) from exc


def load_pixel_data_preallocated(
    paths: Sequence[str],
    headers: Sequence[Dataset],
    *,
    max_workers: int | None = 4,
) -> np.ndarray:
    """
    Decode frames concurrently into a single preallocated destination array.

    Memory usage is approximately:
        destination array
        + max_workers decoded frames
        + decoder/internal overhead

    rather than retaining all decoded frames before stacking.
    """

    if not paths:
        raise ValueError("Cannot load Pixel Data from an empty path sequence")
    if len(paths) != len(headers):
        raise ValueError("paths and headers must have equal lengths")
    if max_workers is None:
        cc = os.cpu_count()
        cc = cc if cc is not None else 1
        max_workers = 5 * cc
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1 or None")

    _validate_pixel_layout(headers)
    target_dtype = _pixel_dtype_from_header(headers[0])

    # Decode the first frame synchronously so we can determine the output shape.
    first_frame = _read_pixel_array(paths[0])
    expected_shape = first_frame.shape

    pixel_data = np.empty(
        (len(paths), *expected_shape),
        dtype=target_dtype,
        order="C",
    )

    try:
        np.copyto(pixel_data[0], first_frame, casting="unsafe")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not copy the first frame with shape {first_frame.shape} "
            f"and dtype {first_frame.dtype} into target dtype {target_dtype}"
        ) from exc
    finally:
        del first_frame

    def decode_frame(frame_index: int):
        frame = _read_pixel_array(paths[frame_index])

        if frame.shape != expected_shape:
            raise ValueError(
                f"Frame {frame_index} has shape {frame.shape}; "
                f"expected {expected_shape}"
            )

        return frame_index, frame

    # Never allow more than max_workers frames to be submitted/in flight.
    next_index = 1
    futures = set()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while next_index < len(paths) and len(futures) < max_workers:
            futures.add(executor.submit(decode_frame, next_index))
            next_index += 1

        with tqdm(
            total=len(paths),
            initial=1,
            desc="Loading Pixel Data",
            unit="frame",
        ) as progress:
            while futures:
                done, futures = wait(
                    futures,
                    return_when=FIRST_COMPLETED,
                )

                for future in done:
                    frame_index, frame = future.result()

                    try:
                        np.copyto(
                            pixel_data[frame_index],
                            frame,
                            casting="unsafe",
                        )
                    finally:
                        del frame

                    progress.update(1)

                    if next_index < len(paths):
                        futures.add(executor.submit(decode_frame, next_index))
                        next_index += 1

    return pixel_data


def _values_equal(left: Any, right: Any) -> bool:
    try:
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        return bool(np.array_equal(left_array, right_array, equal_nan=True))
    except (TypeError, ValueError):
        result = left == right
        if isinstance(result, np.ndarray):
            return bool(np.all(result))
        return bool(result)


def _all_same(
    headers: Sequence[Dataset],
    keyword: str,
    default: Any,
) -> bool:
    reference = headers[0].get(keyword, default)
    return all(
        _values_equal(header.get(keyword, default), reference) for header in headers[1:]
    )


def _add_plane_position(target: Dataset, ds: Dataset) -> None:
    sequence_item = Dataset()
    sequence_item.ImagePositionPatient = ds.get(
        "ImagePositionPatient",
        [0.0, 0.0, 0.0],
    )
    target.PlanePositionSequence = DicomSequence([sequence_item])


def _add_plane_orientation(target: Dataset, ds: Dataset) -> None:
    sequence_item = Dataset()
    sequence_item.ImageOrientationPatient = ds.get(
        "ImageOrientationPatient",
        [1, 0, 0, 0, 1, 0],
    )
    target.PlaneOrientationSequence = DicomSequence([sequence_item])


def _add_pixel_measures(target: Dataset, ds: Dataset) -> None:
    sequence_item = Dataset()
    sequence_item.SliceThickness = ds.get("SliceThickness", 1.0)
    sequence_item.PixelSpacing = ds.get("PixelSpacing", [1.0, 1.0])
    target.PixelMeasuresSequence = DicomSequence([sequence_item])


def _private_float_values(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if len(raw) % np.dtype(np.float32).itemsize != 0:
            raise ValueError(
                "Private float byte value has a length that is not divisible by 4"
            )
        return (
            np.frombuffer(raw, dtype=np.float32)
            .astype(
                np.float32,
                copy=False,
            )
            .tolist()
        )

    return np.asarray(value, dtype=np.float32).tolist()


def _add_varying_elements(item: Dataset, ds: Dataset) -> None:
    for tag in VARYING_PRIVATE_TAGS:
        if tag not in ds:
            continue

        element = ds[tag]
        if tag in _FLOAT32_PRIVATE_TAGS:
            item.add_new(tag, "FL", _private_float_values(element.value))
        else:
            item.add_new(tag, element.VR, element.value)

    for tag in VARYING_PUBLIC_TAGS:
        if tag not in ds:
            continue

        if tag == Tag(0x0020, 0x0013):
            item.add_new(tag, "IS", int(ds[tag].value))
        else:
            item.add_new(tag, "US", int(ds[tag].value))

    if hasattr(ds, "XrayTubeCurrent"):
        item.add_new(Tag(0x0018, 0x1151), "IS", ds.XrayTubeCurrent)


def _build_functional_groups(
    template: Dataset,
    headers: Sequence[Dataset],
) -> None:
    """Use shared functional groups whenever all frames have the same value."""

    position_shared = _all_same(
        headers,
        "ImagePositionPatient",
        [0.0, 0.0, 0.0],
    )
    orientation_shared = _all_same(
        headers,
        "ImageOrientationPatient",
        [1, 0, 0, 0, 1, 0],
    )
    pixel_spacing_shared = _all_same(
        headers,
        "PixelSpacing",
        [1.0, 1.0],
    )
    slice_thickness_shared = _all_same(
        headers,
        "SliceThickness",
        1.0,
    )
    pixel_measures_shared = pixel_spacing_shared and slice_thickness_shared

    shared_item = Dataset()
    first_header = headers[0]

    if position_shared:
        _add_plane_position(shared_item, first_header)
    if orientation_shared:
        _add_plane_orientation(shared_item, first_header)
    if pixel_measures_shared:
        _add_pixel_measures(shared_item, first_header)

    template.SharedFunctionalGroupsSequence = DicomSequence([shared_item])

    per_frame_sequence = DicomSequence([])

    for header in tqdm(headers, desc="Building functional groups", unit="frame"):
        item = Dataset()

        if not position_shared:
            _add_plane_position(item, header)
        if not orientation_shared:
            _add_plane_orientation(item, header)
        if not pixel_measures_shared:
            _add_pixel_measures(item, header)

        _add_varying_elements(item, header)
        per_frame_sequence.append(item)

    template.PerFrameFunctionalGroupsSequence = per_frame_sequence


def _prepare_file_meta(template: Dataset) -> None:
    if getattr(template, "file_meta", None) is None:
        template.file_meta = FileMetaDataset()

    template.is_little_endian = True
    template.is_implicit_VR = False
    template.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    template.file_meta.MediaStorageSOPInstanceUID = template.SOPInstanceUID
    template.file_meta.MediaStorageSOPClassUID = template.SOPClassUID


def _write_atomically(output_file: str, template: Dataset) -> None:
    partial_file = f"{output_file}.partial"

    try:
        estimated_bytes = int(pixel_data_nbytes(template) * 1.015)

        with open(partial_file, "wb") as f:
            with tqdm(
                total=estimated_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Saving {Path(output_file).name}",
            ) as pbar:
                dcmwrite(
                    TqdmWriter(
                        f,
                        pbar,
                        chunk_size=16 * 1024 * 1024,
                    ),
                    template,
                    enforce_file_format=True,
                )

    except Exception:
        try:
            os.remove(partial_file)
        except FileNotFoundError:
            pass
        raise

    else:
        os.replace(partial_file, output_file)
    # try:
    #     print("  Saving dcm...")
    #     dcmwrite(partial_file, template, enforce_file_format=False)
    #     os.replace(partial_file, output_file)
    # except Exception:
    #     try:
    #         os.remove(partial_file)
    #     except FileNotFoundError:
    #         pass
    #     raise


def save_paths_as_multiframe_ct_pd_cine_dicom(
    paths: Sequence[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
    output_name: str,
) -> str:
    """Convert one sorted chunk of single-frame DICOM files to one output."""

    normalized_paths = [os.fspath(path) for path in paths]
    if not normalized_paths:
        raise ValueError("No DICOM files were supplied")

    output_directory = os.fspath(output_dir)
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, output_name)
    headers, pixel_bytes = load_headers_and_pixel_data(
        normalized_paths, max_workers=None
    )

    number_of_frames = len(headers)
    print(f"  Frames: {number_of_frames:,}")
    print(
        "  InstanceNumber range: "
        f"{headers[0].InstanceNumber} ... {headers[-1].InstanceNumber}"
    )

    template = copy.deepcopy(headers[0])

    for tag in (*VARYING_PRIVATE_TAGS, *VARYING_PUBLIC_TAGS):
        if tag in template:
            del template[tag]

    if _PIXEL_DATA_TAG in template:
        del template[_PIXEL_DATA_TAG]

    template.NumberOfFrames = number_of_frames
    template.PixelData = pixel_bytes
    template[_PIXEL_DATA_TAG].VR = "OB" if int(template.BitsAllocated) <= 8 else "OW"
    del pixel_bytes

    # Preserve the original script's cine values and SOP Class behavior.
    template.FrameTime = 1.0 / 30.0
    template.FrameIncrementPointer = [Tag(0x0018, 0x1063)]
    template.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    template.SOPInstanceUID = generate_uid()

    _build_functional_groups(template, headers)
    _prepare_file_meta(template)

    _write_atomically(output_file, template)

    output_size = os.path.getsize(output_file)
    if output_size > OUTPUT_FILE_LIMIT_BYTES:
        os.remove(output_file)
        raise RuntimeError(
            f"Output file would be {output_size:,} bytes, exceeding the "
            f"configured output ceiling of {OUTPUT_FILE_LIMIT_BYTES:,} bytes"
        )

    print(f"  Created: {output_file}")
    print(f"  Output size: {output_size / 1024**2:,.1f} MiB")
    return output_file


# Backward-compatible public function name from the original script.
def save_files_as_multiframe_ct_pd_cine_dicom(
    sources: Sequence[str | os.PathLike[str] | Dataset],
    output_dir: str | os.PathLike[str],
    output_name: str,
) -> str:
    paths = [_source_path(source) for source in sources]
    return save_paths_as_multiframe_ct_pd_cine_dicom(
        paths,
        output_dir,
        output_name,
    )


def save_folder_as_multiframe_ct_pd_cine_dicom(
    folder_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
    output_name: str | None = None,
) -> str | None:
    """Compatibility wrapper for converting all DICOM files in one folder."""

    folder = os.fspath(folder_path)
    paths = [
        os.path.join(folder, filename)
        for filename in os.listdir(folder)
        if filename.lower().endswith((".dcm", ".ima"))
        and not filename.lower().endswith(_GENERATED_FILE_SUFFIX)
    ]

    if not paths:
        print(f"No DICOM files found in {folder!r}; skipped")
        return None

    if output_dir is None:
        output_dir = folder
    if output_name is None:
        batch_name = os.path.basename(folder.rstrip("\\/"))
        output_name = f"{batch_name}_multiframe_ct_pd.dcm"

    return save_paths_as_multiframe_ct_pd_cine_dicom(
        paths,
        output_dir,
        output_name,
    )


# -----------------------------------------------------------------------------
# End-to-end processing
# -----------------------------------------------------------------------------


def print_split_plan(
    tube: int,
    threshold_groups: dict[int, np.ndarray],
    plan: TubeSplitPlan,
) -> None:
    thresholds = ", ".join(str(value) for value in sorted(threshold_groups))
    print(
        f"Tube {tube}: {plan.number_of_groups} output chunk(s), "
        f"thresholds [{thresholds}]"
    )
    max_decimal_inst = int(np.ceil(np.log10(plan.last_instances[-1])))
    max_decimal_ch = int(np.ceil(np.log10(len(plan.last_instances) + 1)))

    for chunk_number, (first, last, count) in enumerate(
        zip(plan.first_instances, plan.last_instances, plan.sizes),
        start=1,
    ):
        byte_summary = ", ".join(
            (
                f"Thr{threshold}="
                f"{int(plan.bytes_by_threshold[threshold][chunk_number - 1]) / 1024**2:,.1f} MiB"
            )
            for threshold in sorted(plan.bytes_by_threshold)
        )
        print(
            f"  Chunk {chunk_number:0{max_decimal_ch}d}: {int(count):,} frames, "
            f"instances {int(first):0{max_decimal_inst}d}...{int(last):0{max_decimal_inst}d}, {byte_summary}"
        )


def process_dataset_tree(
    parent_dir: str | os.PathLike[str],
    *,
    output_root: str | os.PathLike[str] | None = None,
    max_pixel_bytes: int = DEFAULT_MAX_PIXEL_BYTES,
    metadata_reserve_bytes: int = DEFAULT_METADATA_RESERVE_BYTES,
    max_frames_per_chunk: int | None = DEFAULT_MAX_FRAMES_PER_CHUNK,
    extensions: Iterable[str] = (".dcm", ".ima"),
    dry_run: bool = False,
) -> list[str]:
    """Index, plan, and process all tube/threshold series below a root."""

    parent = os.fspath(parent_dir)
    output_base = parent if output_root is None else os.fspath(output_root)
    usable_pixel_limit = calculate_usable_pixel_limit(
        max_pixel_bytes,
        metadata_reserve_bytes,
    )

    paths = discover_dicom_files(
        parent,
        extensions=extensions,
        exclude_generated=True,
    )
    if not paths:
        raise FileNotFoundError(f"No DICOM files were found below {parent!r}")

    print(f"Found {len(paths):,} source DICOM files below {parent}")
    print(
        "Operational Pixel Data cap per output: "
        f"{usable_pixel_limit / 1024**2:,.1f} MiB"
    )
    print(
        "Reserved metadata allowance: " f"{metadata_reserve_bytes / 1024**2:,.1f} MiB"
    )
    if max_frames_per_chunk is not None:
        print(f"Maximum frames per output: {max_frames_per_chunk:,}")

    indexed = build_file_index(paths)
    del paths
    output_files: list[str] = []

    for tube, threshold_groups in sorted(indexed.grouped.items()):
        plan = make_tube_split_plan(
            tube,
            threshold_groups,
            usable_pixel_limit=usable_pixel_limit,
            max_frames_per_chunk=max_frames_per_chunk,
        )
        print_split_plan(tube, threshold_groups, plan)

        if dry_run:
            continue

        for threshold, group in sorted(threshold_groups.items()):
            output_dir = os.path.join(
                output_base,
                f"Tube{tube}_Thr{threshold}",
            )

            for chunk_number, (start, stop) in enumerate(
                zip(plan.starts, plan.stops),
                start=1,
            ):
                chunk = group[int(start) : int(stop)]
                chunk_paths = [
                    indexed.paths[int(path_index)] for path_index in chunk["path_index"]
                ]

                print(
                    f"\nTube {tube}, threshold {threshold}, "
                    f"chunk {chunk_number:05d}/{plan.number_of_groups:05d}"
                )

                output_files.append(
                    save_paths_as_multiframe_ct_pd_cine_dicom(
                        chunk_paths,
                        output_dir,
                        f"Batch_{chunk_number:05d}_multiframe_ct_pd.dcm",
                    )
                )

                # Do not retain chunk-specific paths or pydicom objects beyond
                # the completed write.
                del chunk_paths
                del chunk

    return output_files


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert single-frame DICOM-CT-PD files into memory-bounded, "
            "aligned multi-frame outputs."
        )
    )
    parser.add_argument(
        "parent_dir",
        nargs="?",
        default=DEFAULT_PARENT_DIR,
        help="Root containing the source DICOM files",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root; defaults to parent_dir",
    )
    parser.add_argument(
        "--max-pixel-mib",
        type=_positive_int,
        default=DEFAULT_MAX_PIXEL_BYTES // 1024**2,
        help="Maximum uncompressed Pixel Data payload per output (default: 512)",
    )
    parser.add_argument(
        "--max-frames-per-chunk",
        type=_positive_int,
        default=DEFAULT_MAX_FRAMES_PER_CHUNK,
        help="Maximum number of per-frame metadata items per output (default: 10000)",
    )
    parser.add_argument(
        "--metadata-reserve-mib",
        type=_nonnegative_int,
        default=DEFAULT_METADATA_RESERVE_BYTES // 1024**2,
        help="Space reserved below the output-size ceiling (default: 64)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Index and print the split plans without writing output files",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    output_files = process_dataset_tree(
        args.parent_dir,
        output_root=args.output_root,
        max_pixel_bytes=args.max_pixel_mib * 1024**2,
        metadata_reserve_bytes=args.metadata_reserve_mib * 1024**2,
        max_frames_per_chunk=args.max_frames_per_chunk,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\nDry run complete; no files were written.")
    else:
        print(f"\nAll batches finished. Created {len(output_files):,} files.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
