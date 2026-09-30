import os
import re
from threading import Lock
from collections.abc import Sequence, Iterable
from tqdm.contrib.concurrent import thread_map
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
from tqdm import tqdm
from copy import deepcopy
import numpy as np
from pydicom.dataset import FileDataset
from pydicom import dcmread, dcmwrite, Dataset
from pydicom.sequence import Sequence as dcm_Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from pydicom.tag import Tag
from dataclasses import dataclass

import numpy as np

GroupedHeaders = dict[int, dict[int, np.ndarray]]


ChunkedHeaders = dict[
    int,
    dict[int, tuple[np.ndarray, ...]],
]
FILE_LIMIT = 4_294_967_296
_TUBE_PATTERN = re.compile(r"\bTube(\d+)", re.IGNORECASE)
_THRESHOLD_PATTERN = re.compile(r"\bThr(\d+)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class TubeSplitPlan:
    starts: np.ndarray
    stops: np.ndarray
    sizes: np.ndarray
    first_instances: np.ndarray
    last_instances: np.ndarray
    bytes_by_threshold: dict[int, np.ndarray]

    @property
    def number_of_groups(self) -> int:
        return int(self.sizes.size)


_HEADER_DTYPE = np.dtype(
    [
        ("tube", np.uint8),
        ("threshold", np.uint8),
        ("instance", np.int32),
        ("pixel_bytes", np.uint64),
        ("dataset", object),
    ]
)


def split_grouped_headers_evenly(
    grouped: GroupedHeaders,
    *,
    file_limit: int = FILE_LIMIT,
    reserved_bytes: int = 0,
) -> tuple[ChunkedHeaders, dict[int, TubeSplitPlan]]:
    """
    Split every tube/threshold series into balanced, aligned chunks.

    A separate plan is calculated for each tube because different tubes
    may have different instance-number sequences or instance counts.

    Within a tube, every threshold receives exactly the same instance
    boundaries.
    """

    if file_limit <= 0:
        raise ValueError("file_limit must be positive")

    if not 0 <= reserved_bytes < file_limit:
        raise ValueError(
            "reserved_bytes must be nonnegative and smaller than " "file_limit"
        )

    usable_file_limit = file_limit - reserved_bytes

    chunks: ChunkedHeaders = {}
    plans: dict[int, TubeSplitPlan] = {}

    for tube, threshold_groups in sorted(grouped.items()):
        plan = make_tube_split_plan(
            tube,
            threshold_groups,
            usable_file_limit=usable_file_limit,
        )

        plans[tube] = plan
        chunks[tube] = {}

        for threshold, group in sorted(threshold_groups.items()):
            chunks[tube][threshold] = tuple(
                group[int(start) : int(stop)]
                for start, stop in zip(
                    plan.starts,
                    plan.stops,
                )
            )

    return chunks, plans


def make_tube_split_plan(
    tube: int,
    threshold_groups: dict[int, np.ndarray],
    *,
    usable_file_limit: int,
) -> TubeSplitPlan:
    """
    Create one common split plan for all thresholds in a tube.

    All thresholds must have exactly the same ordered InstanceNumber
    sequence. The returned boundaries are therefore guaranteed to
    represent the same instances at every threshold.
    """

    if not threshold_groups:
        raise ValueError(f"Tube {tube} has no threshold groups")

    threshold_items = sorted(threshold_groups.items())

    reference_threshold, reference_group = threshold_items[0]
    reference_instances = reference_group["instance"]
    number_of_instances = int(reference_group.size)

    if number_of_instances == 0:
        raise ValueError(f"Tube {tube}, threshold {reference_threshold} is empty")

    # InstanceNumber should be strictly increasing after the earlier sort.
    if number_of_instances > 1 and np.any(
        reference_instances[1:] <= reference_instances[:-1]
    ):
        raise ValueError(
            f"Tube {tube}, threshold {reference_threshold} has "
            "duplicate or non-increasing instance numbers"
        )

    # Exact instance alignment is required, not merely equal lengths.
    for threshold, group in threshold_items[1:]:
        if group.size != number_of_instances:
            raise ValueError(
                f"Tube {tube} thresholds do not have equal instance "
                f"counts: threshold {reference_threshold} has "
                f"{number_of_instances:,}, while threshold "
                f"{threshold} has {group.size:,}"
            )

        if not np.array_equal(
            group["instance"],
            reference_instances,
        ):
            differing_positions = np.flatnonzero(
                group["instance"] != reference_instances
            )
            position = int(differing_positions[0])

            raise ValueError(
                f"Tube {tube}, threshold {threshold} is not aligned "
                f"with threshold {reference_threshold}. At sorted "
                f"position {position}, the instance numbers are "
                f"{int(group['instance'][position])} and "
                f"{int(reference_instances[position])}."
            )

    prefix_by_threshold: dict[int, np.ndarray] = {}

    # A lower bound on the required number of files comes from each
    # threshold's total byte count.
    minimum_number_of_groups = 1

    for threshold, group in threshold_items:
        too_large = np.flatnonzero(group["pixel_bytes"] > usable_file_limit)

        if too_large.size:
            position = int(too_large[0])

            raise ValueError(
                f"Tube {tube}, threshold {threshold}, instance "
                f"{int(group['instance'][position])} requires "
                f"{int(group['pixel_bytes'][position]):,} bytes, "
                f"which exceeds the usable file limit of "
                f"{usable_file_limit:,} bytes"
            )

        prefix = pixel_byte_prefix_sum(group)
        prefix_by_threshold[threshold] = prefix

        total_bytes = int(prefix[-1])

        required_for_threshold = (
            total_bytes + usable_file_limit - 1
        ) // usable_file_limit

        minimum_number_of_groups = max(
            minimum_number_of_groups,
            required_for_threshold,
        )

    # Try the fewest possible groups first. Each candidate is balanced
    # by instance count and uses identical boundaries at every threshold.
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

            if np.any(chunk_bytes > usable_file_limit):
                fits_all_thresholds = False
                break

        if not fits_all_thresholds:
            continue

        return TubeSplitPlan(
            starts=starts,
            stops=stops,
            sizes=sizes,
            first_instances=reference_instances[starts].copy(),
            last_instances=reference_instances[stops - 1].copy(),
            bytes_by_threshold=bytes_by_threshold,
        )

    # One instance per group must work because oversized individual
    # instances were rejected above.
    raise RuntimeError(f"Could not construct a valid split plan for tube {tube}")


def pixel_byte_prefix_sum(group: np.ndarray) -> np.ndarray:
    prefix = np.empty(group.size + 1, dtype=np.uint64)
    prefix[0] = 0

    np.cumsum(
        group["pixel_bytes"],
        dtype=np.uint64,
        out=prefix[1:],
    )

    return prefix


def balanced_boundaries(
    number_of_instances: int,
    number_of_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return balanced [start, stop) boundaries.

    Group sizes differ by no more than one instance. Any remainder
    is distributed among the earliest groups.

    Example
    -------
    30_101 instances split into 3 groups produces:

        [10_034, 10_034, 10_033]
    """

    if number_of_instances <= 0:
        raise ValueError("number_of_instances must be positive")

    if not 1 <= number_of_groups <= number_of_instances:
        raise ValueError(
            "number_of_groups must be between 1 and " "number_of_instances"
        )

    base_size, remainder = divmod(
        number_of_instances,
        number_of_groups,
    )

    sizes = np.full(
        number_of_groups,
        base_size,
        dtype=np.intp,
    )
    sizes[:remainder] += 1

    stops = np.cumsum(sizes, dtype=np.intp)

    starts = np.empty_like(stops)
    starts[0] = 0
    starts[1:] = stops[:-1]

    return starts, stops, sizes


def pixel_data_nbytes(ds: Dataset) -> int:
    rows = int(ds.Rows)
    columns = int(ds.Columns)
    samples_per_pixel = int(ds.SamplesPerPixel)
    bits_allocated = int(ds.BitsAllocated)
    number_of_frames = int(ds.get("NumberOfFrames", 1))

    if rows <= 0 or columns <= 0:
        raise ValueError("Rows and Columns must be positive")

    if samples_per_pixel <= 0:
        raise ValueError("SamplesPerPixel must be positive")

    if bits_allocated <= 0:
        raise ValueError("BitsAllocated must be positive")

    if number_of_frames <= 0:
        raise ValueError("NumberOfFrames must be positive")

    total_bits = rows * columns * samples_per_pixel * number_of_frames * bits_allocated

    # Round up in case the total is not byte-aligned.
    return (total_bits + 7) // 8


def organize_headers(
    headers: Sequence[Dataset],
) -> tuple[np.ndarray, dict[int, dict[int, np.ndarray]]]:
    """
    Create a flat, sorted header array and group it by tube and threshold.

    Returns
    -------
    records:
        One-dimensional structured array sorted by:
        tube -> threshold -> instance.

    grouped:
        Nested dictionary accessed as:

            grouped[tube_number][threshold_number]

        Each terminal value is a structured array sorted by instance number.
    """

    records = np.empty(len(headers), dtype=_HEADER_DTYPE)
    records["dataset"] = headers

    for index, ds in enumerate(headers):
        series_description = str(ds.get("SeriesDescription", ""))

        tube_match = _TUBE_PATTERN.search(series_description)
        if tube_match is None:
            raise ValueError(
                f"Dataset {index} has no tube number in "
                f"SeriesDescription={series_description!r}"
            )

        threshold_match = _THRESHOLD_PATTERN.search(series_description)
        if threshold_match is None:
            raise ValueError(
                f"Dataset {index} has no threshold number in "
                f"SeriesDescription={series_description!r}"
            )

        instance_value = ds.get("InstanceNumber")
        if instance_value is None:
            raise ValueError(f"Dataset {index} does not contain InstanceNumber")

        try:
            instance = int(instance_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Dataset {index} has an invalid " f"InstanceNumber={instance_value!r}"
            ) from exc

        tube = int(tube_match.group(1))
        threshold = int(threshold_match.group(1))
        pixel_bytes = pixel_data_nbytes(ds)

        records[index] = (
            tube,
            threshold,
            instance,
            pixel_bytes,
            ds,
        )

    # np.lexsort uses the final key as the primary key:
    # tube -> threshold -> instance.
    order = np.lexsort(
        (
            records["instance"],
            records["threshold"],
            records["tube"],
        )
    )

    records = records[order]

    # Because the records have already been sorted, every tube/threshold
    # combination occupies one contiguous section of the array.
    grouped: dict[int, dict[int, np.ndarray]] = {}

    if records.size == 0:
        return records, grouped

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

        # This is a view into records, not a copy of the DICOM datasets.
        grouped.setdefault(tube, {})[threshold] = records[start:stop]

    return records, grouped


def save_folder_as_multiframe_ct_pd_cine_dicom(
    folder_path, output_dir=None, output_name=None
):
    """
    Convert single-frame DICOM-CT-PD files in one folder into one multi-frame DICOM-CT-PD file.

    Parameters
    ----------
    folder_path : str
        Path to the folder that contains the single-frame .dcm files.
    output_dir : str, optional
        Directory where the multi-frame file will be saved.
        If None, the multi-frame file is saved inside folder_path.
    output_name : str, optional
        Name of the output multi-frame file (without path).
        If None, a default name is generated.
    """
    print(f"\n=== Processing folder: {folder_path} ===")

    # ------------------------------------------------------------------
    # 1. Collect and sort files by InstanceNumber
    # ------------------------------------------------------------------
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".dcm", ".ima"))]

    if not files:
        print(f"  No DICOM files found in {folder_path} → skipped")
        return None

    files = np.array(files)
    ds_headers = []
    ds_pixel_data = []
    for fname in tqdm(files, desc="Allocating headers and pixel data"):
        ds = dcmread(os.path.join(folder_path, fname), force=True)
        pixel_data = ds.pixel_array
        header = deepcopy(ds)
        if "PixelData" in header:
            del header.PixelData
        ds_headers.append(header)
        ds_pixel_data.append(pixel_data)
    ds_headers = np.array(ds_headers)
    ds_pixel_data = np.array(ds_pixel_data).astype(np.uint16)

    indices = np.argsort([int(d.InstanceNumber) for d in ds_headers])
    files = files[indices]
    ds_headers = ds_headers[indices]
    ds_pixel_data = ds_pixel_data[indices]

    num_frames = len(ds_headers)
    print(f"  Found {num_frames} files (sorted by InstanceNumber)")
    print(
        f"  InstanceNumber range: {ds_headers[0].InstanceNumber} … {ds_headers[-1].InstanceNumber}"
    )

    # ------------------------------------------------------------------
    # 2. Create template from first file
    # ------------------------------------------------------------------
    template = ds_headers[0].copy()

    # Tags that must be per-frame (private + public)
    varying_private_tags = [
        Tag(0x7031, 0x1001),  # DetectorFocalCenterAngularPosition
        Tag(0x7031, 0x1002),  # DetectorFocalCenterAxialPosition
        Tag(0x7031, 0x1003),  # DetectorFocalCenterRadialDistance
        Tag(0x7033, 0x100B),  # SourceAngularPositionShift
        Tag(0x7033, 0x1065),  # PhotonStatistics
        Tag(0x7033, 0x1067),  # Timestamp
    ]

    # Public tags that also vary per frame
    # InstanceNumber (0020,0013) is included so each frame keeps its projection order
    varying_public_tags = [
        Tag(0x0020, 0x0013),  # InstanceNumber
        Tag(0x0028, 0x0106),  # SmallestImagePixelValue
        Tag(0x0028, 0x0107),  # LargestImagePixelValue
    ]

    # Remove varying tags from the root dataset
    # (they will be stored only inside PerFrameFunctionalGroupsSequence)
    for t in varying_private_tags + varying_public_tags:
        if t in template:
            del template[t]

    # ------------------------------------------------------------------
    # 3. Prepare multi-frame attributes
    # ------------------------------------------------------------------
    template.NumberOfFrames = num_frames
    template.PixelData = ds_pixel_data.tobytes()

    # Multi-frame + cine
    template.FrameTime = 1.0 / 30.0
    template.FrameIncrementPointer = [0x0018, 0x1063]
    template.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    template.SOPInstanceUID = generate_uid()

    # ------------------------------------------------------------------
    # 4. Shared Functional Groups
    # ------------------------------------------------------------------
    shared_item = Dataset()
    shared_item.PixelMeasuresSequence = dcm_Sequence([Dataset()])
    shared_item.PixelMeasuresSequence[0].PixelSpacing = ds_headers[0].get(
        "PixelSpacing", [1.0, 1.0]
    )
    shared_item.PixelMeasuresSequence[0].SliceThickness = 1.0
    template.SharedFunctionalGroupsSequence = dcm_Sequence([shared_item])

    # ------------------------------------------------------------------
    # 5. Per-Frame Functional Groups
    # ------------------------------------------------------------------
    per_frame_items = []
    for ds in tqdm(ds_headers, desc="Grouping tags"):
        item = Dataset()

        # Standard sequences
        item.PlanePositionSequence = dcm_Sequence([Dataset()])
        item.PlanePositionSequence[0].ImagePositionPatient = ds.get(
            "ImagePositionPatient", [0.0, 0.0, 0.0]
        )

        item.PlaneOrientationSequence = dcm_Sequence([Dataset()])
        item.PlaneOrientationSequence[0].ImageOrientationPatient = ds.get(
            "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]
        )

        item.PixelMeasuresSequence = dcm_Sequence([Dataset()])
        meas = item.PixelMeasuresSequence[0]
        meas.SliceThickness = ds.get("SliceThickness", 1.0)
        meas.PixelSpacing = ds.get("PixelSpacing", [1.0, 1.0])

        # Varying private tags
        for t in varying_private_tags:
            if t in ds:
                elem = ds[t]
                if (
                    t.group == 0x7031
                    or t == Tag(0x7033, 0x1065)
                    or t == Tag(0x7033, 0x1067)
                ):
                    if isinstance(elem.value, bytes):
                        val = np.frombuffer(elem.value, dtype=np.float32).tolist()
                    else:
                        val = np.asarray(elem.value, dtype=np.float32).tolist()
                    item.add_new(t, "FL", val)
                else:
                    item.add_new(t, elem.VR, elem.value)

        # Varying public tags (InstanceNumber, Smallest / Largest Image Pixel Value)
        # InstanceNumber is written per frame so readers can recover projection order
        for t in varying_public_tags:
            if t in ds:
                if t == Tag(0x0020, 0x0013):  # InstanceNumber → IS
                    item.add_new(t, "IS", int(ds[t].value))
                else:  # Smallest / Largest → US
                    item.add_new(t, "US", int(ds[t].value))

        # Varying standard (e.g., XrayTubeCurrent if present)
        if hasattr(ds, "XrayTubeCurrent"):
            item.add_new(Tag(0x0018, 0x1151), "IS", ds.XrayTubeCurrent)

        per_frame_items.append(item)

    # Per-frame sequence
    template.PerFrameFunctionalGroupsSequence = dcm_Sequence(per_frame_items)

    # ------------------------------------------------------------------
    # 6. Transfer syntax
    # ------------------------------------------------------------------
    # Set transfer syntax (explicit VR LE, as per MATLAB output)
    template.is_little_endian = True
    template.is_implicit_VR = False
    if hasattr(template, "file_meta"):
        template.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        template.file_meta.MediaStorageSOPInstanceUID = template.SOPInstanceUID
        template.file_meta.MediaStorageSOPClassUID = template.SOPClassUID

    # ------------------------------------------------------------------
    # 7. Determine output path
    # ------------------------------------------------------------------
    if output_dir is None:
        output_dir = folder_path
    os.makedirs(output_dir, exist_ok=True)

    if output_name is None:
        batch_name = os.path.basename(folder_path.rstrip("\\/"))
        output_name = f"{batch_name}_multiframe_ct_pd.dcm"

    output_file = os.path.join(output_dir, output_name)

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    dcmwrite(output_file, template, write_like_original=False)

    print(f"  ✅ Created multi-frame file with {num_frames} frames:")
    print(f"     {output_file}")
    print(
        f"     Per-frame InstanceNumber preserved in PerFrameFunctionalGroupsSequence"
    )
    return output_file


def save_files_as_multiframe_ct_pd_cine_dicom(ds_headers, output_dir, output_name):
    """
    Convert single-frame DICOM-CT-PD files in one folder into one multi-frame DICOM-CT-PD file.

    Parameters
    ----------
    folder_path : str
        Path to the folder that contains the single-frame .dcm files.
    output_dir : str, optional
        Directory where the multi-frame file will be saved.
        If None, the multi-frame file is saved inside folder_path.
    output_name : str, optional
        Name of the output multi-frame file (without path).
        If None, a default name is generated.
    """

    ds_pixel_data = get_pixel_data(ds_headers)

    num_frames = len(ds_headers)
    print(f"  Found {num_frames} files (sorted by InstanceNumber)")
    print(
        f"  InstanceNumber range: {ds_headers[0].InstanceNumber} … {ds_headers[-1].InstanceNumber}"
    )

    # ------------------------------------------------------------------
    # 2. Create template from first file
    # ------------------------------------------------------------------
    template = ds_headers[0].copy()

    # Tags that must be per-frame (private + public)
    varying_private_tags = [
        Tag(0x7031, 0x1001),  # DetectorFocalCenterAngularPosition
        Tag(0x7031, 0x1002),  # DetectorFocalCenterAxialPosition
        Tag(0x7031, 0x1003),  # DetectorFocalCenterRadialDistance
        Tag(0x7033, 0x100B),  # SourceAngularPositionShift
        Tag(0x7033, 0x1065),  # PhotonStatistics
        Tag(0x7033, 0x1067),  # Timestamp
    ]

    # Public tags that also vary per frame
    # InstanceNumber (0020,0013) is included so each frame keeps its projection order
    varying_public_tags = [
        Tag(0x0020, 0x0013),  # InstanceNumber
        Tag(0x0028, 0x0106),  # SmallestImagePixelValue
        Tag(0x0028, 0x0107),  # LargestImagePixelValue
    ]

    # Remove varying tags from the root dataset
    # (they will be stored only inside PerFrameFunctionalGroupsSequence)
    for t in varying_private_tags + varying_public_tags:
        if t in template:
            del template[t]

    # ------------------------------------------------------------------
    # 3. Prepare multi-frame attributes
    # ------------------------------------------------------------------
    template.NumberOfFrames = num_frames
    template.PixelData = ds_pixel_data.tobytes()

    # Multi-frame + cine
    template.FrameTime = 1.0 / 30.0
    template.FrameIncrementPointer = [0x0018, 0x1063]
    template.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    template.SOPInstanceUID = generate_uid()

    # ------------------------------------------------------------------
    # 4. Shared Functional Groups
    # ------------------------------------------------------------------
    shared_item = Dataset()
    shared_item.PixelMeasuresSequence = dcm_Sequence([Dataset()])
    shared_item.PixelMeasuresSequence[0].PixelSpacing = ds_headers[0].get(
        "PixelSpacing", [1.0, 1.0]
    )
    shared_item.PixelMeasuresSequence[0].SliceThickness = 1.0
    template.SharedFunctionalGroupsSequence = dcm_Sequence([shared_item])

    # ------------------------------------------------------------------
    # 5. Per-Frame Functional Groups
    # ------------------------------------------------------------------
    per_frame_items = []
    for ds in tqdm(ds_headers, desc="Grouping tags"):
        item = Dataset()

        # Standard sequences
        item.PlanePositionSequence = dcm_Sequence([Dataset()])
        item.PlanePositionSequence[0].ImagePositionPatient = ds.get(
            "ImagePositionPatient", [0.0, 0.0, 0.0]
        )

        item.PlaneOrientationSequence = dcm_Sequence([Dataset()])
        item.PlaneOrientationSequence[0].ImageOrientationPatient = ds.get(
            "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]
        )

        item.PixelMeasuresSequence = dcm_Sequence([Dataset()])
        meas = item.PixelMeasuresSequence[0]
        meas.SliceThickness = ds.get("SliceThickness", 1.0)
        meas.PixelSpacing = ds.get("PixelSpacing", [1.0, 1.0])

        # Varying private tags
        for t in varying_private_tags:
            if t in ds:
                elem = ds[t]
                if (
                    t.group == 0x7031
                    or t == Tag(0x7033, 0x1065)
                    or t == Tag(0x7033, 0x1067)
                ):
                    if isinstance(elem.value, bytes):
                        val = np.frombuffer(elem.value, dtype=np.float32).tolist()
                    else:
                        val = np.asarray(elem.value, dtype=np.float32).tolist()
                    item.add_new(t, "FL", val)
                else:
                    item.add_new(t, elem.VR, elem.value)

        # Varying public tags (InstanceNumber, Smallest / Largest Image Pixel Value)
        # InstanceNumber is written per frame so readers can recover projection order
        for t in varying_public_tags:
            if t in ds:
                if t == Tag(0x0020, 0x0013):  # InstanceNumber → IS
                    item.add_new(t, "IS", int(ds[t].value))
                else:  # Smallest / Largest → US
                    item.add_new(t, "US", int(ds[t].value))

        # Varying standard (e.g., XrayTubeCurrent if present)
        if hasattr(ds, "XrayTubeCurrent"):
            item.add_new(Tag(0x0018, 0x1151), "IS", ds.XrayTubeCurrent)

        per_frame_items.append(item)

    # Per-frame sequence
    template.PerFrameFunctionalGroupsSequence = dcm_Sequence(per_frame_items)

    # ------------------------------------------------------------------
    # 6. Transfer syntax
    # ------------------------------------------------------------------
    # Set transfer syntax (explicit VR LE, as per MATLAB output)
    template.is_little_endian = True
    template.is_implicit_VR = False
    if hasattr(template, "file_meta"):
        template.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        template.file_meta.MediaStorageSOPInstanceUID = template.SOPInstanceUID
        template.file_meta.MediaStorageSOPClassUID = template.SOPClassUID

    # ------------------------------------------------------------------
    # 7. Determine output path
    # ------------------------------------------------------------------

    output_file = os.path.join(output_dir, output_name)

    # ------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------
    dcmwrite(output_file, template, write_like_original=False)

    print(f"  ✅ Created multi-frame file with {num_frames} frames:")
    print(f"     {output_file}")
    print(
        f"     Per-frame InstanceNumber preserved in PerFrameFunctionalGroupsSequence"
    )
    return output_file


def get_single_header(h):
    if isinstance(h, str):
        return dcmread(h, force=True, stop_before_pixels=True)
    elif isinstance(h, FileDataset):
        return dcmread(h.filename, force=True, stop_before_pixels=True)


def get_single_pixel_data(h):
    if isinstance(h, str):
        return dcmread(h, force=True).pixel_array
    elif isinstance(h, FileDataset):
        return dcmread(h.filename, force=True).pixel_array


def get_headers(files):
    return np.array(thread_map(get_single_header, files))


def get_pixel_data(files):
    with ThreadPoolExecutor() as executor:
        e_map = executor.map(get_single_pixel_data, files)
        ds_pixel_data = list(tqdm(e_map, total=len(files)))
    return np.array(ds_pixel_data).astype(np.uint16)


def split_by_tube(headers):
    tubes = np.array(
        [int(re.search(r"\bTube(\d+)", d.SeriesDescription).group(1)) for d in headers]
    )
    sort_indices = np.argsort(tubes)
    headers = headers[sort_indices]
    tubes = tubes[sort_indices]

    # 3. Get the starting indices of unique values (skip the first index 0)
    _, unique_indices = np.unique(tubes, return_index=True)
    split_indices = unique_indices[1:]

    # 4. Split the target array
    return np.split(headers, split_indices)


def split_by_threshold(headers):
    tubes = np.array(
        [int(re.search(r"\bTube(\d+)", d.SeriesDescription).group(1)) for d in headers]
    )
    sort_indices = np.argsort(tubes)
    headers = headers[sort_indices]
    tubes = tubes[sort_indices]

    # 3. Get the starting indices of unique values (skip the first index 0)
    _, unique_indices = np.unique(tubes, return_index=True)
    split_indices = unique_indices[1:]

    # 4. Split the target array
    return np.split(headers, split_indices)


# ----------------------------------------------------------------------
# Main: process all Batch_* subfolders
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parent_dir = (
        r"U:\PUBLIC\DICOM-CTPD\PCD_DICOMCTPD_Generator\Full_dose_CTPD_Single_Frame"
    )
    input_dirs = glob.glob(os.path.join(parent_dir, "**", "*batch_*"))
    # input_dirs = [parent_dir]
    input_dcms = np.array(
        glob.glob(os.path.join(parent_dir, "**", "*.dcm"), recursive=True)
    )
    # with np.load(
    #     r"U:\PUBLIC\DICOM-CTPD\PCD_DICOMCTPD_Generator\Full_dose_CTPD_Single_Frame\ds_headers.npz"
    # ) as data:
    #     ds_headers = data["ds_headers"]
    ds_headers = get_headers(input_dcms)
    records, grouped = organize_headers(ds_headers)

    chunks, plans = split_grouped_headers_evenly(
        grouped,
        file_limit=FILE_LIMIT,
        reserved_bytes=0,
    )
    for tube, thresholds in chunks.items():
        for threshold, _chunks in thresholds.items():
            for c, chunk in enumerate(_chunks, start=1):
                save_files_as_multiframe_ct_pd_cine_dicom(
                    chunk["dataset"],
                    os.path.join(parent_dir, f"Tube{tube}_Thr{threshold}"),
                    f"Batch_{c:05d}_multiframe_ct_pd.dcm",
                )
    # for tube, thresholds in chunks.items():
    #     for threshold in thresholds.items():
    #         mask = (ds_tubes == tube) * (ds_thresholds == threshold)
    #         threshold_headers = ds_headers[mask]
    #         threshold_files = input_dcms[mask]
    #         file_sizes = np.array(
    #             [
    #                 ds.Rows * ds.Columns * ds.SamplesPerPixel * (ds.BitsAllocated / 8)
    #                 for ds in threshold_headers
    #             ]
    #         )
    #         diffs = np.diff([s % FILE_LIMIT for s in np.cumsum(file_sizes)])
    #         jump_indices = np.where(diffs < -FILE_LIMIT // 2)[0]
    #         jump_cnt = len(jump_indices)
    #         all_indices = np.linspace(
    #             0, len(threshold_headers), num=jump_cnt + 2
    #         ).astype(np.uint)
    #         start_indices = all_indices[:-1]
    #         end_indices = all_indices[1:]
    #         for b, (s, e) in enumerate(zip(start_indices, end_indices), start=1):
    #             batch_mask = np.zeros_like(threshold_headers).astype(np.bool)
    #             batch_mask[s:e] = True
    #             save_files_as_multiframe_ct_pd_cine_dicom(
    #                 threshold_files[batch_mask],
    #                 threshold_headers[batch_mask],
    #                 os.path.join(parent_dir, f"Tube{tube}_Thr{threshold}"),
    #                 f"Batch_{b:05d}_multiframe_ct_pd.dcm",
    #             )

    # input_dirs.sort()
    # output_dirs = [f.replace("_Single_", "_Multi_") for f in input_dirs]

    # print(f"Found {len(input_dirs)} batch folders under:\n  {parent_dir}\n")

    # for input_dir, output_dir in zip(input_dirs, output_dirs):
    #     batch_name = os.path.basename(input_dir)
    #     save_folder_as_multiframe_ct_pd_cine_dicom(
    #         folder_path=input_dir,
    #         output_dir=output_dir,
    #         output_name=f"{batch_name}_multiframe_ct_pd.dcm",
    #     )

    print("\n=== All batches finished ===")

# ----------------------------------------------------------------------
# if __name__ == "__main__":
#    save_folder_as_multiframe_ct_pd_cine_dicom(r'F:\DICOMCTPD_Generator_PCD\Batch_00001')
