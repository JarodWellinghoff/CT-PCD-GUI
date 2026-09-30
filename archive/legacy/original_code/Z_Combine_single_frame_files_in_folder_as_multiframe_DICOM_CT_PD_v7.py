import os
import glob
import numpy as np
from pydicom import dcmread, dcmwrite, Dataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from pydicom.tag import Tag


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
    # metadata_list = [
    #     dcmread(os.path.join(folder_path, fname), stop_before_pixels=True)
    #     for fname in files
    # ]
    # metadata_list.sort(key=lambda x: int(x.InstanceNumber))
    file_datasets = []
    for fname in files:
        path = os.path.join(folder_path, fname)
        ds = dcmread(path, force=True)
        instance_num = int(getattr(ds, "InstanceNumber", len(file_datasets) + 1))
        file_datasets.append((instance_num, ds, fname))

    file_datasets.sort(key=lambda x: x[0])
    sorted_ds = [item[1] for item in file_datasets]
    num_frames = len(sorted_ds)
    print(f"  Found {num_frames} files (sorted by InstanceNumber)")

    # ------------------------------------------------------------------
    # 2. Create template from first file
    # ------------------------------------------------------------------
    template = sorted_ds[0].copy()

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
    varying_public_tags = [
        Tag(0x0028, 0x0106),  # SmallestImagePixelValue
        Tag(0x0028, 0x0107),  # LargestImagePixelValue
    ]

    # Remove varying tags from the root dataset
    for t in varying_private_tags + varying_public_tags:
        if t in template:
            del template[t]

    # ------------------------------------------------------------------
    # 3. Prepare multi-frame attributes
    # ------------------------------------------------------------------
    rows = int(template.Rows)
    cols = int(template.Columns)

    # Stack all frames into a 3-D array (frames, rows, cols)
    pixel_array = np.zeros((num_frames, rows, cols), dtype=np.uint16)
    for i, ds in enumerate(sorted_ds):
        arr = ds.pixel_array.astype(np.uint16)
        pixel_array[i] = arr

    template.NumberOfFrames = num_frames
    template.PixelData = pixel_array.tobytes()

    # Multi-frame + cine
    template.FrameTime = 1.0 / 30.0
    template.FrameIncrementPointer = [0x0018, 0x1063]
    template.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    template.SOPInstanceUID = generate_uid()

    # ------------------------------------------------------------------
    # 4. Shared Functional Groups
    # ------------------------------------------------------------------
    shared_item = Dataset()
    shared_item.PixelMeasuresSequence = Sequence([Dataset()])
    shared_item.PixelMeasuresSequence[0].PixelSpacing = sorted_ds[0].get(
        "PixelSpacing", [1.0, 1.0]
    )
    shared_item.PixelMeasuresSequence[0].SliceThickness = 1.0
    template.SharedFunctionalGroupsSequence = Sequence([shared_item])

    # ------------------------------------------------------------------
    # 5. Per-Frame Functional Groups
    # ------------------------------------------------------------------
    per_frame_items = []
    for ds in sorted_ds:
        item = Dataset()

        # Standard sequences
        item.PlanePositionSequence = Sequence([Dataset()])
        item.PlanePositionSequence[0].ImagePositionPatient = ds.get(
            "ImagePositionPatient", [0.0, 0.0, 0.0]
        )

        item.PlaneOrientationSequence = Sequence([Dataset()])
        item.PlaneOrientationSequence[0].ImageOrientationPatient = ds.get(
            "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]
        )

        item.PixelMeasuresSequence = Sequence([Dataset()])
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

        # Varying public tags (Smallest / Largest Image Pixel Value)
        for t in varying_public_tags:
            if t in ds:
                item.add_new(t, "US", int(ds[t].value))

        if hasattr(ds, "XrayTubeCurrent"):
            item.add_new(Tag(0x0018, 0x1151), "IS", ds.XrayTubeCurrent)

        per_frame_items.append(item)

    template.PerFrameFunctionalGroupsSequence = Sequence(per_frame_items)

    # ------------------------------------------------------------------
    # 6. Transfer syntax
    # ------------------------------------------------------------------
    template.is_little_endian = True
    template.is_implicit_VR = False
    if hasattr(template, "file_meta"):
        template.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

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
    return output_file


# ----------------------------------------------------------------------
# Main: process all Batch_* subfolders
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parent_dir = r"U:\PUBLIC\DICOM-CTPD\PCD_DICOMCTPD_Generator\Full_dose_CTPD_Single_Frame\Tube1_Thr1\Batch_00001"
    # input_dirs = glob.glob(os.path.join(parent_dir, "**", "*batch_*"))
    # input_dirs.sort()
    input_dirs = [parent_dir]
    input_dirs.sort()
    output_dirs = [f.replace("_Single_", "_Multi_v7_") for f in input_dirs]

    print(f"Found {len(input_dirs)} batch folders under:\n  {parent_dir}\n")

    for input_dir, output_dir in zip(input_dirs, output_dirs):
        batch_name = os.path.basename(input_dir)
        save_folder_as_multiframe_ct_pd_cine_dicom(
            folder_path=input_dir,
            output_dir=output_dir,
            output_name=f"{batch_name}_multiframe_ct_pd.dcm",
        )

    print("\n=== All batches finished ===")

# ----------------------------------------------------------------------
# if __name__ == "__main__":
#    save_folder_as_multiframe_ct_pd_cine_dicom(r'F:\DICOMCTPD_Generator_PCD\Batch_00001')
