import os
import numpy as np
from pydicom import dcmread, dcmwrite, Dataset
from pydicom.uid import generate_uid, ImplicitVRLittleEndian
from pydicom.tag import Tag
from copy import deepcopy


def extract_single_frames_from_multiframe(input_file, output_dir):
    """
    Fast version: convert multi-frame DICOM-CT-PD back to single-frame files.

    Restores per-frame private tags, Smallest/Largest, XrayTubeCurrent,
    and InstanceNumber (projection order in the full scan).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load multi-frame file
    # ------------------------------------------------------------------
    ds_multi = dcmread(input_file, force=True)

    if not hasattr(ds_multi, 'NumberOfFrames') or ds_multi.NumberOfFrames is None:
        raise ValueError('Not a multi-frame file.')

    num_frames = int(ds_multi.NumberOfFrames)
    rows = int(ds_multi.Rows)
    cols = int(ds_multi.Columns)

    print(f'Loaded multi-frame file with {num_frames} frames')
    print(f'Dimensions: {rows} × {cols}')

    # ------------------------------------------------------------------
    # 2. Get all frames at once
    # ------------------------------------------------------------------
    pa = ds_multi.pixel_array
    if pa.shape[0] != num_frames:
        if pa.shape[2] == num_frames:
            pa = np.moveaxis(pa, 2, 0)
        else:
            raise ValueError(f'Unexpected pixel_array shape: {pa.shape}')

    print(f'Extracted pixel array shape: {pa.shape}')

    # ------------------------------------------------------------------
    # 3. Collect per-frame tags once
    # ------------------------------------------------------------------
    varying_tags = [
        # private geometry / statistics
        Tag(0x7031, 0x1001),  # DetectorFocalCenterAngularPosition
        Tag(0x7031, 0x1002),  # DetectorFocalCenterAxialPosition
        Tag(0x7031, 0x1003),  # DetectorFocalCenterRadialDistance
        Tag(0x7033, 0x100B),  # SourceAngularPositionShift
        Tag(0x7033, 0x1065),  # PhotonStatistics
        Tag(0x7033, 0x1067),  # Timestamp
        # public tags that also vary per frame
        Tag(0x0020, 0x0013),  # InstanceNumber  ← projection order in full scan
        Tag(0x0028, 0x0106),  # SmallestImagePixelValue
        Tag(0x0028, 0x0107),  # LargestImagePixelValue
    ]

    per_frame_meta = []
    if hasattr(ds_multi, 'PerFrameFunctionalGroupsSequence') and ds_multi.PerFrameFunctionalGroupsSequence:
        for item in ds_multi.PerFrameFunctionalGroupsSequence:
            meta = {
                'private': {},
                'XrayTubeCurrent': None,
                'InstanceNumber': None,
            }
            for t in varying_tags:
                if t in item:
                    if t == Tag(0x0020, 0x0013):
                        # Keep InstanceNumber separate for easy use when writing
                        meta['InstanceNumber'] = int(item[t].value)
                    else:
                        meta['private'][t] = (item[t].VR, item[t].value)
            if Tag(0x0018, 0x1151) in item:
                meta['XrayTubeCurrent'] = item[Tag(0x0018, 0x1151)].value
            per_frame_meta.append(meta)
    else:
        per_frame_meta = [
            {'private': {}, 'XrayTubeCurrent': None, 'InstanceNumber': None}
            for _ in range(num_frames)
        ]

    print(f'Collected metadata for {len(per_frame_meta)} frames')

    # ------------------------------------------------------------------
    # 4. Create a clean template once (remove multi-frame attributes)
    # ------------------------------------------------------------------
    template = Dataset()

    # Copy all elements except multi-frame specific ones and PixelData
    skip_tags = {
        Tag(0x0028, 0x0008),  # NumberOfFrames
        Tag(0x7FE0, 0x0010),  # PixelData
        Tag(0x5200, 0x9229),  # SharedFunctionalGroupsSequence
        Tag(0x5200, 0x9230),  # PerFrameFunctionalGroupsSequence
        Tag(0x0018, 0x1063),  # FrameTime
        Tag(0x0028, 0x0009),  # FrameIncrementPointer
        Tag(0x0020, 0x0013),  # InstanceNumber (restored per-frame)
        Tag(0x0028, 0x0106),  # SmallestImagePixelValue (restored per-frame)
        Tag(0x0028, 0x0107),  # LargestImagePixelValue  (restored per-frame)
    }

    for elem in ds_multi:
        if elem.tag not in skip_tags:
            template.add(elem)

    # Force Implicit VR Little Endian (matches typical MATLAB single-frame CTPD)
    template.is_little_endian = True
    template.is_implicit_VR = True
    if not hasattr(template, 'file_meta'):
        template.file_meta = Dataset()
    template.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    template.file_meta.MediaStorageSOPClassUID = template.get(
        'SOPClassUID', '1.2.840.10008.5.1.4.1.1.2'
    )

    # ------------------------------------------------------------------
    # 5. Write frames (fast)
    # ------------------------------------------------------------------
    output_files = []

    for idx in range(num_frames):
        single = template.copy()  # cheap copy of the clean template

        # Pixel data
        single.PixelData = pa[idx].astype(np.uint16).tobytes()

        # Restore per-frame tags (private + Smallest/Largest)
        meta = per_frame_meta[idx]
        for tag, (vr, value) in meta['private'].items():
            single.add_new(tag, vr, value)

        if meta['XrayTubeCurrent'] is not None:
            single.add_new(Tag(0x0018, 0x1151), 'IS', meta['XrayTubeCurrent'])

        # InstanceNumber: prefer value from multi-frame per-frame item
        # (original global projection order). Fallback to local index only if missing.
        if meta['InstanceNumber'] is not None:
            single.InstanceNumber = meta['InstanceNumber']
        else:
            single.InstanceNumber = idx + 1

        # Unique identifiers for this single-frame file
        single.SOPInstanceUID = generate_uid()
        single.file_meta.MediaStorageSOPInstanceUID = single.SOPInstanceUID

        # File name can follow InstanceNumber for clarity
        out_name = os.path.join(
            output_dir, f'frame_{int(single.InstanceNumber):05d}.dcm'
        )
        dcmwrite(out_name, single, write_like_original=False)
        output_files.append(out_name)

        if (idx + 1) % 200 == 0 or (idx + 1) == num_frames:
            print(f'  Wrote {idx + 1}/{num_frames}')

    print(f'\n✅ Successfully extracted {num_frames} single-frame files to:\n   {output_dir}')
    return output_files


# ----------------------------------------------------------------------
if __name__ == "__main__":
    input_file = r'F:\PCD_DICOMCTPD_Generator\Batch_00001_multiframe_ct_pd.dcm'
    output_dir = r'F:\PCD_DICOMCTPD_Generator\PCD_Batch_00001_restored'

    extract_single_frames_from_multiframe(input_file, output_dir)