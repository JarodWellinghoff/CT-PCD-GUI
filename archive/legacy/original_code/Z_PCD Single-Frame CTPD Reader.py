import os
import numpy as np
from pydicom import dcmread
from pydicom.tag import Tag
from collections import defaultdict


def load_ctpd_dictionary(dict_path):
    """Load custom DICOM-CT-PD dictionary"""
    tag_to_names = defaultdict(list)
    
    if not os.path.exists(dict_path):
        print(f"Warning: Dictionary file not found: {dict_path}")
        return tag_to_names
    
    with open(dict_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '(' not in line:
                continue
            try:
                parts = line.split()
                tag_str = parts[0]
                name = parts[2]
                g, e = tag_str.strip('()').split(',')
                tag = (int(g, 16), int(e, 16))
                tag_to_names[tag].append(name)
            except Exception:
                continue
    
    print(f"Loaded {len(tag_to_names)} named tags from dictionary.")
    return tag_to_names


def _decode_ctpd_value(value):
    """
    Convert raw private-tag values to readable Python types.
    Handles both string (LO/CS/DS) and binary (US/SS/FL) representations.
    """
    if not isinstance(value, (bytes, bytearray)):
        return value

    # Try string decoding first
    try:
        text = value.decode('ascii', errors='ignore').strip()
        if text and any(c.isprintable() and not c.isspace() for c in text):
            try:
                return int(text)
            except ValueError:
                pass
            try:
                return float(text)
            except ValueError:
                pass
            return text
    except Exception:
        pass

    # Binary integer (US / SS / UL …)
    if len(value) == 2:
        return int.from_bytes(value, byteorder='little', signed=False)
    if len(value) == 4:
        return int.from_bytes(value, byteorder='little', signed=False)

    return value


def DICOMCTPDreader_PCD_SingleFrame(input_file, dictionary_path=None):
    """
    Reader for single-frame PCD DICOM-CT-PD files.
    
    Returns
    -------
    projection2D : np.ndarray
        Shape (Rows, Columns), after Rescale
    header : pydicom.Dataset
        Full DICOM header
    geometry : dict
        Dictionary containing geometry / statistics of this single frame:
            - DetectorFocalCenterAngularPosition
            - DetectorFocalCenterAxialPosition
            - DetectorFocalCenterRadialDistance
            - SourceAngularPositionShift
            - Timestamp
            - PhotonStatistics
    """
    print("=== PCD SINGLE-FRAME DICOM-CT-PD READER ===\n")
    
    tag_to_names = load_ctpd_dictionary(dictionary_path) if dictionary_path else {}
    
    ds = dcmread(input_file, force=True)
    
    # ------------------------------------------------------------------
    # Basic information
    # ------------------------------------------------------------------
    # Handle PatientName that may be a PersonName object / struct
    patient_name = ds.get('PatientName', 'N/A')
    if hasattr(patient_name, 'family_name'):
        patient_name = patient_name.family_name
    elif hasattr(patient_name, 'FamilyName'):
        patient_name = patient_name.FamilyName
    
    print(f"Patient Name         : {patient_name}")
    print(f"Modality             : {ds.get('Modality', 'N/A')}")
    print(f"Manufacturer         : {ds.get('Manufacturer', 'N/A')}")
    print(f"Rows x Columns       : {ds.get('Rows', 'N/A')} x {ds.get('Columns', 'N/A')}")
    print(f"InstanceNumber       : {ds.get('InstanceNumber', 'N/A')}")
    
    # ------------------------------------------------------------------
    # Important CTPD metadata
    # ------------------------------------------------------------------
    print("\n--- Important CTPD Metadata ---")
    
    important_tags = [
        (0x7033, 0x0001),   # DictionaryVersion
        (0x7033, 0x1059),   # TypeOfMECT
        (0x7033, 0x105B),   # NumberofSources
        (0x7033, 0x105D),   # SourceIndex
        (0x7033, 0x1061),   # NumberofSpectra
        (0x7033, 0x1063),   # SpectrumIndex
        (0x7033, 0x1062),   # ListOfSpectrum
        (0x7033, 0x1066),   # PhotonStatistics_mA
    ]
    
    for tag in important_tags:
        if tag in ds:
            elem = ds[tag]
            name_list = tag_to_names.get(tag, [f"Private_{tag[0]:04X}_{tag[1]:04X}"])
            display_name = name_list[-1]
            
            value = _decode_ctpd_value(elem.value)
            
            if isinstance(value, (list, np.ndarray)) and len(value) > 10:
                print(f"{display_name:<35} : [{len(value)} elements]")
            else:
                print(f"{display_name:<35} : {value}")
    
    # ------------------------------------------------------------------
    # Geometry / PhotonStatistics (all in root for single-frame)
    # ------------------------------------------------------------------
    print("\n--- Geometry / PhotonStatistics ---")
    
    geometry = {
        'DetectorFocalCenterAngularPosition': np.nan,
        'DetectorFocalCenterAxialPosition'  : np.nan,
        'DetectorFocalCenterRadialDistance' : np.nan,
        'SourceAngularPositionShift'        : np.nan,
        'Timestamp'                         : np.nan,
        'PhotonStatistics'                  : None,
    }
    
    tags = {
        'DetectorFocalCenterAngularPosition': Tag(0x7031, 0x1001),
        'DetectorFocalCenterAxialPosition'  : Tag(0x7031, 0x1002),
        'DetectorFocalCenterRadialDistance' : Tag(0x7031, 0x1003),
        'SourceAngularPositionShift'        : Tag(0x7033, 0x100B),
        'Timestamp'                         : Tag(0x7033, 0x1067),
        'PhotonStatistics'                  : Tag(0x7033, 0x1065),
    }
    
    # Angular position
    t = tags['DetectorFocalCenterAngularPosition']
    if t in ds:
        val = ds[t].value
        if isinstance(val, (bytes, bytearray)):
            val = np.frombuffer(val, dtype=np.float32)
        geometry['DetectorFocalCenterAngularPosition'] = float(val[0] if isinstance(val, (list, np.ndarray)) else val)
        print(f"DetectorFocalCenterAngularPosition : {geometry['DetectorFocalCenterAngularPosition']}")
    
    # Axial position
    t = tags['DetectorFocalCenterAxialPosition']
    if t in ds:
        val = ds[t].value
        if isinstance(val, (bytes, bytearray)):
            val = np.frombuffer(val, dtype=np.float32)
        geometry['DetectorFocalCenterAxialPosition'] = float(val[0] if isinstance(val, (list, np.ndarray)) else val)
        print(f"DetectorFocalCenterAxialPosition   : {geometry['DetectorFocalCenterAxialPosition']}")
    
    # Radial distance
    t = tags['DetectorFocalCenterRadialDistance']
    if t in ds:
        val = ds[t].value
        if isinstance(val, (bytes, bytearray)):
            val = np.frombuffer(val, dtype=np.float32)
        geometry['DetectorFocalCenterRadialDistance'] = float(val[0] if isinstance(val, (list, np.ndarray)) else val)
        print(f"DetectorFocalCenterRadialDistance  : {geometry['DetectorFocalCenterRadialDistance']}")
    
    # Source angular shift
    t = tags['SourceAngularPositionShift']
    if t in ds:
        val = ds[t].value
        if isinstance(val, (bytes, bytearray)):
            val = np.frombuffer(val, dtype=np.float32)
        geometry['SourceAngularPositionShift'] = float(val[0] if isinstance(val, (list, np.ndarray)) else val)
        print(f"SourceAngularPositionShift         : {geometry['SourceAngularPositionShift']}")
    else:
        print("SourceAngularPositionShift         : NOT FOUND")
    
    # Timestamp
    t = tags['Timestamp']
    if t in ds:
        val = ds[t].value
        if isinstance(val, (bytes, bytearray)):
            val = np.frombuffer(val, dtype=np.float32)
        geometry['Timestamp'] = float(val[0] if isinstance(val, (list, np.ndarray)) else val)
        print(f"Timestamp                          : {geometry['Timestamp']}")
    
    # PhotonStatistics
    t = tags['PhotonStatistics']
    if t in ds:
        val = ds[t].value
        if isinstance(val, (bytes, bytearray)):
            geometry['PhotonStatistics'] = np.frombuffer(val, dtype=np.float32)
        else:
            geometry['PhotonStatistics'] = np.asarray(val, dtype=np.float32)
        print(f"PhotonStatistics                   : [{len(geometry['PhotonStatistics'])} elements]")
    
    # ------------------------------------------------------------------
    # Projection data
    # ------------------------------------------------------------------
    print("\nReading projection data...")
    raw_data = ds.pixel_array.astype(np.float64)
    
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        projection2D = raw_data * slope + intercept
        print(f"Rescale applied      : Slope = {slope:.8f}, Intercept = {intercept:.8f}")
    else:
        projection2D = raw_data
        print("Warning: Rescale tags not found!")
    
    print(f"Final Projection Shape : {projection2D.shape}")
    print("\nReader finished successfully.")
    
    return projection2D, ds, geometry


# ====================== Example Usage ======================
if __name__ == "__main__":
    input_file = r'F:\PCD_DICOMCTPD_Generator\PCD_Batch_00001_restored\frame_00001.dcm'  # single frame CTPD file
    dict_path  = r'F:\PCD_DICOMCTPD_Generator\DICOM-CT-PD-dict_v11.txt'
    
    projection2D, header, geometry = DICOMCTPDreader_PCD_SingleFrame(input_file, dict_path)
    
    # Example access
    print("\n=== Quick access example ===")
    print(f"Angular position : {geometry['DetectorFocalCenterAngularPosition']}")
    print(f"Axial position   : {geometry['DetectorFocalCenterAxialPosition']}")
    print(f"Radial distance  : {geometry['DetectorFocalCenterRadialDistance']}")
    print(f"Timestamp        : {geometry['Timestamp']}")
    if geometry['PhotonStatistics'] is not None:
        print(f"PhotonStatistics length : {len(geometry['PhotonStatistics'])}")
    print(f"Projection shape : {projection2D.shape}")