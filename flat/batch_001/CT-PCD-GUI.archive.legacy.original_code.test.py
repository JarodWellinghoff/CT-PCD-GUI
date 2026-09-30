import pydicom

dcm_v7 = r"U:\PUBLIC\DICOM-CTPD\PCD_DICOMCTPD_Generator\Full_dose_CTPD_Multi_v7_Frame\Tube1_Thr1\Batch_00001\Batch_00001_multiframe_ct_pd.dcm"
dcm_v8 = r"U:\PUBLIC\DICOM-CTPD\PCD_DICOMCTPD_Generator\Full_dose_CTPD_Multi_v8_Frame\Tube1_Thr1\Batch_00001\Batch_00001_multiframe_ct_pd.dcm"
dcm_test = (
    r"U:\PUBLIC\DICOM-CTPD\PCD_DICOMCTPD_Generator\Batch_00001_multiframe_ct_pd.dcm"
)

ds_v7 = pydicom.dcmread(dcm_v7)
ds_v8 = pydicom.dcmread(dcm_v8)
ds_test = pydicom.dcmread(dcm_test)

print("v7")
print(ds_v7.PerFrameFunctionalGroupsSequence[0])
print("v8")
print(ds_v8.PerFrameFunctionalGroupsSequence[0])
print("test")
print(ds_test.PerFrameFunctionalGroupsSequence[0])
