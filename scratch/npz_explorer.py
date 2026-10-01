import numpy as np

yu_npz_data = None
yu_npz_files = None
with np.load(
    r"V:\Yu_Zitong\code_repositories\Python_lesion_insertion\test\lesion_models\Lesion01.npz"
) as data:
    # 1. View all the available array keys inside the file
    print("Available arrays:", data.files)
    yu_npz_data = data
    yu_npz_files = data.files

my_npz_data = None
my_npz_files = None
with np.load(
    r"U:\PUBLIC\Lesion_library\Liver\Liver_P2S2_Scott_40cases\Lesions_npz\L005A\L005-1-Lesion.npz"
) as data:
    # 1. View all the available array keys inside the file
    print("Available arrays:", data.files)
    my_npz_data = data
    my_npz_files = data.files
print()
