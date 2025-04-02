import pydicom
import os
import numpy as np

def extract_patient_details_numpy_fast(dicom_folder, max_files=18001):
    """
    Extracts patient details (Sex, Age, Size, Weight) from DICOM files 
    and directly stores them in a preallocated NumPy array for speed.
    
    :param dicom_folder: Path to folder containing DICOM files.
    :param max_files: Estimated max number of files (to preallocate array).
    :return: NumPy array with extracted details.
    """

    # Preallocate a fixed-size NumPy array (assuming max_files limit)
    records_np = np.empty((max_files, 5), dtype=object)  # 5 columns (Filename, Sex, Age, Size, Weight)
    
    count = 0  # Track actual number of files processed

    # Iterate over DICOM files in the folder
    for filename in os.listdir(dicom_folder):
        dicom_path = os.path.join(dicom_folder, filename)

        try:
            dicom_data = pydicom.dcmread(dicom_path)

            # Extract details, using np.nan for missing values
            patient_sex = dicom_data.PatientSex if 'PatientSex' in dicom_data else np.nan
            patient_age = dicom_data.PatientAge if 'PatientAge' in dicom_data else np.nan
            patient_size = dicom_data.PatientSize if 'PatientSize' in dicom_data else np.nan
            patient_weight = dicom_data.PatientWeight if 'PatientWeight' in dicom_data else np.nan

            # Store directly in preallocated NumPy array
            records_np[count] = [filename, patient_sex, patient_age, patient_size, patient_weight]
            count += 1

            # Stop if max_files limit is reached (for efficiency)
            if count >= max_files:
                break

        except Exception as e:
            print(f"Skipping {filename} (Error: {e})")

    # Trim excess empty rows
    records_np = records_np[:count]

    return records_np

dicom_folder = "../../mnt/shared_dataset/physionet.org/files/vindr-cxr/1.0.0/train"
max_files = 100
numpy_filename = "full_patient_details.npy"
columns = ["Filename", "Sex", "Age", "Size", "Weight"]

patient_info_np = extract_patient_details_numpy_fast(dicom_folder, max_files=max_files)

#Save as NumPy binary file for fast reloading
np.save(numpy_filename, patient_info_np)
print(f"Extracted patient details saved as NumPy binary file: {numpy_filename}")