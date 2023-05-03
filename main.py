import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
# from radiomics import RadiomicsFeatureExtractor
import os

# Read the DICOM file
dcm_file_path = '/Users/zyzou/Desktop/dataset/2.16.840.1.114362.1.11940992.22933840372.569057274.359.6523.dcm'
dcm_data = pydicom.dcmread(dcm_file_path)

# Convert the DICOM data to a SimpleITK Image
ct_image = sitk.GetImageFromArray(dcm_data.pixel_array)
ct_image.SetSpacing((float(dcm_data.PixelSpacing[0]), float(dcm_data.PixelSpacing[1]), float(dcm_data.SliceThickness)))
slice = sitk.GetArrayViewFromImage(ct_image)[50, :, :]
#
# # Plot the slice using matplotlib
plt.imshow(slice, cmap='gray')
plt.show()
# Define the settings for the radiomic feature extraction
settings = {}
settings['binWidth'] = 25
settings['resampledPixelSpacing'] = None
settings['interpolator'] = 'sitkBSpline'
settings['label'] = 1

# Initialize the extractor
# extractor = RadiomicsFeatureExtractor(**settings)

# Extract radiomic features from the CT image
# radiomic_features = extractor.execute(ct_image)

# Print the extracted radiomic features
# for feature_name, feature_value in radiomic_features.items():
#     print(f"{feature_name}: {feature_value}")
