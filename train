from DicomRTTool.ReaderWriter import DicomReaderWriter, ROIAssociationClass
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np


def display_slices(image, mask, skip=1):
    slice_locations = np.unique(np.where(mask != 0)[0])  # get indexes for where there is a contour present
    slice_start = slice_locations[0]  # first slice of contour
    slice_end = slice_locations[len(slice_locations) - 1]  # last slice of contour

    counter = 1

    for img_arr, contour_arr in zip(image[slice_start:slice_end + 1],
                                    mask[slice_start:slice_end + 1]):  # plot the slices with contours overlayed ontop
        if counter % skip == 0:  # if current slice is divisible by desired skip amount
            masked_contour_arr = np.ma.masked_where(contour_arr == 0, contour_arr)
            plt.imshow(img_arr, cmap='gray', interpolation='none')
            plt.imshow(masked_contour_arr, cmap='cool', interpolation='none', alpha=0.5, vmin=1, vmax=np.amax(
                mask))  # vmax is set as total number of contours so same colors can be displayed for each slice
            plt.show()
        counter += 1


# In[15]:


Dicom_path = r'/home/zzt8010/zzy_CT_test/data/train'
Dicom_reader = DicomReaderWriter(description='Examples', arg_max=True)
Dicom_reader.walk_through_folders(Dicom_path)  # This will parse through all DICOM present in the folder and subfolders
all_rois = Dicom_reader.return_rois(print_rois=True)  # Return a list of all rois present

# In[16]:

Contour_names = ['whole lung']
Dicom_reader.set_contour_names_and_associations(contour_names=Contour_names)

# In[17]:

indexes = Dicom_reader.which_indexes_have_all_rois()

# In[18]:

# Original data
data = [
    (0,  "/home/zzt8010/zzy_CT_test/data/train/pos/P0971"),
    (1,  "/home/zzt8010/zzy_CT_test/data/train/pos/P1087"),
    (2,  "/home/zzt8010/zzy_CT_test/data/train/pos/P1097"),
    (3,  "/home/zzt8010/zzy_CT_test/data/train/neg/P1104"),
    (4,  "/home/zzt8010/zzy_CT_test/data/train/neg/P1118"),
    (5,  "/home/zzt8010/zzy_CT_test/data/train/neg/P1253"),
    (8,  "/home/zzt8010/zzy_CT_test/data/train/neg/P1254"),
    (9,  "/home/zzt8010/zzy_CT_test/data/train/pos/P0874"),
    (10, "/home/zzt8010/zzy_CT_test/data/train/neg/P0805"),
    (44, "/home/zzt8010/zzy_CT_test/data/train/neg/P0198"),
    (46, "/home/zzt8010/zzy_CT_test/data/train/neg/P0202"),
    (51, "/home/zzt8010/zzy_CT_test/data/train/neg/P0210"),
    (57, "/home/zzt8010/zzy_CT_test/data/train/neg/P0504"),
    (75, "/home/zzt8010/zzy_CT_test/data/train/neg/P0029"),
    (76, "/home/zzt8010/zzy_CT_test/data/train/pos/P0054"),
    (77, "/home/zzt8010/zzy_CT_test/data/train/neg/P0364"),
    (78, "/home/zzt8010/zzy_CT_test/data/train/neg/P0003"),
    (88, "/home/zzt8010/zzy_CT_test/data/train/neg/P0047"),
    (90, "/home/zzt8010/zzy_CT_test/data/train/pos/P0149"),
    (91, "/home/zzt8010/zzy_CT_test/data/train/neg/P0257"),
    (92, "/home/zzt8010/zzy_CT_test/data/train/neg/P0222"),
    (93, "/home/zzt8010/zzy_CT_test/data/train/neg/P0260"),
    (95, "/home/zzt8010/zzy_CT_test/data/train/neg/P0138"),
    (96, "/home/zzt8010/zzy_CT_test/data/train/neg/P0158"),
    (97, "/home/zzt8010/zzy_CT_test/data/train/neg/P0262"),
    (98, "/home/zzt8010/zzy_CT_test/data/train/neg/P0241"),
    (99, "/home/zzt8010/zzy_CT_test/data/train/neg/P0140"),
    (123,"/home/zzt8010/zzy_CT_test/data/train/neg/P0123"),
    (131,"/home/zzt8010/zzy_CT_test/data/train/neg/P0450"),
    (132,"/home/zzt8010/zzy_CT_test/data/train/neg/P0677"),
    (138,"/home/zzt8010/zzy_CT_test/data/train/neg/P0393"),
    (141,"/home/zzt8010/zzy_CT_test/data/train/neg/P0672"),
    (147,"/home/zzt8010/zzy_CT_test/data/train/neg/P0621"),
    (156,"/home/zzt8010/zzy_CT_test/data/train/pos/P0581"),
    (160,"/home/zzt8010/zzy_CT_test/data/train/neg/P0573"),
    (161,"/home/zzt8010/zzy_CT_test/data/train/neg/P0570"),
    (172,"/home/zzt8010/zzy_CT_test/data/train/neg/P0571"),
    (178,"/home/zzt8010/zzy_CT_test/data/train/neg/P0554"),
    (182,"/home/zzt8010/zzy_CT_test/data/train/neg/P0506"),
    (189,"/home/zzt8010/zzy_CT_test/data/train/neg/P0589"),
    (191,"/home/zzt8010/zzy_CT_test/data/train/neg/P0517"),
    (192,"/home/zzt8010/zzy_CT_test/data/train/pos/P0695"),
    (196,"/home/zzt8010/zzy_CT_test/data/train/pos/P0500"),
    (198,"/home/zzt8010/zzy_CT_test/data/train/pos/P0472"),
    (199,"/home/zzt8010/zzy_CT_test/data/train/neg/P0444"),
    (200,"/home/zzt8010/zzy_CT_test/data/train/pos/P0704"),
    (202,"/home/zzt8010/zzy_CT_test/data/train/neg/P0643"),
    (210,"/home/zzt8010/zzy_CT_test/data/train/neg/P0612"),
    (213,"/home/zzt8010/zzy_CT_test/data/train/pos/P0468"),
    (254,"/home/zzt8010/zzy_CT_test/data/train/pos/P0567"),
    (257,"/home/zzt8010/zzy_CT_test/data/train/pos/P0636"),
    (258,"/home/zzt8010/zzy_CT_test/data/train/neg/P0640"),
    (261,"/home/zzt8010/zzy_CT_test/data/train/neg/P0681"),
    (262,"/home/zzt8010/zzy_CT_test/data/train/neg/P0715"),
    (268,"/home/zzt8010/zzy_CT_test/data/train/neg/P0741"),
    (269,"/home/zzt8010/zzy_CT_test/data/train/neg/P0742"),
    (270,"/home/zzt8010/zzy_CT_test/data/train/neg/P0745"),
    (271,"/home/zzt8010/zzy_CT_test/data/train/pos/P0751"),
    (272,"/home/zzt8010/zzy_CT_test/data/train/pos/P0753"),
    (274,"/home/zzt8010/zzy_CT_test/data/train/pos/P0784"),
    (275,"/home/zzt8010/zzy_CT_test/data/train/neg/P0820"),
    (276,"/home/zzt8010/zzy_CT_test/data/train/neg/P0827"),
    (277,"/home/zzt8010/zzy_CT_test/data/train/neg/P0830"),
    (278,"/home/zzt8010/zzy_CT_test/data/train/neg/P0834"),
    (279,"/home/zzt8010/zzy_CT_test/data/train/neg/P0835"),
    (280,"/home/zzt8010/zzy_CT_test/data/train/neg/P0842"),
    (281,"/home/zzt8010/zzy_CT_test/data/train/neg/P0859"),
    (282,"/home/zzt8010/zzy_CT_test/data/train/pos/P0862"),
    (283,"/home/zzt8010/zzy_CT_test/data/train/neg/P0865"),
    (284,"/home/zzt8010/zzy_CT_test/data/train/neg/P0877"),
    (285,"/home/zzt8010/zzy_CT_test/data/train/neg/P0885"),
    (286,"/home/zzt8010/zzy_CT_test/data/train/pos/P0887"),
    (287,"/home/zzt8010/zzy_CT_test/data/train/pos/P0889"),
    (328,"/home/zzt8010/zzy_CT_test/data/train/neg/P0948"),
    (329,"/home/zzt8010/zzy_CT_test/data/train/neg/P0959"),
    (331,"/home/zzt8010/zzy_CT_test/data/train/neg/P0964"),
    (333,"/home/zzt8010/zzy_CT_test/data/train/neg/P0983"),
    (334,"/home/zzt8010/zzy_CT_test/data/train/pos/P0985"),
    (363,"/home/zzt8010/zzy_CT_test/data/train/neg/P0992"),
    (365,"/home/zzt8010/zzy_CT_test/data/train/pos/P1012"),
    (366,"/home/zzt8010/zzy_CT_test/data/train/pos/P1016"),
    (367,"/home/zzt8010/zzy_CT_test/data/train/neg/P1019"),
    (368,"/home/zzt8010/zzy_CT_test/data/train/pos/P1020"),
    (369,"/home/zzt8010/zzy_CT_test/data/train/neg/P1031"),
    (371,"/home/zzt8010/zzy_CT_test/data/train/neg/P1037"),
    (375,"/home/zzt8010/zzy_CT_test/data/train/neg/P1055"),
    (377,"/home/zzt8010/zzy_CT_test/data/train/pos/P1060"),
    (378,"/home/zzt8010/zzy_CT_test/data/train/pos/P1064"),
    (380,"/home/zzt8010/zzy_CT_test/data/train/pos/P1069"),
    (381,"/home/zzt8010/zzy_CT_test/data/train/neg/P1070"),
    (382,"/home/zzt8010/zzy_CT_test/data/train/neg/P1072"),
    (383,"/home/zzt8010/zzy_CT_test/data/train/neg/P1073"),
    (384,"/home/zzt8010/zzy_CT_test/data/train/neg/P1080"),
    (385,"/home/zzt8010/zzy_CT_test/data/train/pos/P1084"),
    (388,"/home/zzt8010/zzy_CT_test/data/train/neg/P1107"),
    (389,"/home/zzt8010/zzy_CT_test/data/train/neg/P1108"),
    (390,"/home/zzt8010/zzy_CT_test/data/train/neg/P1117"),
    (391,"/home/zzt8010/zzy_CT_test/data/train/neg/P1121"),
    (394,"/home/zzt8010/zzy_CT_test/data/train/neg/P1132"),
    (498,"/home/zzt8010/zzy_CT_test/data/train/pos/P1161"),
    (499,"/home/zzt8010/zzy_CT_test/data/train/pos/P1163"),
    (500,"/home/zzt8010/zzy_CT_test/data/train/neg/P1177"),
    (501,"/home/zzt8010/zzy_CT_test/data/train/pos/P1181"),
    (502,"/home/zzt8010/zzy_CT_test/data/train/neg/P1182"),
    (504,"/home/zzt8010/zzy_CT_test/data/train/neg/P1190"),
    (511,"/home/zzt8010/zzy_CT_test/data/train/neg/P1230"),
    (512,"/home/zzt8010/zzy_CT_test/data/train/neg/P1238"),
    (514,"/home/zzt8010/zzy_CT_test/data/train/neg/P1251"),
    (517,"/home/zzt8010/zzy_CT_test/data/train/neg/P1269"),
    (518,"/home/zzt8010/zzy_CT_test/data/train/neg/P1270"),
    (519,"/home/zzt8010/zzy_CT_test/data/train/neg/P1277"),
    (520,"/home/zzt8010/zzy_CT_test/data/train/neg/P1287"),
    (521,"/home/zzt8010/zzy_CT_test/data/train/neg/P1290"),
    (620,"/home/zzt8010/zzy_CT_test/data/train/neg/P1292"),
    (621,"/home/zzt8010/zzy_CT_test/data/train/neg/P1307"),
    (625,"/home/zzt8010/zzy_CT_test/data/train/neg/P1308"),
    (650,"/home/zzt8010/zzy_CT_test/data/train/neg/P0312"),
    (651,"/home/zzt8010/zzy_CT_test/data/train/pos/P0356"),
    (656,"/home/zzt8010/zzy_CT_test/data/train/neg/P0657"),
]


# Function to extract label from path
def get_label(path):
    label = path.split('/')[1]
    return 1 if label == 'pos' else 0


# New data list
new_data = []

# Iterate over the original data
for item in data:
    index, path = item
    new_item = (index, get_label(path))
    new_data.append(new_item)

# Print the transformed data
for item in new_data:
    print(f"Index {item[0]}, labeled as {item[1]}")

# In[19]:

new_data[0][0]

# Dictionary to store patient index, images, masks and labels
patient_data = {}

for patient in new_data:
    pt_indx, label = patient[0], patient[1]
    try:
        Dicom_reader.set_index(pt_indx)
        Dicom_reader.get_images_and_mask()

        # Directly assign image and mask data from Dicom_reader to dictionary
        patient_data[pt_indx] = {
            'label': label,
            'image': Dicom_reader.ArrayDicom,  # Image array
            'mask': Dicom_reader.mask,  # Mask array
        }
    except TypeError:
        print(f"An error occurred while processing patient index {pt_indx}. Skipping...")
        continue

# In[24]:

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

"""## initialize DicomRTTool"""
import pandas as pd
from torch.utils.data import Dataset

#

# In[25]:

import torch
from torch.utils.data import Dataset
import pickle

# class DicomDataset(Dataset):
#     def __init__(self, patient_data):
#         self.patient_data = patient_data
#         self.keys = list(patient_data.keys())

#     def __len__(self):
#         return len(self.patient_data)

#     def __getitem__(self, idx):
#         patient = self.patient_data[self.keys[idx]]
#         try:
#             image = patient['image']
#             # Normalize the image slice
#             image = image.astype(float)
#             image /= image.max()

#             mask = patient['mask']
#             # Convert mask into binary format (0 for background, 1 for lungs)
#             mask = (mask > 0).astype(int)

#             image_tensor = torch.from_numpy(image)
#             mask_tensor = torch.from_numpy(mask)
#             label = patient['label']

#             return image_tensor, mask_tensor, label
#         except KeyError:
#             print(f"An error occurred while processing patient index {self.keys[idx]}. Skipping...")
#             return None

# # Create Dataset
# dataset = DicomDataset(patient_data)

# # Save dataset as pickle
# with open('dicom_dataset.pkl', 'wb') as f:
#     pickle.dump(dataset, f)

# # Create DataLoader
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# class CNNClassifier(nn.Module):
#     def __init__(self):
#         super(CNNClassifier, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 128 * 128, 512)
#         self.fc2 = nn.Linear(512, 2)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 128 * 128)
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# import pickle

# # Load the dataset
# with open('dicom_dataset.pkl', 'rb') as f:
#     dataset = pickle.load(f)

# # Split dataset into training and validation
# train_size = int(0.8 * len(dataset)) # 80% for training
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# # Create DataLoader
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# # Check if a GPU is available and if not, default to CPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Training on {device}")

# # Initialize the model, loss function, and optimizer
# model = CNNClassifier().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Now you can use the model in your training loop
# n_epochs = 10
# for epoch in range(n_epochs):
#     model.train()
#     running_loss = 0.0
#     for i, (inputs, masks, labels) in enumerate(train_loader):
#         # Move data and labels to device
#         inputs, labels = inputs.to(device), labels.to(device)
#         mask = masks.to(device)
#         inputs = masks.unsqueeze(1).float()
#         optimizer.zero_grad()
#         outputs = model(inputs).float()
#         labels = labels.long()
#         loss = criterion(outputs, labels)

#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {running_loss / len(train_loader)}")
#         # Validate
#     model.eval()
#     running_val_loss = 0.0
#     with torch.no_grad():
#         for i, (inputs, masks, labels) in enumerate(val_loader):
#             inputs, labels = inputs.to(device), labels.to(device)
#             mask = masks.to(device)
#             inputs = masks.unsqueeze(1).float()
#             outputs = model(inputs).float()
#             labels = labels.long()
#             val_loss = criterion(outputs, labels)
#             running_val_loss += val_loss.item()
#     print(f"Val - Epoch {epoch + 1}/{n_epochs}, Loss: {running_val_loss / len(val_loader)}")
# # Save the model
# model_save_path = "./cnn_model.pth"
# torch.save(model.state_dict(), model_save_path)
# print(f"Model saved to {model_save_path}")
import torch.nn.functional as F


class DicomDataset(Dataset):
    def __init__(self, patient_data, desired_num_slices=200):
        self.patient_data = patient_data
        self.keys = list(patient_data.keys())
        self.desired_num_slices = desired_num_slices

    def __len__(self):
        return len(self.patient_data)

    def pad_tensor(self, data):
        # Convert the numpy array to a torch tensor
        data = torch.from_numpy(data)

        # Calculate padding
        diff = self.desired_num_slices - data.shape[0]
        # Check if we need padding
        if diff > 0:
            # Apply padding
            data = F.pad(data, (0, 0, 0, 0, 0, diff))
        else:
            # Truncate
            data = data[:self.desired_num_slices]

        return data

    def __getitem__(self, idx):
        patient = self.patient_data[self.keys[idx]]
        try:
            image = patient['image']
            # Normalize the image slice
            image = image.astype(float)
            image /= image.max()

            mask = patient['mask']
            # Convert mask into binary format (0 for background, 1 for lungs)
            mask = (mask > 0).astype(int)

            # Apply padding or truncation
            image = self.pad_tensor(image)
            mask = self.pad_tensor(mask)

            label = patient['label']

            return image, mask, label
        except KeyError:
            print(f"An error occurred while processing patient index {self.keys[idx]}. Skipping...")
            return None


# Create Dataset
dataset = DicomDataset(patient_data)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 128 * 128, 512)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 128 * 128)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Split dataset into training and validation
train_size = int(0.7 * len(dataset))  # 80% for training
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Check if a GPU is available and if not, default to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

# Initialize the model, loss function, and optimizer
model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Now you can use the model in your training loop
n_epochs = 10
import matplotlib.pyplot as plt

# Initialize lists to save the losses
train_losses = []
val_losses = []

n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, masks, labels) in enumerate(train_loader):
        # Move data and labels to device
        inputs, labels = inputs.to(device), labels.to(device)
        mask = masks.to(device)
        inputs = masks.unsqueeze(1).float()
        optimizer.zero_grad()
        outputs = model(inputs).float()
        labels = labels.long()
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Save the training loss for this epoch
    train_losses.append(running_loss / len(train_loader))

    print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {running_loss / len(train_loader)}")

    # Validate
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for i, (inputs, masks, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            mask = masks.to(device)
            inputs = masks.unsqueeze(1).float()
            outputs = model(inputs).float()
            labels = labels.long()
            val_loss = criterion(outputs, labels)
            running_val_loss += val_loss.item()

    # Save the validation loss for this epoch
    val_losses.append(running_val_loss / len(val_loader))

    print(f"Epoch {epoch + 1}/{n_epochs}, Val Loss: {running_val_loss / len(val_loader)}")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
pl# Save the model
model_save_path = "./cnn_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# test 
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import numpy as np

model.eval()  # Set the model to evaluation mode

true_labels = []
pred_labels = []
outputs_list = []

# Loop through the test data
for inputs, masks, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    masks = masks.to(device)
    inputs = masks.unsqueeze(1).float()

    # Forward pass
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)  # Get the predicted classes

    true_labels.extend(labels.cpu().numpy())
    pred_labels.extend(preds.cpu().numpy())
    outputs_list.extend(outputs.detach().cpu().numpy())

# Convert to numpy arrays for use with sklearn
true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

# Compute ROC AUC
roc_auc = roc_auc_score(label_binarize(true_labels, classes=[0,1]),
                        label_binarize(pred_labels, classes=[0,1]), 
                        average='macro')

# Compute accuracy
accuracy = accuracy_score(true_labels, pred_labels)

# Compute confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# Plot confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print(f"Accuracy: {accuracy}")
print(f"ROC AUC: {roc_auc}")
