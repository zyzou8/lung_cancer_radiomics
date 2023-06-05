# Split dataset into training and validation
train_size = int(0.7 * len(dataset))  # 80% for training
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

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

wandb.init(project="dicom_classifier", entity="zyzou")
config = dict(
    learning_rate=0.001,
    epochs=10,
    batch_size=16,
)

wandb.config.update(config)

model = CNNClassifier().to(device)
wandb.watch(model, log_freq=100) # log model gradients and parameters


# Initialize lists to save the losses
train_losses = []
val_losses = []

n_epochs = 10
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        # inputs, labels = inputs.to(device), labels.to(device)
        # mask = masks.to(device)
        # inputs = masks.unsqueeze(1).float()

        inputs, masks, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        masks = masks.to(device)
        inputs = masks.unsqueeze(1).float()
        # Iterate over each slice
        for slice_idx in range(inputs.shape[1]):
            # Get the current slice
            curr_slice = inputs[:, slice_idx]
            # Add an extra dimension for the channels
            # curr_slice = curr_slice.unsqueeze(1)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(curr_slice.float())
            loss = criterion(outputs, labels.long())
            
            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    # Save the training loss for this epoch
    train_losses.append(running_loss / len(train_loader))
    print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {running_loss / len(train_loader)}")
    avg_train_loss = running_loss / len(train_loader)
    wandb.log({"Train Loss": avg_train_loss})
    print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {avg_train_loss}")

    # Validate
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            # inputs, labels = inputs.to(device), labels.to(device)
            # mask = masks.to(device)
            # inputs = masks.unsqueeze(1).float()

            inputs, masks, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            masks = masks.to(device)
            inputs = masks.unsqueeze(1).float()

            # Iterate over each slice
            for slice_idx in range(inputs.shape[1]):
                # Get the current slice
                curr_slice = inputs[:, slice_idx]
                # Add an extra dimension for the channels
                # curr_slice = curr_slice.unsqueeze(1)

                # Forward pass
                outputs = model(curr_slice.float())
                val_loss = criterion(outputs, labels.long())

                running_val_loss += val_loss.item()

    # Save the validation loss for this epoch
    val_losses.append(running_val_loss / len(val_loader))

    print(f"Epoch {epoch + 1}/{n_epochs}, Val Loss: {running_val_loss / len(val_loader)}")
    avg_val_loss = running_val_loss / len(val_loader)
    wandb.log({"Val Loss": avg_val_loss})
    print(f"Epoch {epoch + 1}/{n_epochs}, Val Loss: {avg_val_loss}")

    # Plotting
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# Save the model
model_save_path = "./cnn_model.pth"
torch.save(model.state_dict(), model_save_path)
wandb.save(model_save_path)
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
# Log metrics
wandb.log({"Accuracy": accuracy, "ROC AUC": roc_auc})
wandb.finish()
