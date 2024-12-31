import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import torch
import torch.optim as optim
import torch.nn as nn  # Import nn module for loss functions and neural network layers
import torch.nn.functional as F
import pickle  # Import pickle for saving the model
from tqdm import tqdm
from data_loader import get_data_loaders
from model import resnet50
from config import DATA_DIR, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE
import boto3

# Initialize S3 client
s3 = boto3.client('s3')

#torch.amp.autocast("cuda", enabled=True, dtype=torch.float16)

def save_checkpoint(state, filename='checkpoint.pth'):
    """Save the model checkpoint."""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

    # Define file and S3 details
    local_file_path = filename  # Local file to upload
    bucket_name = 'imagenethouse'       # S3 bucket name
    object_key = 'checkpoint-30-12-2024-11-24'  # S3 object key

    # Upload the file
    try:
        s3.upload_file(local_file_path, bucket_name, object_key)
        print(f"File '{local_file_path}' successfully uploaded to 's3://{bucket_name}/{object_key}'.")
    except Exception as e:
        print(f"Error uploading file: {e}")

def load_checkpoint(model, optimizer, scheduler, filename='checkpoint.pth'):
    """Load the model checkpoint."""
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Check if the scheduler state dict is present
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # Load scheduler state
        else:
            print("Warning: 'scheduler_state_dict' not found in checkpoint. Skipping scheduler loading.")
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded (epoch {epoch}, loss {loss:.4f})")
        return epoch, loss
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0, float('inf')  # Return epoch 0 and infinite loss if no checkpoint found

def save_model(model, filename='model.pt'):
    """Save the model in .pt format."""
    torch.save(model.state_dict(), filename)
    print(f"Model saved in .pt format to {filename}")

def save_model_pickle(model, filename='model.pkl'):
    """Save the model in pickle format."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved in pickle format to {filename}")

def top1_accuracy(outputs, labels):
    """Calculate top-1 accuracy."""
    _, preds = outputs.max(1)  # Get the index of the max log-probability
    correct = preds.eq(labels).sum().item()  # Count correct predictions
    return correct

def train_model(data_dir=DATA_DIR, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, checkpoint_path='checkpoint.pth'):
    # Set the device to "mps" if on macOS, otherwise check for CUDA
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")  # Use Metal Performance Shaders on macOS
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU if available
    else:
        device = torch.device("cpu")  # Fallback to CPU
    
    # Get data loaders for training and validation
    train_loader, val_loader = get_data_loaders(data_dir, batch_size)

    model = resnet50().to(device)  # Instantiate the model and move it to the selected device

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)  # Wrap the model with DataParallel
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)  # SGD optimizer with momentum
    criterion = nn.CrossEntropyLoss()  # Loss function

    # Initialize ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Load checkpoint if it exists
    start_epoch, _ = load_checkpoint(model, optimizer, scheduler, checkpoint_path)

    # Initialize GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0  # Initialize loss
        correct = 0  # Initialize correct predictions
        total = 0  # Initialize total predictions

        # Training loop
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the selected device
            optimizer.zero_grad()  # Zero the gradients

            # Mixed precision forward pass
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Calculate loss

            # Backpropagation with mixed precision
            scaler.scale(loss).backward()  # Scale the loss and backpropagate
            scaler.step(optimizer)  # Update weights
            scaler.update()  # Update the scale for next iteration

            running_loss += loss.item()  # Accumulate loss
            correct += top1_accuracy(outputs, labels)  # Update correct predictions
            total += labels.size(0)  # Update total predictions

        train_loss = running_loss / len(train_loader)  # Average training loss
        train_accuracy = (correct / total) * 100  # Training accuracy in percentage

        # Validation
        model.eval()  # Set the model to evaluation mode
        val_correct = 0  # Initialize correct predictions for validation
        val_total = 0  # Initialize total predictions for validation
        val_running_loss = 0.0  # Initialize validation loss

        with torch.no_grad():  # Disable gradient calculation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to the selected device
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)  # Forward pass
                    loss = criterion(outputs, labels)  # Calculate loss
                val_running_loss += loss.item()  # Accumulate validation loss
                val_correct += top1_accuracy(outputs, labels)  # Update correct predictions
                val_total += labels.size(0)  # Update total predictions

        val_loss = val_running_loss / len(val_loader)  # Average validation loss
        val_accuracy = (val_correct / val_total) * 100  # Validation accuracy in percentage

        # Print training and validation statistics
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        # Save checkpoint after each epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
            'loss': train_loss,
        }, checkpoint_path)

    # Save the model in .pt format after training
    save_model(model, filename='model.pt')

    # Save the model in pickle format after training
    save_model_pickle(model, filename='model.pkl')

if __name__ == "__main__":
    train_model()  # Start training with default configuration 

