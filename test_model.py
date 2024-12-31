import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import pickle
from model import resnet50  # Import the model definition

# Load class index mapping
def load_class_labels(label_path):
    with open(label_path, 'r') as f:
        class_labels = json.load(f)
    return class_labels

# Load the model
def load_model(model_path, device):
    model = resnet50()  # Create an instance of the ResNet-50 model
    state_dict = torch.load(model_path, map_location=device)  # Load the state dictionary

    # Remove "module." prefix if present
    if 'module.' in next(iter(state_dict.keys())):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)  # Load the state dictionary
    model.to(device)  # Move the model to the appropriate device
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess the input image
def preprocess_image(image_path):
    # Define the same transformations used during training
    preprocess = transforms.Compose([
        transforms.Resize(256),  # Resize to 256x256
        transforms.CenterCrop(224),  # Center crop to 224x224
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])
    
    image = Image.open(image_path).convert("RGB")  # Open the image and convert to RGB
    image = preprocess(image)  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Perform inference and get top-5 predictions
def predict_top5(model, image_tensor, device):
    image_tensor = image_tensor.to(device)  # Move the image tensor to the appropriate device
    with torch.no_grad():  # Disable gradient calculation
        output = model(image_tensor)  # Forward pass
        probabilities = torch.nn.functional.softmax(output, dim=1)  # Get probabilities
        top5_probabilities, top5_indices = probabilities.topk(5)  # Get top-5 probabilities and indices
    return top5_indices[0].tolist(), top5_probabilities[0].tolist()  # Return as lists

# Main function to test the model
if __name__ == "__main__":
    model_path = 'model.pt'  # Path to the trained model
    image_path = 'images/cat.jpg'  # Path to the input image
    label_path = 'imagenet_class_index.json'  # Path to the class index mapping

    # Load class labels
    class_labels = load_class_labels(label_path)

    # Determine the device to use
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")  # Use Metal Performance Shaders on macOS
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU if available
    else:
        device = torch.device("cpu")  # Fallback to CPU

    # Load the model
    model = load_model(model_path, device)

    # Preprocess the image
    image_tensor = preprocess_image(image_path)

    # Make predictions
    top5_indices, top5_probabilities = predict_top5(model, image_tensor, device)

    # Print the top-5 predicted class indices and their probabilities
    print("Top-5 Predictions:")
    for i in range(5):
        class_index = top5_indices[i]
        class_label = class_labels.get(str(class_index), "Unknown class")  # Convert index to string
        print(f"Class index: {class_index}, Label: {class_label}, Probability: {top5_probabilities[i]:.4f}")