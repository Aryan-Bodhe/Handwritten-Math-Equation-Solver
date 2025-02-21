import torch
from torchvision import transforms
from PIL import Image

# Load the trained model
model = torch.load('./model.pth')
model.eval()  # Set to evaluation mode

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Use your model's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess the image
img_path = 'path_to_your_image.jpg'
img = Image.open(img_path)
img = transform(img)
img = img.unsqueeze(0)  # Add batch dimension

# Make the prediction
with torch.no_grad():
    output = model(img)
    _, predicted_class = torch.max(output, 1)

# If you have class labels
class_labels = ['Class1', 'Class2', 'Class3']  # Change this to your class names
print(f"Predicted class: {class_labels[predicted_class.item()]}")
