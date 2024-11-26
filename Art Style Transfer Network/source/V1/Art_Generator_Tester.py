import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from StyleTransferNetwork import StyleTransferNetwork, ResidualBlock


# Load and preprocess an image
def load_image(image_path, size=(512, 512)):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

# Load the full model
def load_styled_model(style):
    model_path = f"{style}_style_model.pth"
    if os.path.exists(model_path):
        return torch.load(model_path, map_location=torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        print(f"No model found for style '{style}'.")
        return None

# Display the images using Matplotlib
def display_images(content_image_path):
    content_image = Image.open(content_image_path).convert('RGB')
    content_tensor = load_image(content_image_path).to(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))

    # Set up the plot for displaying the original and stylized images
    model_files = [file for file in os.listdir() if file.endswith("_style_model.pth")]
    num_images = len(model_files) + 1  # +1 for the original image
    fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 4))
    
    axes[0].imshow(content_image.resize((256, 256)))
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Apply each style model and display the result
    for i, model_file in enumerate(model_files):
        style_name = model_file.replace("_style_model.pth", "")
        styled_model = load_styled_model(style_name)
        
        if styled_model is not None:
            with torch.no_grad():
                styled_output = styled_model(content_tensor).cpu().squeeze()
            
            # Convert the tensor back to an image
            styled_image = styled_output.detach().numpy().transpose(1, 2, 0)
            styled_image = (styled_image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
            styled_pil_image = Image.fromarray((styled_image * 255).astype("uint8"))

            axes[i + 1].imshow(styled_pil_image.resize((256, 256)))
            axes[i + 1].set_title(style_name)
            axes[i + 1].axis("off")

    plt.show()

def apply_all_styles():
    content_image_path = input("Enter the path to the content image: ")
    if not os.path.exists(content_image_path):
        print("File not found.")
        return

    display_images(content_image_path)


apply_all_styles()
