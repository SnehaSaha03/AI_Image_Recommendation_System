from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Load the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Set your image path here
image_path = r"C:\Users\SNEHA SAHA\Pictures\your_image.jpg"  # change this to your actual image path
image = Image.open(image_path)

# List of caption candidates
captions = [
    "A dog running through a field",
    "A plate of delicious pasta",
    "A scenic view of mountains at sunset",
    "A person using a laptop in a cafe",
    "A group of people dancing at a party"
]

# Process inputs for the model
inputs = processor(text=captions, images=image, return_tensors="pt", padding=True)

# Run the model
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # shape: [1, num_captions]
    probs = logits_per_image.softmax(dim=1)      # convert logits to probabilities

# Rank captions by probability
top_probs, top_indices = torch.topk(probs, k=5)

# Print results
print("Top Caption Recommendations:")
for i, idx in enumerate(top_indices[0]):
    print(f"{i+1}. {captions[idx]} (score: {top_probs[0][i].item():.4f})")
