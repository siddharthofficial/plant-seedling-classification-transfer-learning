import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import cv2
import numpy as np

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Plant Classifier 🌱",
    page_icon="🌿",
    layout="centered"
)

st.title("🌱 Plant Seedling Classifier")
st.write("Upload a plant image and the model will predict its species.")

# -------------------------------
# DEVICE
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# LOAD CLASS NAMES
# -------------------------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

idx_to_class = {i: c for i, c in enumerate(class_names)}
NUM_CLASSES = len(class_names)

# -------------------------------
# IMAGE TRANSFORM 
# -------------------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# MODEL ARCHITECTURE
# -------------------------------
def load_resnet50(num_classes):

    model = models.resnet50(weights=None)

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )

    return model


# -------------------------------
# LOAD MODEL (CACHED)
# -------------------------------
@st.cache_resource
def load_model():

    model = load_resnet50(NUM_CLASSES)

    checkpoint = torch.load(
        "resnet50_best.pth",
        map_location=device
    )

    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):

        self.model.zero_grad()

        output = self.model(input_tensor)
        score = output[:, class_idx]

        score.backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cam / cam.max()

        return cam


model = load_model()
gradcam = GradCAM(model, model.layer4[-1])

def overlay_gradcam(image, cam):

    image_np = np.array(image)

    cam = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    overlay = heatmap * 0.4 + image_np
    overlay = np.clip(overlay / overlay.max() * 255, 0, 255).astype(np.uint8)

    return overlay

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict(image):

    input_tensor = test_transform(image).unsqueeze(0).to(device)

    with torch.enable_grad():

        outputs = model(input_tensor)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

        cam = gradcam.generate(input_tensor, pred.item())

    predicted_class = idx_to_class[pred.item()]
    confidence = conf.item()

    heatmap = overlay_gradcam(image, cam)

    return predicted_class, confidence, heatmap
# -------------------------------
# FILE UPLOADER
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload Plant Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# DISPLAY + PREDICT
# -------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width="stretch")

    if st.button("Predict 🌿"):

        with st.spinner("Predicting..."):

            label, confidence, heatmap = predict(image)

        st.success(f"Prediction: **{label}**")
        st.write(f"Confidence: **{confidence:.2%}**")

        st.subheader("🔍 Model Attention (Grad-CAM)")
        st.image(heatmap, caption="Grad-CAM Visualization", width="stretch")
