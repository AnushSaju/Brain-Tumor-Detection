from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import cv2

# ==========================
# CONFIGURATION
# ==========================
app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
GRADCAM_FOLDER = os.path.join(BASE_DIR, "gradcam")
MODEL_PATH = os.path.join(BASE_DIR, "../outputs/checkpoints/best_model.pth")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/gradcam", StaticFiles(directory=GRADCAM_FOLDER), name="gradcam")

# ==========================
# MODEL SETUP
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ==========================
# TRANSFORMS
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

# ==========================
# GRAD-CAM FUNCTION
# ==========================
def generate_gradcam(image_tensor, model, target_class, output_path):
    model.eval()
    grad = {}
    fmap = {}

    def save_grad(module, grad_in, grad_out):
        grad["value"] = grad_out[0].detach()

    def save_fmap(module, input, output):
        fmap["value"] = output.detach()

    layer = model.layer4[-1]
    layer.register_forward_hook(save_fmap)
    layer.register_backward_hook(save_grad)

    output = model(image_tensor)
    loss = output[0, target_class]
    model.zero_grad()
    loss.backward()

    gradients = grad["value"].cpu().numpy()[0]
    activations = fmap["value"].cpu().numpy()[0]
    weights = np.mean(gradients, axis=(1, 2))
    cam = np.zeros(activations.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activations[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    image_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image_np = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))
    image_np = np.uint8(255 * image_np)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(output_path, superimposed)

# ==========================
# ROUTES
# ==========================
@app.get("/", response_class=HTMLResponse)
def home():
    html = """
    <html>
        <head>
            <title>Brain Tumor Detection</title>
            <style>
                body {
                    background: linear-gradient(to right, #141e30, #243b55);
                    color: white;
                    text-align: center;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                }
                h1 {
                    margin-top: 40px;
                    font-size: 2.5em;
                    color: #00d4ff;
                    text-shadow: 0 0 10px #00d4ff;
                }
                .upload-box {
                    margin: 50px auto;
                    padding: 40px;
                    border: 2px dashed #00d4ff;
                    width: 50%;
                    border-radius: 15px;
                    background-color: rgba(0, 0, 0, 0.2);
                }
                input[type=file], input[type=submit] {
                    margin-top: 20px;
                    padding: 12px;
                    font-size: 1em;
                    border-radius: 8px;
                    border: none;
                }
                input[type=submit] {
                    background-color: #00d4ff;
                    color: black;
                    cursor: pointer;
                    transition: 0.3s;
                }
                input[type=submit]:hover {
                    background-color: #00a6c9;
                }
            </style>
        </head>
        <body>
            <h1> Brain Tumor Detection System</h1>
            <div class='upload-box'>
                <h3>Upload your MRI Scan (JPG / PNG)</h3>
                <form action="/predict" enctype="multipart/form-data" method="post">
                    <input type="file" name="file" accept="image/*" required><br>
                    <input type="submit" value="Predict">
                </form>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(html)


@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(image_path, "wb") as f:
        f.write(await file.read())

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)
        pred_label = "Tumor" if pred_class == 1 else "No Tumor"

    gradcam_path = os.path.join(GRADCAM_FOLDER, f"gradcam_{file.filename}")
    generate_gradcam(tensor, model, pred_class, gradcam_path)

    html = f"""
    <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{
                    background: linear-gradient(to right, #232526, #414345);
                    color: white;
                    text-align: center;
                    font-family: 'Segoe UI', sans-serif;
                    margin: 0;
                    padding: 0;
                }}
                h2 {{
                    margin-top: 40px;
                    font-size: 2em;
                    color: #00d4ff;
                }}
                .container {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    gap: 50px;
                    margin-top: 40px;
                }}
                .image-box {{
                    background-color: rgba(0, 0, 0, 0.4);
                    padding: 20px;
                    border-radius: 10px;
                }}
                img {{
                    border-radius: 10px;
                    width: 300px;
                    height: auto;
                    box-shadow: 0 0 15px #00d4ff;
                }}
                table {{
                    margin: 30px auto;
                    border-collapse: collapse;
                    width: 40%;
                    background-color: rgba(0, 0, 0, 0.5);
                    border-radius: 10px;
                }}
                th, td {{
                    padding: 12px;
                    border: 1px solid #00d4ff;
                    color: white;
                }}
                th {{
                    background-color: #00d4ff;
                    color: black;
                }}
                .footer {{
                    margin-top: 30px;
                    font-size: 0.9em;
                    color: #aaa;
                }}
            </style>
        </head>
        <body>
            <h2>Prediction: {pred_label}</h2>
            <div class="container">
                <div class="image-box">
                    <h3>Original MRI</h3>
                    <img src='/uploads/{file.filename}' alt="Original MRI">
                </div>
                <div class="image-box">
                    <h3>Grad-CAM Visualization</h3>
                    <img src='/gradcam/gradcam_{file.filename}' alt="GradCAM Result">
                </div>
            </div>
            <table>
                <tr><th>Class</th><th>Probability</th></tr>
                <tr><td>No Tumor</td><td>{probs[0]*100:.2f}%</td></tr>
                <tr><td>Tumor</td><td>{probs[1]*100:.2f}%</td></tr>
            </table>
            <div class="footer">
                <p> MRI Tumor Detection Project</p>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(html)
