import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms


# =====================================================
# LOAD MODEL
# =====================================================

def load_model(model_path, device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False
    )

    model = checkpoint["model"]

    model.to(device)
    model.float()

    for p in model.parameters():
        p.requires_grad = True

    # Grad-CAM requires gradients
    model.train()

    return model, device


# =====================================================
# REGISTER HOOKS
# =====================================================

def register_hooks(model):

    activation = {}
    gradient = {}

    target_layer = model.model[8]

    def forward_hook(module, inp, out):

        activation["value"] = out
        out.retain_grad()

    def backward_hook(module, grad_input, grad_output):

        gradient["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    return (
        activation,
        gradient,
        forward_handle,
        backward_handle
    )


# =====================================================
# IMAGE PREPROCESS
# =====================================================

def preprocess_image(
        image_path,
        size=(224,224)
):

    img = Image.open(
        image_path
    ).convert("RGB")

    img_np = np.array(img)

    img_resize = cv2.resize(
        img_np,
        size
    )

    transform = transforms.ToTensor()

    input_tensor = transform(
        img_resize
    ).unsqueeze(0)

    return img_resize, input_tensor
# =====================================================
# RUN XAI
# =====================================================

def run_xai(
        model,
        device,
        activation,
        gradient,
        image_path,
        save_folder,
        classes,
        method,
        show=True,
        save=True
):

    img_resize, input_tensor = preprocess_image(image_path)

    input_tensor = input_tensor.to(device).float()

    # ==========================================
    # Forward
    # ==========================================

    features = model.model[:9](input_tensor)

    head = model.model[9]

    x = head.conv(features)
    x = head.pool(x)
    x = torch.flatten(x, 1)

    output = head.linear(x)

    # ==========================================
    # Prediction
    # ==========================================

    prob = torch.softmax(output, dim=1)

    pred = torch.argmax(prob, dim=1)

    confidence = prob[
        0,
        pred.item()
    ].item()

    print("-"*40)
    print("Prediction :", classes[pred.item()])
    print("Confidence :", confidence)
    print("-"*40)

    # ==========================================
    # Backward
    # ==========================================

    model.zero_grad()

    score = output[
        0,
        pred.item()
    ]

    score.backward()

    # ==========================================
    # GradCAM
    # ==========================================

    act = activation["value"]
    grad = gradient["value"]
    print("Activation:", act.shape, act.min().item(), act.max().item())
    print("Gradient:", grad.shape, grad.min().item(), grad.max().item())

    if method == "gradcam":
    
        weights = torch.mean(
            grad,
            dim=(2,3),
            keepdim=True
        )
    
    elif method == "gradcam++":
    
        grad2 = grad.pow(2)
        grad3 = grad.pow(3)
    
        sum_act_grad3 = torch.sum(
            act * grad3,
            dim=(2,3),
            keepdim=True
        )
    
        alpha = grad2 / (
            2 * grad2 +
            sum_act_grad3 +
            1e-7
        )
    
        alpha = torch.where(
            torch.isfinite(alpha),
            alpha,
            torch.zeros_like(alpha)
        )
    
        positive_grad = torch.relu(grad)
    
        weights = torch.sum(
            alpha * positive_grad,
            dim=(2,3),
            keepdim=True
        )
    
    cam = torch.sum(
        weights * act,
        dim=1
    )
    
    cam = torch.relu(cam)

    cam = cam.squeeze()

    cam = cam.detach().cpu().numpy()

    # ==========================================
    # Normalize
    # ==========================================
    print("CAM before resize:", cam.min(), cam.max(), cam.mean())
    cam = cv2.resize(
        cam,
        (
            img_resize.shape[1],
            img_resize.shape[0]
        )
    )

    cam = (
        cam - cam.min()
    ) / (
        cam.max() - cam.min() + 1e-8
    )

    # ==============================
    # HEATMAP
    # ==============================
    
    cam_resized = cv2.resize(
        cam,
        (img_resize.shape[1], img_resize.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )
    
    cam_resized = cv2.normalize(
        cam_resized,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)
    
    heatmap = cv2.applyColorMap(
        cam_resized,
        cv2.COLORMAP_JET
    )
    
    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )
    # ==========================================
    # Fish Mask
    # ==========================================

    gray = cv2.cvtColor(
        img_resize,
        cv2.COLOR_RGB2GRAY
    )

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    kernel = np.ones(
        (5,5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    fish_mask = np.zeros_like(gray)

    if len(contours):

        largest = max(
            contours,
            key=cv2.contourArea
        )

        cv2.drawContours(
            fish_mask,
            [largest],
            -1,
            255,
            -1
        )

    else:

        fish_mask[:] = 255

    fish_mask = fish_mask.astype(bool)

    # ==========================================
    # Overlay
    # ==========================================
    
    overlay = img_resize.copy()
    
    
    overlay[fish_mask] = (
        0.5 * img_resize[fish_mask]
        +
        0.5 * heatmap[fish_mask]
    )
    
    
    overlay = np.uint8(overlay)


    # ==========================================
    # Save
    # ==========================================

    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    if save:

        cv2.imwrite(

            os.path.join(
                save_folder,
                f"{image_name}_{method}.jpg"
            ),

            cv2.cvtColor(
                overlay,
                cv2.COLOR_RGB2BGR
            )

        )

        np.save(

            os.path.join(
                save_folder,
                f"{image_name}_{method}.npy"
            ),

            cam.astype(np.float32)

        )

    # ==========================================
    # Show
    # ==========================================

    if show:

        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.imshow(img_resize)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(overlay)
        plt.title(classes[pred.item()])
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return {

        "prediction": pred.item(),

        "label": classes[pred.item()],

        "confidence": confidence,

        "heatmap": cam,

        "overlay": overlay

    }

from glob import glob

MODEL_PATH = r"D:\fish models\YOLO\SalmonScan_YOLOv8n\weights\best.pt"

IMAGE_FOLDER = r"D:\fish models\SalmonScan_Split\val\InfectedFish"

SAVE_FOLDER = r"heatmaps++"
os.makedirs(SAVE_FOLDER, exist_ok=True)

model, device = load_model(MODEL_PATH)

activation, gradient, fh, bh = register_hooks(model)

classes = [
    "FreshFish",
    "InfectedFish"
]

Method = "gradcam++"

# همه تصاویر
image_list = []
for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
    image_list.extend(glob(os.path.join(IMAGE_FOLDER, ext)))

# اجرای XAI روی همه تصاویر
for image_path in image_list:
    print(f"Processing: {os.path.basename(image_path)}")

    run_xai(
        model=model,
        device=device,
        activation=activation,
        gradient=gradient,
        image_path=image_path,
        save_folder=SAVE_FOLDER,
        method=Method,
        classes=classes,
        show=False,
        save=True
    )


fh.remove()
bh.remove()
