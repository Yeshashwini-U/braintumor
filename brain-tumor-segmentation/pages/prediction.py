"""
    Contract Study
    Madan Baduwal
"""
import cv2
import math
import streamlit as st
import numpy as np
import torch
from PIL import Image
from unet.predict import predict
from unet import model

model_dict = {'dice loss + cross entropy loss, l_rate = 0.0001': "pretrained1",
              'dice loss, l_rate = 0.0001': "pretrained2",
              'dice loss + cross entropy loss, l_rate = 0.001': "pretrained3",
              'dice loss + cross entropy loss, l_rate = 0.01': "pretrained4",
              'dice loss + cross entropy loss, l_rate = 0.1': "pretrained5"}


# import pretrained model
@st.cache_resource
def load_model(model_name: str):
    unet = model.Unet(3)
    model_params = torch.load(f"data/{model_name}/{model_name}.pth",
                              map_location=torch.device('cpu'))
    unet.load_state_dict(model_params['model'])
    return unet


# model option
st.header("Contract Study: Brain tumor segmentation")
st.markdown("<h5 style='text-align: center;'>Department of Computer Science, University of Texas Permian Basin</h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>baduwal_m63609@utpb.edu</h5>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: left;'>Select a pre-trained model:</h5>", unsafe_allow_html=True)
option = st.selectbox("select a model",
                      ('dice loss + cross entropy loss, l_rate = 0.0001',
                       'dice loss, l_rate = 0.0001',
                       'dice loss + cross entropy loss, l_rate = 0.001',
                       'dice loss + cross entropy loss, l_rate = 0.01',
                       'dice loss + cross entropy loss, l_rate = 0.1'),
                      label_visibility='collapsed')

# file uploader
st.markdown("<h5 style='text-align: left;'>Select a brain MRI image:</h5>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg', 'tif'])

col1, col2 = st.columns(2)
with col1:
    # image cache
    image_cache = st.container()
    if uploaded_file is not None:
        # convert image into np array
        img = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(img)

        # check if the file is valid (3 * 256 * 256)
        if img_array.shape != (256, 256, 3):
            st.write("Image size should be 256*256")
        else:
            # display image
            image_cache.subheader("Your uploaded file:")
            image_cache.image(img_array)

            # store in streamlit session
            st.session_state.img_array = img_array
            img_array = img_array / 255
    elif 'img_array' in st.session_state:
        img_array = st.session_state.img_array
        # display image
        image_cache.subheader("Your uploaded file:")
        image_cache.image(img_array)
    img_pred_button = st.button('Predict')

with col2:
    if img_pred_button:
        if "img_array" not in st.session_state:
            st.write("You haven't uploaded any file!")
        else:
            st.subheader("Model Prediction:")
            
            # Normalize and predict
            pred_img = st.session_state.img_array / 255
            pred_mask = predict(pred_img, torch.device('cpu'),
                                load_model(model_dict[option]))
            pred_mask = pred_mask[0].permute(1, 2, 0).numpy()
            pred_mask_gray = (pred_mask[:, :, 0] * 255).astype(np.uint8)  # Convert to grayscale

            # --- Tumor Classification Logic ---
            _, binary_mask = cv2.threshold(pred_mask_gray, 127, 255, cv2.THRESH_BINARY)
            h_img, w_img = binary_mask.shape
            x, y, w, h = cv2.boundingRect(binary_mask)
            area = cv2.countNonZero(binary_mask)
            aspect_ratio = w / h if h != 0 else 0

            # Compute centroid
            M = cv2.moments(binary_mask)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = -1, -1

            # Tumor classification function
            def classify_tumor(area, aspect_ratio, cx, cy, w_img, h_img):
                if area == 0 or cx == -1 or cy == -1:
                    return "No Tumor"

                center_x, center_y = w_img // 2, h_img // 2
                distance = math.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)

                if 0.85 <= aspect_ratio <= 1.15:
                    if area <= 1300:
                        return "Pituitary"
                    elif area <= 3000:
                        return "Meningioma" if distance >= 90 else "Glioma"
                    else:
                        return "Glioma"
                elif aspect_ratio < 0.7 or aspect_ratio > 1.5:
                    if area <= 3000:
                        return "Meningioma"
                    else:
                        return "Glioma"
                else:
                    if area > 3000:
                        return "Glioma"
                    elif distance >= 90:
                        return "Meningioma"
                    else:
                        return "Glioma"

            # Get prediction
            prediction = classify_tumor(area, aspect_ratio, cx, cy, w_img, h_img)

            # Show prediction
            st.image(pred_mask, caption="Predicted Mask")
            st.markdown(f"<h5 style='color: green;'>Tumor Type Prediction: <u>{prediction}</u></h5>", unsafe_allow_html=True)

            clear = st.button("Clear Prediction")
            if clear:
                del st.session_state["pred_mask"]
