import pandas as pd
from ultralytics import YOLO
from PIL import Image
import streamlit as st
import tempfile
import os

# Streamlit page config
st.set_page_config(
    page_title="Disease Detection & Diagnosis",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Disease Detection & Diagnosis from Chest X-rays")
st.markdown("""
Upload a chest X-ray image and let the AI detect diseases like **Pneumonia**, **Effusion**, **Atelectasis**, and more using YOLOv8.
""")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")  # Replace with your trained model if available

model = load_model()

# Load ground truth CSV and clean columns
@st.cache_data
def load_ground_truth():
    # Always read with header row
    df = pd.read_csv("Ground_Truth.csv", header=0)
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    return df

ground_truth_df = load_ground_truth()

# Show available columns
st.write("📄 Available columns in Ground_Truth.csv:")
st.write(ground_truth_df.columns.tolist())

# Upload image
uploaded_file = st.file_uploader("📤 Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    image_filename = uploaded_file.name
    st.write(f"📁 **Uploaded Image:** `{image_filename}`")

    # Match image filename
    matching_gt = ground_truth_df[ground_truth_df["Image Index"] == image_filename]

    # Find label column dynamically
    label_col = next((col for col in ground_truth_df.columns if "label" in col.lower()), None)

    # Display ground truth label if available
    if not matching_gt.empty and label_col:
        gt_label = matching_gt.iloc[0][label_col]
        st.info(f"🧾 **Ground Truth:** `{gt_label}`")
    elif not matching_gt.empty:
        st.warning("⚠️ Matching image found, but no label column detected.")
    else:
        st.warning("⚠️ No ground truth found for this image.")

    # Save temp image and run YOLO
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model(tmp.name)

    # Convert to DataFrame
    try:
        results_df = results[0].to_df()
        st.write("📊 YOLO Result Columns:")
        st.write(results_df.columns.tolist())
    except Exception as e:
        st.error(f"❌ Error converting YOLO results to DataFrame: {e}")
        os.remove(tmp.name)
        st.stop()

    # Process detection results
    if "box" in results_df.columns:
        st.write("Sample 'box' values:")
        st.write(results_df["box"].head())

        try:
            results_df["xmin"] = results_df["box"].apply(lambda x: x["x1"])
            results_df["ymin"] = results_df["box"].apply(lambda x: x["y1"])
            results_df["xmax"] = results_df["box"].apply(lambda x: x["x2"])
            results_df["ymax"] = results_df["box"].apply(lambda x: x["y2"])

            # Filter and deduplicate
            results_df = results_df[results_df.confidence > 0.3]
            results_df = results_df.drop_duplicates(subset=["xmin", "ymin", "xmax", "ymax"])

            st.subheader("🔍 Detected Conditions")
            st.image(results[0].plot(), caption="Detection Result", use_container_width=True)

            if len(results_df) > 0:
                for _, row in results_df.iterrows():
                    class_name = row["name"]
                    confidence = row["confidence"] * 100
                    st.markdown(f"- **{class_name}** — {confidence:.2f}% confidence")
            else:
                st.success("✅ No diseases confidently detected.")
        except Exception as e:
            st.error(f"❌ Error extracting bounding boxes: {e}")
    else:
        st.error("Bounding box info not found in detection result.")

    os.remove(tmp.name)
