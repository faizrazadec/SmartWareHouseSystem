import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import pandas as pd
from rfdetr import RFDETRBase
import supervision as sv
import sqlite3
import uuid
import datetime

st.set_page_config(page_title="Defect Detection", layout="wide")

st.markdown("""
    <style>
    body, .stApp {
        background-color: #0f1117;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    # .main-frame {
    #     background-color: #1e1e1e;
    #     border-radius: 8px;
    #     padding: 15px;
    #     margin: 10px 0;
    #     box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    # }
    /* More compact metric boxes */
    .metric-box {
        background-color: #2c2f36;
        border-radius: 6px;
        padding: 10px 5px;
        margin-bottom: 8px;
        color: white;
        text-align: center;
        font-size: 0.9em;
    }
    /* Smaller headers */
    h1 {
        font-size: 1.8rem !important;
        margin-bottom: 0.5rem !important;
    }
    h2, .stSubheader {
        color: #61dafb;
        font-size: 1.4rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    h3 {
        font-size: 1.1rem !important;
        margin-top: 0.3rem !important;
        margin-bottom: 0.3rem !important;
    }
    /* Smaller buttons */
    .stButton > button {
        background-color: #61dafb;
        color: #000000;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 14px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #3fa6e6;
        box-shadow: 0 1px 4px rgba(0,0,0,0.2);
    }
    /* Reduce spacing */
    .section-gap {
        margin-top: 15px;
        margin-bottom: 15px;
    }
    /* More compact slider */
    .stSlider {
        padding-top: 0.5rem !important;
        padding-bottom: 1rem !important;
    }
    /* More compact file uploader */
    .stFileUploader {
        padding-bottom: 0.5rem !important;
    }
    /* Reduce empty space in containers */
    .stContainer, div[data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
        padding: 0.2rem 0 !important;
    }
    /* Smaller text in info boxes */
    .stAlert {
        font-size: 0.9rem !important;
        padding: 8px 10px !important;
    }
    /* Smaller horizontal rule */
    hr {
        margin: 10px 0 !important;
    }
    /* Reduce image caption size */
    .stImage img + div {
        font-size: 0.8rem !important;
    }
    /* Reduce expander padding */
    .streamlit-expanderContent {
        padding-top: 0.5rem !important;
    }
    /* Reduce chart height */
    .st-emotion-cache-zq5wmm {
        height: auto !important;
    }
    /* Reduce camera widget size */
    .st-emotion-cache-1v04i6q {
        max-height: 300px !important;
    }
    /* Right-aligned radio buttons */
    .header-radio .row-widget.stRadio {
        display: flex;
        justify-content: flex-end;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        st.session_state.model_loading_error = None
        model = RFDETRBase(
            pretrain_weights="/home/faizraza/Personal/Projects/SmartWareHouseSystem/models/rfdetr_model/checkpoint_best_total (2).pth"
        )
        return model
    except Exception as e:
        st.session_state.model_loading_error = str(e)
        st.error(f"Error loading model: {e}")
        return None

def load_data():
    try:
        st.session_state.model_loading_error = None
        ds = sv.DetectionDataset.from_coco(
            images_directory_path="dataset/box_dataset/test",
            annotations_path="dataset/box_dataset/test/_annotations.coco.json",
        )
        return ds
    except Exception as e:
        st.session_state.model_loading_error = str(e)
        st.error(f"Error loading dataset: {e}")
        return None

def process_image(image: Image.Image, model, ds, confidence_threshold=0.5):
    """Run RF-DETR inference on a PIL Image, draw bboxes + scores with OpenCV, return PIL."""
    try:
        if model is None:
            st.error("Model failed to load. Cannot process image.")
            return None, None, None

        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        detections = model.predict(img_cv, threshold=confidence_threshold)

        for (x1, y1, x2, y2), score, class_id in zip(
            detections.xyxy, detections.confidence, detections.class_id
        ):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{ds.classes[class_id]} {score:.2f}"
            cv2.putText(
                img_cv, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

        annotated_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        detections_labels = [
            f"{ds.classes[c]} {s:.2f}"
            for c, s in zip(detections.class_id, detections.confidence)
        ]

        return annotated_pil, detections, detections_labels

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None, None

def create_defect_table():
    """Create the defect detection table if it doesn't exist."""
    conn = None
    try:
        conn = sqlite3.connect('db.db', 
                              timeout=10, isolation_level="IMMEDIATE")
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS defect_detections (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            defect_class TEXT,
            confidence REAL
        )
        ''')
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def save_defect_detection(defect_class, confidence):
    """Save a defect detection to the database."""
    detection_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    
    conn = None
    try:
        conn = sqlite3.connect('db.db', 
                              timeout=10, isolation_level="IMMEDIATE")
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO defect_detections (id, timestamp, defect_class, confidence) VALUES (?, ?, ?, ?)',
            (detection_id, timestamp, defect_class, confidence)
        )
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error saving to database: {e}")
        if conn:
            conn.rollback()
        return None
    finally:
        if conn:
            conn.close()
    return detection_id

def get_recent_defect_detections(limit=10):
    """Get recent defect detections from the database."""
    conn = None
    results = []
    try:
        conn = sqlite3.connect('db.db', 
                              timeout=10, isolation_level="IMMEDIATE")
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, timestamp, defect_class, confidence FROM defect_detections ORDER BY timestamp DESC LIMIT ?',
            (limit,)
        )
        results = cursor.fetchall()
    except sqlite3.Error as e:
        st.error(f"Error retrieving from database: {e}")
    finally:
        if conn:
            conn.close()
    
    if not results:
        return pd.DataFrame(columns=["ID", "Timestamp", "Defect Class", "Confidence"])
    
    df = pd.DataFrame(results, columns=["ID", "Timestamp", "Defect Class", "Confidence"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.strftime('%Y-%m-%d %H:%M:%S')
    df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.2%}")
    return df
    
    if not results:
        return pd.DataFrame(columns=["ID", "Timestamp", "Defect Class", "Confidence"])
    
    df = pd.DataFrame(results, columns=["ID", "Timestamp", "Defect Class", "Confidence"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.strftime('%Y-%m-%d %H:%M:%S')
    df["Confidence"] = df["Confidence"].apply(lambda x: f"{x:.2%}")
    return df

def main():
    st.markdown('<div class="main-frame">', unsafe_allow_html=True)
    
    create_defect_table()
    
    st.title("Defect Detection")
    
    for key in ("image", "annotated_image", "detections", "labels"):
        if key not in st.session_state:
            st.session_state[key] = None

    with st.spinner("Loading model..."):
        model = load_model()
    with st.spinner("Loading data..."):
        ds = load_data()
    
    source_option = st.radio("Source:", ["Upload Image", "Capture from Camera"], horizontal=True)
    
    left_col, right_col = st.columns([3, 2])
    
    with right_col:
        if source_option == "Upload Image":
            uploaded_file = st.file_uploader("Browse Image", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                st.session_state.image = Image.open(uploaded_file)
        else:
            st.info("Camera access requires permission from your browser.")
            camera_image = st.camera_input("Take a picture")
            if camera_image:
                st.session_state.image = Image.open(camera_image)
        
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### Sensitivity")
        confidence_threshold = st.slider(
            "", 0.0, 1.0, 0.5, 0.05, key="confidence_slider",
            help="Lower values detect more objects but may include false positives."
        )
        
        if st.session_state.image:
            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            if st.button("Detect Defects", key="detect_button", use_container_width=True):
                with st.spinner("Processing..."):
                    annotated, dets, labels = process_image(
                        st.session_state.image, model, ds, confidence_threshold
                    )
                    if annotated:
                        st.session_state.annotated_image = annotated
                        st.session_state.detections = dets
                        st.session_state.labels = labels
                        
                        if len(dets.class_id) > 0:
                            saved_detections = []
                            for class_id, confidence in zip(dets.class_id, dets.confidence):
                                defect_class = ds.classes[class_id]
                                detection_id = save_defect_detection(defect_class, float(confidence))
                                if detection_id:
                                    saved_detections.append(f"{defect_class} ({confidence:.2f})")
                            
                            if saved_detections:
                                st.success(f"Saved {len(saved_detections)} defect(s) to database: {', '.join(saved_detections)}")
                            else:
                                st.warning("Detected defects but had issues saving to the database.")
                    else:
                        st.error("Failed to process image.")
        
        if st.session_state.annotated_image:
            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            if st.button("Save Results", key="save_button", use_container_width=True):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    tmp_path = tmp.name
                    st.session_state.annotated_image.save(tmp_path)
                with open(tmp_path, 'rb') as f:
                    st.download_button(
                        label="Download Image",
                        data=f,
                        file_name="defect_detection_result.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
            
            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            if st.button("Clear Results", key="clear_button", use_container_width=True):
                for k in ('annotated_image','detections','labels'):
                    st.session_state[k] = None
                st.rerun()
    
    with left_col:
        if st.session_state.image:
            caption_original = "Original Image" if source_option == "Upload Image" else "Captured Image"
            
            if st.session_state.annotated_image:
                img_col1, img_col2 = st.columns(2)
                
                with img_col1:
                    st.subheader("Original Image")
                    st.image(st.session_state.image, caption=caption_original, width=300, use_container_width=False)
                
                with img_col2:
                    st.subheader("Detection Results")
                    st.image(st.session_state.annotated_image, caption="Detected Defects", width=300, use_container_width=False)
                    
                    if len(st.session_state.detections.class_id) == 0:
                        st.info("No defects detected in the image.")
            else:
                st.image(st.session_state.image, caption=caption_original, width=400, use_container_width=False)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    with st.expander("How to Use Defect Detection"):
        st.write("""
        **Step 1:** Upload or capture an image  
        **Step 2:** Adjust sensitivity  
        **Step 3:** Click Detect Defects  
        **Step 4:** Review, save or clear results
        """)
    with st.expander("About the Technology"):
        st.write("""
        Uses RF-DETR to detect defects in cardboard boxes.  
        Confidence threshold filters predictions.  
        """)
    
    with st.expander("Recent Defect Detections"):
        recent_detections = get_recent_defect_detections(limit=10)
        if not recent_detections.empty:
            st.dataframe(recent_detections, use_container_width=True)
        else:
            st.info("No defect detections have been recorded yet.")
            
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
