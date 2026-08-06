import av
import cv2
import numpy as np
import streamlit as st
from tflite_runtime.interpreter import Interpreter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Traffic Sign Recognition", layout="centered")
st.title("🚦 Real-Time Traffic Sign Recognition")
st.write("Point your webcam at a traffic sign. Predictions appear on the video feed below.")

THRESHOLD = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.75, 0.05)

# ---------------------------
# Load TFLite model (cached so it only loads once)
# ---------------------------
@st.cache_resource
def load_model():
    interpreter = Interpreter(model_path="model_trained.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------------------
# Class names
# ---------------------------
def get_class_name(class_no):
    classes = {
        0: "Speed Limit 20",
        1: "Stop",
        2: "Road Work",
        3: "Bumpy Road",
        4: "Road narrows on the right",
    }
    return classes.get(class_no, "Unknown")

# ---------------------------
# Preprocessing
# ---------------------------
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img

# ---------------------------
# Video processor for streamlit-webrtc
# ---------------------------
class SignProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        processed = cv2.resize(img, (32, 32))
        processed = preprocess(processed)
        processed = processed.reshape(1, 32, 32, 1).astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], processed)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]["index"])
        class_index = int(np.argmax(predictions))
        probability = float(np.max(predictions))

        cv2.putText(img, "CLASS: ", (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "PROBABILITY: ", (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), 2, cv2.LINE_AA)

        if probability > THRESHOLD:
            class_name = get_class_name(class_index)
            cv2.putText(img, class_name, (150, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, f"{round(probability * 100, 2)}%", (220, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------------------
# Launch webcam stream
# ---------------------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="traffic-sign-recognition",
    video_processor_factory=SignProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

st.caption("Model: model_trained.h5 · Runs entirely in your browser session via WebRTC")
