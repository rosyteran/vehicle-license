import tempfile

import cv2
from time import time
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import Annotator, colors
from datetime import datetime
from paddleocr import PaddleOCR


class SpeedEstimator(BaseSolution):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_region()  # Initialize speed region
        self.spd = {}  # Dictionary to store speed data
        self.trkd_ids = []  # List for already tracked and speed-estimated IDs
        self.trk_pt = {}  # Dictionary for previous timestamps
        self.trk_pp = {}  # Dictionary for previous positions
        self.logged_ids = set()  # Set to keep track of already logged IDs

        # Initialize the OCR system
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def perform_ocr(self, image_array):
        """Performs OCR on the given image and returns the extracted text."""
        if image_array is None:
            raise ValueError("Image is None")
        if isinstance(image_array, np.ndarray):
            results = self.ocr.ocr(image_array, rec=True)
        else:
            raise TypeError("Input image is not a valid numpy array")
        return ' '.join([result[1][0] for result in results[0]] if results[0] else "")

    def estimate_speed(self, im0):
        """Estimate speed of objects and track them."""
        self.annotator = Annotator(im0, line_width=self.line_width)  # Initialize annotator
        self.extract_tracks(im0)  # Extract tracks

        # Get current date and time
        current_time = datetime.now()

        for box, track_id, cls in zip(self.boxes, self.track_ids, self.clss):
            self.store_tracking_history(track_id, box)  # Store track history

            if track_id not in self.trk_pt:
                self.trk_pt[track_id] = 0
            if track_id not in self.trk_pp:
                self.trk_pp[track_id] = self.track_line[-1]

            # Speed and direction calculation
            if self.LineString([self.trk_pp[track_id], self.track_line[-1]]).intersects(self.r_s):
                direction = "known"
            else:
                direction = "unknown"

            # Calculate speed if the direction is known and the object is new
            if direction == "known" and track_id not in self.trkd_ids:
                self.trkd_ids.append(track_id)
                time_difference = time() - self.trk_pt[track_id]
                if time_difference > 0:
                    speed = np.abs(self.track_line[-1][1].item() - self.trk_pp[track_id][1].item()) / time_difference
                    self.spd[track_id] = round(speed)

            # Update the previous tracking time and position
            self.trk_pt[track_id] = time()
            self.trk_pp[track_id] = self.track_line[-1]
            x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers
            cropped_image = np.array(im0)[y1:y2, x1:x2]
            ocr_text = self.perform_ocr(cropped_image)

            # Get the class name and speed
            class_name = self.names[int(cls)]
            speed = self.spd.get(track_id)

            label = f"ID: {track_id} {ocr_text.strip()}"  # Show track ID along with speed
            if speed is not None:
                label += f" Speed: {speed}"

            self.annotator.box_label(box, label=label, color=colors(track_id, True))  # Draw bounding box

            # Ensure OCR text is not empty and save OCR text with the relevant details if not already logged
            if track_id not in self.logged_ids and ocr_text.strip() and speed is not None:
                self.logged_ids.add(track_id)

        self.display_output(im0)  # Display output with base class function
        return im0


def process_video(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name
    # Open the uploaded video file
    cap = cv2.VideoCapture(temp_path)

    # Define region points for counting
    region_points = [(0, 145), (1018, 145)]

    # Initialize the object counter
    speed_obj = SpeedEstimator(
        region=region_points,
        model="best.pt",  # Replace with your YOLO model file
        line_width=2
    )

    count = 0
    frame_placeholder = st.empty()
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % 3 != 0:  # Skip odd frames
            continue

        frame = cv2.resize(frame, (1020, 500))

        # Process the frame with the object counter
        result = speed_obj.estimate_speed(frame)

        # Convert frame to RGB for Streamlit display
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(result_rgb)

        frame_placeholder.image(img, use_container_width =True)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    st.title("Speed estimation from Video")

    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov"])

    if uploaded_file is not None:
        st.video(uploaded_file)
        st.write("Processing video...")
        process_video(uploaded_file)
