import cv2
from transformers import pipeline
from PIL import Image

print("Loading emotion detection model...")

# Load Hugging Face emotion model
emotion_pipeline = pipeline(
    "image-classification",
    model="trpakov/vit-face-expression"
)

print("Model loaded successfully.")

# Load Haar Cascade face detector
facecasc = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if facecasc.empty():
    print("Error: Could not load Haar Cascade.")
    exit()

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

print("Webcam started. Press 'q' to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = facecasc.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Process each detected face
    for (x, y, w, h) in faces:

        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # Crop face region
        roi = frame[y:y+h, x:x+w]

        if roi.size == 0:
            continue

        try:
            # Convert BGR to RGB
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            # Convert to PIL image
            pil_img = Image.fromarray(roi_rgb)

            # Predict emotion
            preds = emotion_pipeline(pil_img)

            # Get highest confidence prediction
            label = preds[0]["label"]
            score = preds[0]["score"]

            # Display prediction
            text = f"{label}: {score:.2f}"

            cv2.putText(
                frame,
                text,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        except Exception as e:
            print("Prediction error:", e)

    # Show video frame
    cv2.imshow("Real-Time Emotion Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()