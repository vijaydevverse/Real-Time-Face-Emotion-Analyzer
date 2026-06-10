# 🎭 Real-Time Emotion Detection

A real-time facial emotion detection system built using **Python, OpenCV, Haar Cascade Face Detection, and Hugging Face Transformers**. The application captures live webcam video, detects faces, and predicts facial emotions with confidence scores using a Vision Transformer (ViT) model.

---

## 📖 Project Description

This project uses computer vision and deep learning techniques to recognize human emotions from facial expressions in real time. Faces are detected using OpenCV's Haar Cascade Classifier, and emotions are classified using the pre-trained **trpakov/vit-face-expression** model from Hugging Face.

The system displays the detected emotion and confidence score directly on the webcam feed.

---

## ✨ Features

* 🎥 Real-time webcam video processing
* 😊 Facial emotion recognition
* 👤 Face detection using Haar Cascade
* 🤖 Deep learning-based emotion classification
* 📊 Confidence score display
* 👥 Multiple face support
* ⚡ Fast and lightweight implementation

---

## 🛠️ Technologies Used

* Python
* OpenCV
* Hugging Face Transformers
* PyTorch
* Pillow (PIL)
* Haar Cascade Classifier
* Vision Transformer (ViT)

---

## 📂 Project Structure

```text
real-time-emotion-detection/
│
├── main.py
├── haarcascade_frontalface_default.xml
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/real-time-emotion-detection.git
cd real-time-emotion-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python main.py
```

Press **Q** to quit the application.

---

## 🤖 AI Model

**Model Name:** `trpakov/vit-face-expression`

The model is a Vision Transformer (ViT) trained for facial expression recognition and is accessed through the Hugging Face Transformers library.

---

## 🔄 Workflow

1. Access webcam feed.
2. Detect faces using Haar Cascade.
3. Extract face region.
4. Convert image to RGB format.
5. Pass image to ViT emotion model.
6. Predict emotion and confidence score.
7. Display results on the video frame.

---

## 🎯 Emotions Detected

* Happy 😊
* Sad 😢
* Angry 😠
* Neutral 😐
* Surprise 😲
* Fear 😨
* Disgust 🤢

---

## 📸 Sample Output

The application displays:

```text
Happy : 0.98
Sad : 0.87
Neutral : 0.95
```

Above each detected face in real time.

---

## 🚀 Future Enhancements

* Emotion analytics dashboard
* Emotion history tracking
* Image upload support
* Video file analysis
* YOLO-based face detection
* Streamlit web application deployment
* Emotion statistics and reporting

---

## 💡 Learning Outcomes

Through this project, I gained experience in:

* Computer Vision
* Deep Learning
* Face Detection
* Image Processing
* Hugging Face Transformers
* Real-Time AI Applications
* OpenCV Integration

---

## 👨‍💻 Author

**Vijay Krishnan**

---

## ⭐ Support

If you found this project useful, consider giving it a star on GitHub.

---

## 📜 License

This project is open-source and available under the MIT License.
