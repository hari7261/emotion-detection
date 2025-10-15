# Real-Time Emotion Detection System

A sophisticated Python-based application that performs real-time emotion detection using computer vision and deep learning. The system captures live video from a webcam, detects faces, analyzes facial expressions, and identifies seven different emotions with confidence scores.

## 🎯 Features

- **Real-time Emotion Detection**: Live video processing with instant emotion recognition
- **Advanced Face Detection**: Uses MTCNN (Multi-task Cascaded Convolutional Networks) for accurate face detection
- **Seven Emotion Categories**: Detects angry, disgust, fear, happy, sad, surprise, and neutral emotions
- **Confidence Scoring**: Provides percentage confidence for each emotion type
- **Data Persistence**: Automatically saves all detections to CSV format with timestamps
- **Interactive GUI**: Modern Tkinter-based interface with real-time visualization
- **Analytics Dashboard**: Comprehensive statistics and emotion distribution analysis
- **Data Export**: Export analytics data to Excel format for further analysis
- **Performance Optimized**: Frame skipping and threading for smooth real-time performance
- **Visual Feedback**: Live video feed with face bounding boxes and emotion labels

## 📸 Screenshots

### Main Application Interface
![Emotion Detection Interface](Screenshot%202025-10-16%20011405.png)
*The main application interface showing real-time emotion detection with live video feed and confidence bars*

### Analytics Dashboard
![Analytics Dashboard](Screenshot%202025-10-16%20011425.png)
*Comprehensive analytics dashboard displaying emotion distribution, statistics, and recent detections*

### Data Export Feature
![Data Export](Screenshot%202025-10-16%20011450.png)
*Excel export functionality showing data analysis and export options*

### Real-time Detection in Action
![Live Detection](Screenshot%202025-10-16%20011521.png)
*Real-time emotion detection with face bounding boxes and emotion labels overlay*

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (built-in or external)
- Windows/Linux/macOS

### Step-by-Step Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd emotion-detection
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv emotion_env
   emotion_env\Scripts\activate  # On Windows
   # or
   source emotion_env/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python optimized_app.py
   ```

## 📋 Dependencies

The project uses the following key libraries:

- **OpenCV** (`opencv-python==4.8.1.78`): Computer vision and video processing
- **FER** (`fer==22.5.1`): Facial Expression Recognition library
- **TensorFlow** (`>=2.16.0`): Deep learning framework for emotion detection
- **MTCNN** (`mtcnn==0.1.1`): Multi-task Cascaded Convolutional Networks for face detection
- **Pillow** (`pillow==10.1.0`): Image processing
- **NumPy** (`>=1.24.3`): Numerical computing
- **Pandas** (`>=1.5.0`): Data manipulation and CSV handling
- **Matplotlib** (`>=3.5.0`): Data visualization for analytics
- **OpenPyXL** (`>=3.0.0`): Excel file generation

## 🚀 Usage

### Basic Operation

1. **Launch the Application**: Run `python optimized_app.py`
2. **Position Yourself**: Ensure your face is clearly visible in the camera
3. **View Real-time Results**: Watch as emotions are detected and displayed
4. **Monitor Confidence**: Check the confidence bars for each emotion type
5. **Pause/Resume**: Use the pause button to temporarily stop detection

### Analytics Features

- **View Analytics**: Click "View Analytics" to see comprehensive statistics
- **Export Data**: Export your emotion data to Excel format
- **Open CSV**: Directly access the raw data file
- **Real-time Statistics**: Monitor total detections and emotion distributions

### Data Storage

All emotion detections are automatically saved to `emotion_data.csv` with the following columns:
- `id`: Unique detection identifier
- `timestamp`: Date and time of detection
- `dominant_emotion`: The primary detected emotion
- `angry_conf`, `disgust_conf`, `fear_conf`, `happy_conf`, `sad_conf`, `surprise_conf`, `neutral_conf`: Confidence scores for each emotion

## 🏗️ Project Structure

```
emotion-detection/
│
├── optimized_app.py          # Main application file
├── emotion_data.csv          # Generated emotion detection data
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── screenshot.png            # Application screenshot
└── __pycache__/              # Python cache files
```

## 🔧 Technical Details

### Emotion Detection Algorithm

The system uses the FER (Facial Expression Recognition) library, which implements:
- **Face Detection**: MTCNN for robust face localization
- **Feature Extraction**: Deep convolutional neural networks
- **Emotion Classification**: Pre-trained model for 7 emotion categories
- **Confidence Calculation**: Probability distribution across all emotions

### Performance Optimizations

- **Frame Skipping**: Processes every 3rd frame to reduce computational load
- **Threading**: Emotion detection runs in background threads
- **Detection Intervals**: Minimum 200ms between detections for stability
- **Memory Management**: Efficient frame processing and cleanup

### GUI Framework

Built with Tkinter featuring:
- **Modern Design**: Dark theme with professional styling
- **Real-time Updates**: Live video feed and emotion displays
- **Interactive Controls**: Buttons for analytics, data export, and controls
- **Progress Bars**: Visual confidence indicators for each emotion

## 📊 Analytics and Reporting

The analytics dashboard provides:
- **Total Detection Count**: Running total of all emotion detections
- **Emotion Distribution**: Pie chart of emotion frequencies
- **Average Confidence Scores**: Mean confidence across all detections
- **Recent Detections**: Last 50 detections with timestamps
- **Data Export**: Excel export with multiple worksheets

## 🐛 Troubleshooting

### Common Issues

1. **Camera Not Detected**
   - Ensure your webcam is properly connected
   - Check camera permissions in your OS settings
   - Try closing other applications using the camera

2. **Low Detection Accuracy**
   - Ensure good lighting on your face
   - Position yourself 2-3 feet from the camera
   - Remove glasses or hats that might obstruct facial features

3. **Performance Issues**
   - Close other resource-intensive applications
   - Ensure you have sufficient RAM (4GB+ recommended)
   - Update your graphics drivers

4. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)
   - Try reinstalling TensorFlow if CUDA errors occur

### Error Messages

- **"FER model not loaded"**: Check internet connection for initial model download
- **"Camera not accessible"**: Grant camera permissions to Python/Tkinter
- **"CSV write error"**: Ensure write permissions in the project directory

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -am 'Add new feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **FER Library**: Facial Expression Recognition by Justin Shenk
- **OpenCV**: Computer Vision library
- **TensorFlow**: Machine Learning framework
- **MTCNN**: Face detection implementation

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the error messages in the console
3. Ensure all dependencies are correctly installed
4. Test with different lighting conditions and camera angles

---

**Note**: This application requires a functioning webcam and may take a few seconds to load the emotion detection model on first run.