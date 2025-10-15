import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import numpy as np
import time
import os
from fer import FER
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class OptimizedEmotionDetectionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("900x700")
        self.window.configure(bg='#2C3E50')

        # Load emotion detection model (using OpenCV DNN)
        self.load_emotion_model()

        # Initialize CSV data storage
        self.init_csv_storage()

        # Video capture with optimized settings
        self.vid = cv2.VideoCapture(0)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.vid.set(cv2.CAP_PROP_FPS, 30)

        self.is_running = False
        self.frame_skip = 3  # Process every 3rd frame for better performance
        self.frame_count = 0

        # Current emotion
        self.current_emotion = "No face detected"
        self.emotion_confidence = {}
        self.last_detection_time = 0
        self.detection_interval = 0.2  # Detect emotion every 200ms

        # Create GUI elements
        self.create_widgets()

        # Start video loop
        self.is_running = True
        self.update()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def load_emotion_model(self):
        """Load pre-trained emotion detection model using FER library"""
        try:
            # Initialize FER detector with MTCNN for better face detection
            self.emotion_detector = FER(mtcnn=True)
            
            # Fallback face cascade for visualization
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Emotion labels that FER uses
            self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            
            print("FER emotion detection model loaded successfully")
        except Exception as e:
            print(f"Error loading emotion model: {e}")
            self.emotion_detector = None
            # Fallback to basic cascade
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def init_csv_storage(self):
        """Initialize CSV file for storing emotion data"""
        self.csv_file = "emotion_data.csv"
        self.csv_lock = threading.Lock()

        # Create CSV file with headers if it doesn't exist
        if not os.path.exists(self.csv_file):
            df = pd.DataFrame(columns=[
                'id', 'timestamp', 'dominant_emotion',
                'angry_conf', 'disgust_conf', 'fear_conf', 'happy_conf',
                'sad_conf', 'surprise_conf', 'neutral_conf'
            ])
            df.to_csv(self.csv_file, index=False)
            print("CSV file initialized successfully")
        else:
            print("CSV file already exists")

    def save_emotion_data(self, emotion, emotions_dict):
        """Save emotion detection data to CSV file"""
        try:
            with self.csv_lock:
                # Read existing data to get next ID
                df = pd.read_csv(self.csv_file)
                next_id = len(df) + 1 if not df.empty else 1

                # Create new row data
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                new_row = {
                    'id': next_id,
                    'timestamp': timestamp,
                    'dominant_emotion': emotion,
                    'angry_conf': emotions_dict.get('angry', 0),
                    'disgust_conf': emotions_dict.get('disgust', 0),
                    'fear_conf': emotions_dict.get('fear', 0),
                    'happy_conf': emotions_dict.get('happy', 0),
                    'sad_conf': emotions_dict.get('sad', 0),
                    'surprise_conf': emotions_dict.get('surprise', 0),
                    'neutral_conf': emotions_dict.get('neutral', 0)
                }

                # Append new row
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df.to_csv(self.csv_file, index=False)

        except Exception as e:
            print(f"Error saving emotion data to CSV: {e}")

    def get_analytics_data(self):
        """Get analytics data from CSV file"""
        try:
            with self.csv_lock:
                df = pd.read_csv(self.csv_file)

                if df.empty:
                    return None

                # Get total detections
                total_detections = len(df)

                # Get emotion distribution
                emotion_counts = df['dominant_emotion'].value_counts()
                emotion_distribution = [(emotion, count) for emotion, count in emotion_counts.items()]

                # Get average confidence scores
                avg_confidences = []
                emotion_columns = ['angry_conf', 'disgust_conf', 'fear_conf', 'happy_conf', 'sad_conf', 'surprise_conf', 'neutral_conf']
                for col in emotion_columns:
                    avg_val = df[col].mean()
                    avg_confidences.append(avg_val if not pd.isna(avg_val) else 0)

                # Get recent detections (last 50)
                recent_df = df.tail(50)
                recent_detections = [(f"{int(row['id'])} - {row['timestamp']}", row['dominant_emotion']) for _, row in recent_df.iterrows()]

                return {
                    'total_detections': total_detections,
                    'emotion_distribution': emotion_distribution,
                    'avg_confidences': avg_confidences,
                    'recent_detections': recent_detections,
                    'dataframe': df  # Include full dataframe for additional analysis
                }
        except Exception as e:
            print(f"Error reading analytics data from CSV: {e}")
            return None

    def create_widgets(self):
        # Title
        title_label = tk.Label(
            self.window,
            text="Real-Time Emotion Detection (Optimized)",
            font=("Helvetica", 24, "bold"),
            bg='#2C3E50',
            fg='white'
        )
        title_label.pack(pady=10)

        # Main frame
        main_frame = tk.Frame(self.window, bg='#2C3E50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Video frame
        video_frame = tk.Frame(main_frame, bg='#34495E', relief=tk.RIDGE, bd=3)
        video_frame.pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(video_frame, width=640, height=480, bg='black')
        self.canvas.pack(padx=5, pady=5)

        # Info frame
        info_frame = tk.Frame(main_frame, bg='#34495E', relief=tk.RIDGE, bd=3)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        # Current emotion label
        emotion_title = tk.Label(
            info_frame,
            text="Detected Emotion:",
            font=("Helvetica", 16, "bold"),
            bg='#34495E',
            fg='white'
        )
        emotion_title.pack(pady=10)

        self.emotion_label = tk.Label(
            info_frame,
            text=self.current_emotion,
            font=("Helvetica", 28, "bold"),
            bg='#34495E',
            fg='#3498DB'
        )
        self.emotion_label.pack(pady=10)

        # Emotion confidence bars
        confidence_title = tk.Label(
            info_frame,
            text="Emotion Confidence:",
            font=("Helvetica", 14, "bold"),
            bg='#34495E',
            fg='white'
        )
        confidence_title.pack(pady=(20, 10))

        # Frame for progress bars
        self.progress_frame = tk.Frame(info_frame, bg='#34495E')
        self.progress_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        # Emotion colors
        self.emotion_colors = {
            'angry': '#E74C3C',
            'disgust': '#9B59B6',
            'fear': '#34495E',
            'happy': '#F39C12',
            'sad': '#3498DB',
            'surprise': '#E67E22',
            'neutral': '#95A5A6'
        }

        # Create progress bars for each emotion
        self.progress_bars = {}
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        for emotion in emotions:
            frame = tk.Frame(self.progress_frame, bg='#34495E')
            frame.pack(fill=tk.X, pady=5)

            label = tk.Label(
                frame,
                text=emotion.capitalize(),
                font=("Helvetica", 10),
                bg='#34495E',
                fg='white',
                width=10,
                anchor='w'
            )
            label.pack(side=tk.LEFT)

            progress = ttk.Progressbar(
                frame,
                length=150,
                mode='determinate',
                maximum=100
            )
            progress.pack(side=tk.LEFT, padx=5)

            value_label = tk.Label(
                frame,
                text="0%",
                font=("Helvetica", 10),
                bg='#34495E',
                fg='white',
                width=5
            )
            value_label.pack(side=tk.LEFT)

            self.progress_bars[emotion] = (progress, value_label)

        # Control buttons
        button_frame = tk.Frame(self.window, bg='#2C3E50')
        button_frame.pack(pady=10)

        self.toggle_button = tk.Button(
            button_frame,
            text="Pause Detection",
            command=self.toggle_detection,
            font=("Helvetica", 12),
            bg='#E74C3C',
            fg='white',
            width=15,
            height=2
        )
        self.toggle_button.pack(side=tk.LEFT, padx=10)

        analytics_button = tk.Button(
            button_frame,
            text="View Analytics",
            command=self.show_analytics,
            font=("Helvetica", 12),
            bg='#3498DB',
            fg='white',
            width=15,
            height=2
        )
        analytics_button.pack(side=tk.LEFT, padx=10)

        csv_button = tk.Button(
            button_frame,
            text="Open CSV Data",
            command=self.open_csv_file,
            font=("Helvetica", 12),
            bg='#27AE60',
            fg='white',
            width=15,
            height=2
        )
        csv_button.pack(side=tk.LEFT, padx=10)

        exit_button = tk.Button(
            button_frame,
            text="Exit",
            command=self.on_closing,
            font=("Helvetica", 12),
            bg='#95A5A6',
            fg='white',
            width=15,
            height=2
        )
        exit_button.pack(side=tk.LEFT, padx=10)

    def detect_emotion_simple(self, frame):
        """Advanced emotion detection using FER library"""
        try:
            if self.emotion_detector is None:
                return None, None

            # Detect emotions in the frame
            result = self.emotion_detector.detect_emotions(frame)
            
            if result:
                # Get the face with highest bounding box area (largest face)
                largest_face = max(result, key=lambda x: x['box'][2] * x['box'][3])
                emotions = largest_face['emotions']
                
                # Get the emotion with highest confidence
                dominant_emotion = max(emotions, key=emotions.get)
                
                # Normalize emotion scores to percentages
                total = sum(emotions.values())
                normalized_emotions = {k: v/total for k, v in emotions.items()}
                
                return dominant_emotion, normalized_emotions
            else:
                return None, None
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return None, None

    def update(self):
        """Update video frame with optimized processing"""
        if self.is_running:
            ret, frame = self.vid.read()

            if ret:
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                self.frame_count += 1

                # Only process emotion detection every few frames and at intervals
                current_time = time.time()
                if (self.frame_count % self.frame_skip == 0 and
                    current_time - self.last_detection_time > self.detection_interval):
                    self.last_detection_time = current_time
                    # Detect emotion in a separate thread to avoid blocking
                    threading.Thread(target=self.process_frame, args=(frame.copy(),), daemon=True).start()

                # Always update face detection for visual feedback
                self.update_face_detection(frame)

                # Convert frame to PhotoImage
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(image=img)

                # Update canvas
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(10, self.update)

    def update_face_detection(self, frame):
        """Update face detection overlay using FER"""
        if self.emotion_detector is not None:
            try:
                # Detect faces using FER (which uses MTCNN internally)
                result = self.emotion_detector.detect_emotions(frame)
                
                for face_data in result:
                    box = face_data['box']
                    x, y, w, h = box
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, self.current_emotion, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Face detection error: {e}")
        else:
            # Fallback to OpenCV cascade if FER fails
            if self.face_cascade is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, self.current_emotion, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    def process_frame(self, frame):
        """Process frame for emotion detection"""
        emotion, emotions = self.detect_emotion_simple(frame)

        if emotion and emotions:
            self.current_emotion = emotion.capitalize()
            self.emotion_confidence = emotions

            # Save to database
            self.save_emotion_data(emotion, emotions)

            # Update GUI
            self.window.after(0, self.update_emotion_display)

    def update_emotion_display(self):
        """Update emotion display on GUI"""
        self.emotion_label.config(text=self.current_emotion)

        # Update progress bars
        for emotion, (progress, label) in self.progress_bars.items():
            if emotion in self.emotion_confidence:
                value = self.emotion_confidence[emotion] * 100
                progress['value'] = value
                label.config(text=f"{value:.1f}%")

    def toggle_detection(self):
        """Toggle detection on/off"""
        self.is_running = not self.is_running
        if self.is_running:
            self.toggle_button.config(text="Pause Detection", bg='#E74C3C')
        else:
            self.toggle_button.config(text="Resume Detection", bg='#27AE60')

    def open_csv_file(self):
        """Open the CSV file in default application"""
        try:
            if os.path.exists(self.csv_file):
                os.startfile(self.csv_file)  # Windows specific
            else:
                messagebox.showerror("Error", "CSV file does not exist yet. Start emotion detection first.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open CSV file: {e}")

    def show_analytics(self):
        """Show analytics window with emotion detection statistics"""
        analytics_data = self.get_analytics_data()

        if analytics_data is None:
            tk.messagebox.showerror("Error", "No analytics data available")
            return

        # Create analytics window
        analytics_window = tk.Toplevel(self.window)
        analytics_window.title("Emotion Detection Analytics")
        analytics_window.geometry("800x600")
        analytics_window.configure(bg='#2C3E50')

        # Title
        title_label = tk.Label(
            analytics_window,
            text="Emotion Detection Analytics (CSV Storage)",
            font=("Helvetica", 20, "bold"),
            bg='#2C3E50',
            fg='white'
        )
        title_label.pack(pady=20)

        # Main frame
        main_frame = tk.Frame(analytics_window, bg='#2C3E50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Data file info
        file_info_frame = tk.Frame(main_frame, bg='#34495E', relief=tk.RIDGE, bd=3)
        file_info_frame.pack(fill=tk.X, pady=(0, 10))

        file_info_title = tk.Label(
            file_info_frame,
            text="Data Storage Information:",
            font=("Helvetica", 12, "bold"),
            bg='#34495E',
            fg='white'
        )
        file_info_title.pack(pady=10)

        # Get file info
        file_path = os.path.abspath(self.csv_file)
        file_size = os.path.getsize(self.csv_file) if os.path.exists(self.csv_file) else 0
        file_size_kb = file_size / 1024

        file_info_text = f"📁 File: {file_path}\n📊 Size: {file_size_kb:.1f} KB\n⏰ Last Modified: {datetime.fromtimestamp(os.path.getmtime(self.csv_file)).strftime('%Y-%m-%d %H:%M:%S') if os.path.exists(self.csv_file) else 'N/A'}"

        file_info_label = tk.Label(
            file_info_frame,
            text=file_info_text,
            font=("Courier", 9),
            bg='#34495E',
            fg='white',
            justify=tk.LEFT
        )
        file_info_label.pack(pady=(0, 10), padx=10)

        # Statistics frame
        stats_frame = tk.Frame(main_frame, bg='#34495E', relief=tk.RIDGE, bd=3)
        stats_frame.pack(fill=tk.X, pady=10)

        # Total detections
        total_label = tk.Label(
            stats_frame,
            text=f"Total Emotion Detections: {analytics_data['total_detections']}",
            font=("Helvetica", 14, "bold"),
            bg='#34495E',
            fg='white'
        )
        total_label.pack(pady=10)

        # Emotion distribution
        dist_title = tk.Label(
            stats_frame,
            text="Emotion Distribution:",
            font=("Helvetica", 12, "bold"),
            bg='#34495E',
            fg='white'
        )
        dist_title.pack(pady=(20, 10))

        for emotion, count in analytics_data['emotion_distribution']:
            emotion_label = tk.Label(
                stats_frame,
                text=f"{emotion.capitalize()}: {count} detections",
                font=("Helvetica", 10),
                bg='#34495E',
                fg='white'
            )
            emotion_label.pack(pady=2)

        # Average confidence scores
        conf_title = tk.Label(
            stats_frame,
            text="Average Confidence Scores:",
            font=("Helvetica", 12, "bold"),
            bg='#34495E',
            fg='white'
        )
        conf_title.pack(pady=(20, 10))

        avg_conf = analytics_data['avg_confidences']
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        for i, emotion in enumerate(emotions):
            conf_value = avg_conf[i] if i < len(avg_conf) else 0
            conf_label = tk.Label(
                stats_frame,
                text=f"{emotion.capitalize()}: {conf_value:.1%}",
                font=("Helvetica", 10),
                bg='#34495E',
                fg='white'
            )
            conf_label.pack(pady=2)

        # Recent detections
        recent_frame = tk.Frame(main_frame, bg='#34495E', relief=tk.RIDGE, bd=3)
        recent_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        recent_title = tk.Label(
            recent_frame,
            text="Recent Detections (Last 50):",
            font=("Helvetica", 12, "bold"),
            bg='#34495E',
            fg='white'
        )
        recent_title.pack(pady=10)

        # Create text widget for recent detections
        text_widget = tk.Text(
            recent_frame,
            height=15,
            width=60,
            bg='#2C3E50',
            fg='white',
            font=("Courier", 9)
        )
        text_widget.pack(pady=10, padx=10)

        # Add recent detections to text widget
        for timestamp, emotion in analytics_data['recent_detections']:
            text_widget.insert(tk.END, f"{timestamp} - {emotion.capitalize()}\n")
        text_widget.config(state=tk.DISABLED)

        # Add export button
        export_frame = tk.Frame(analytics_window, bg='#2C3E50')
        export_frame.pack(pady=(0, 20))

        export_button = tk.Button(
            export_frame,
            text="Export Data to Excel",
            command=lambda: self.export_to_excel(analytics_data),
            font=("Helvetica", 10),
            bg='#F39C12',
            fg='white',
            width=20,
            height=2
        )
        export_button.pack(side=tk.LEFT, padx=10)

        refresh_button = tk.Button(
            export_frame,
            text="Refresh Analytics",
            command=lambda: self.refresh_analytics(analytics_window),
            font=("Helvetica", 10),
            bg='#9B59B6',
            fg='white',
            width=20,
            height=2
        )
        refresh_button.pack(side=tk.LEFT, padx=10)

    def export_to_excel(self, analytics_data):
        """Export analytics data to Excel file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_file = f"emotion_analytics_{timestamp}.xlsx"

            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Raw data
                analytics_data['dataframe'].to_excel(writer, sheet_name='Raw_Data', index=False)

                # Summary statistics
                summary_data = {
                    'Metric': ['Total Detections', 'Most Common Emotion', 'Average Confidence'],
                    'Value': [
                        analytics_data['total_detections'],
                        analytics_data['emotion_distribution'][0][0] if analytics_data['emotion_distribution'] else 'N/A',
                        f"{sum(analytics_data['avg_confidences'])/len(analytics_data['avg_confidences']):.1%}" if analytics_data['avg_confidences'] else 'N/A'
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

                # Emotion distribution
                dist_data = {'Emotion': [], 'Count': [], 'Percentage': []}
                total = analytics_data['total_detections']
                for emotion, count in analytics_data['emotion_distribution']:
                    dist_data['Emotion'].append(emotion.capitalize())
                    dist_data['Count'].append(count)
                    dist_data['Percentage'].append(f"{(count/total)*100:.1f}%")
                pd.DataFrame(dist_data).to_excel(writer, sheet_name='Distribution', index=False)

            messagebox.showinfo("Success", f"Data exported to {excel_file}")
            os.startfile(excel_file)

        except Exception as e:
            messagebox.showerror("Error", f"Could not export to Excel: {e}")

    def refresh_analytics(self, analytics_window):
        """Refresh the analytics window with latest data"""
        analytics_window.destroy()
        self.show_analytics()

    def on_closing(self):
        """Clean up on closing"""
        self.is_running = False
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()

if __name__ == "__main__":
    app = OptimizedEmotionDetectionApp(tk.Tk(), "Emotion Detection App")