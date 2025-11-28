import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import json
import pyaudio
import wave
import threading
import speech_recognition as sr
from collections import deque
import os
import webbrowser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

class AudioAnalyzer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_recording = False
        self.audio_thread = None
        
        # Metrics
        self.total_words = 0
        self.filler_words = ['um', 'uh', 'like', 'you know', 'actually', 'basically', 'literally', 'so', 'right', 'okay']
        self.filler_count = {}
        self.speaking_times = []
        self.silence_times = []
        self.speech_rate = []  # words per minute
        self.volume_levels = deque(maxlen=100)
        self.last_speech_time = time.time()
        self.speech_segments = []
        self.energy_levels = []
        
        # Audio recording
        self.audio_frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
    def start_recording(self, filename="session_audio.wav"):
        self.is_recording = True
        self.audio_filename = filename
        
        # Start audio stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024,
            stream_callback=self.audio_callback
        )
        
        # Start speech recognition thread
        self.audio_thread = threading.Thread(target=self.continuous_recognition)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.audio_frames.append(in_data)
            # Calculate volume level
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            volume = np.abs(audio_data).mean()
            self.volume_levels.append(volume)
            self.energy_levels.append(volume)
        return (in_data, pyaudio.paContinue)
    
    def continuous_recognition(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while self.is_recording:
                try:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    text = self.recognizer.recognize_google(audio).lower()
                    
                    current_time = time.time()
                    self.speech_segments.append({
                        'time': current_time,
                        'text': text,
                        'duration': current_time - self.last_speech_time
                    })
                    
                    # Analyze speech
                    words = text.split()
                    self.total_words += len(words)
                    
                    # Count filler words
                    for filler in self.filler_words:
                        count = text.count(filler)
                        if count > 0:
                            self.filler_count[filler] = self.filler_count.get(filler, 0) + count
                    
                    self.last_speech_time = current_time
                    
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except Exception as e:
                    print(f"Audio error: {e}")
                    break
    
    def stop_recording(self):
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        # Save audio file
        if self.audio_frames:
            wf = wave.open(self.audio_filename, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(self.audio_frames))
            wf.close()
        
        self.audio.terminate()
    
    def get_metrics(self):
        total_fillers = sum(self.filler_count.values())
        avg_volume = np.mean(self.volume_levels) if self.volume_levels else 0
        energy_score = min(100, (avg_volume / 500) * 100)  # Normalize to 0-100
        
        # Calculate speech clarity (fewer fillers = better)
        clarity_score = 100
        if self.total_words > 0:
            filler_ratio = (total_fillers / self.total_words) * 100
            clarity_score = max(0, 100 - (filler_ratio * 5))
        
        return {
            'total_words': self.total_words,
            'filler_count': self.filler_count,
            'total_fillers': total_fillers,
            'clarity_score': int(clarity_score),
            'energy_score': int(energy_score),
            'avg_volume': int(avg_volume),
            'speech_segments': len(self.speech_segments)
        }


class InterviewCoach:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Tracking metrics
        self.session_start = None
        self.eye_contact_frames = 0
        self.total_frames = 0
        self.posture_alerts = 0
        self.fidget_count = 0
        self.smile_count = 0
        self.hand_gesture_count = 0
        self.excessive_hand_movement = 0
        self.looking_away_duration = 0
        self.head_tilt_count = 0
        
        # Advanced metrics
        self.confidence_over_time = []
        self.eye_contact_over_time = []
        self.energy_over_time = []
        
        # Audio analyzer
        self.audio_analyzer = AudioAnalyzer()
        
        # Recording
        self.output_video = None
        self.session_data = []
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def calculate_eye_contact(self, face_landmarks, frame_shape):
        """Enhanced eye contact detection with head pose estimation"""
        h, w = frame_shape[:2]
        
        # Get facial landmarks for gaze
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        nose_tip = face_landmarks.landmark[1]
        chin = face_landmarks.landmark[152]
        
        # Eye center
        eye_center_x = (left_eye.x + right_eye.x) / 2
        eye_center_y = (left_eye.y + right_eye.y) / 2
        
        # Face center
        face_center_x = nose_tip.x
        face_center_y = (nose_tip.y + chin.y) / 2
        
        # Check horizontal and vertical gaze
        h_diff = abs(eye_center_x - 0.5)  # Should be centered
        v_diff = abs(eye_center_y - 0.4)  # Eyes should be in upper half
        
        # Good eye contact if face is centered and looking forward
        is_looking = h_diff < 0.15 and v_diff < 0.15
        
        return is_looking
    
    def calculate_head_tilt(self, face_landmarks):
        """Detect excessive head tilting"""
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        
        # Calculate angle of eyes
        eye_slope = abs(left_eye.y - right_eye.y)
        is_tilted = eye_slope > 0.03
        
        return is_tilted
    
    def calculate_posture_score(self, pose_landmarks):
        """Enhanced posture analysis"""
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_ear = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
        nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
        
        # Shoulder alignment
        shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
        
        # Forward head posture (ears should be above shoulders)
        ear_avg_y = (left_ear.y + right_ear.y) / 2
        shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2
        forward_head = ear_avg_y > shoulder_avg_y
        
        # Check if leaning too far forward or backward
        shoulder_center_y = shoulder_avg_y
        lean_forward = nose.y > shoulder_center_y - 0.05
        
        good_posture = shoulder_slope < 0.05 and not forward_head and not lean_forward
        
        return good_posture
    
    def analyze_hand_gestures(self, hand_landmarks):
        """Analyze hand movements for natural gesturing"""
        if not hand_landmarks:
            return 0
        
        # Count hands visible
        hands_visible = len(hand_landmarks)
        
        # Natural gesturing: 1 hand occasionally, 2 hands sparingly
        return hands_visible
    
    def detect_smile(self, face_landmarks):
        """Enhanced smile detection with better accuracy"""
        # Mouth corners
        left_mouth = face_landmarks.landmark[61]
        right_mouth = face_landmarks.landmark[291]
        
        # Lips
        upper_lip = face_landmarks.landmark[13]
        lower_lip = face_landmarks.landmark[14]
        
        # Additional lip points for better detection
        left_lip_inner = face_landmarks.landmark[78]
        right_lip_inner = face_landmarks.landmark[308]
        
        # Calculate distances
        mouth_width = abs(left_mouth.x - right_mouth.x)
        mouth_height = abs(upper_lip.y - lower_lip.y)
        
        # Cheek points (to detect smile muscles)
        left_cheek = face_landmarks.landmark[205]
        right_cheek = face_landmarks.landmark[425]
        
        # Smile indicators
        # 1. Mouth should be wider than tall (good smile aspect ratio)
        mouth_aspect_ratio = mouth_width / (mouth_height + 0.001)
        
        # 2. Lips should curve upward (lower lip below upper lip significantly)
        lips_curve = lower_lip.y - upper_lip.y
        
        # 3. Mouth corners should be raised (positive y movement)
        mouth_corner_raise = abs(left_mouth.y - upper_lip.y)
        
        # 4. Cheeks should be raised (smile lines)
        cheek_raise_left = upper_lip.y - left_cheek.y
        cheek_raise_right = upper_lip.y - right_cheek.y
        
        # Stricter conditions for genuine smile
        is_smiling = (
            mouth_aspect_ratio > 3.5 and           # Wide mouth
            mouth_height > 0.008 and               # Mouth open enough
            mouth_height < 0.035 and               # But not too wide (not a yawn)
            lips_curve > 0.01 and                  # Lips curved
            mouth_corner_raise > 0.02 and          # Corners raised
            (cheek_raise_left > 0.01 or cheek_raise_right > 0.01)  # Cheeks raised
        )
        
        return is_smiling
    
    def draw_advanced_feedback(self, frame, metrics):
        """Enhanced UI with more information"""
        h, w = frame.shape[:2]
        
        # Main overlay panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Confidence score - Large and prominent
        confidence = metrics['confidence']
        color = (0, 255, 0) if confidence >= 70 else (0, 165, 255) if confidence >= 50 else (0, 0, 255)
        cv2.putText(frame, f"CONFIDENCE: {confidence}%", (w//2 - 200, 60), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)
        
        # Status indicators - Top row
        indicators = [
            ("Eye Contact", metrics['eye_contact'], (0, 255, 0), (0, 0, 255)),
            ("Posture", metrics['posture'], (0, 255, 0), (0, 165, 255)),
            ("Voice", metrics.get('voice_clarity', 0) > 70, (0, 255, 0), (0, 165, 255))
        ]
        
        x_pos = 20
        for label, status, good_col, bad_col in indicators:
            color = good_col if status else bad_col
            status_text = "✓" if status else "!"
            cv2.putText(frame, f"{label}: {status_text}", (x_pos, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            x_pos += 200
        
        # Metrics - Bottom row
        metrics_text = [
            f"Smiles: {metrics.get('smiles', 0)}",
            f"Gestures: {metrics.get('gestures', 0)}",
            f"Words: {metrics.get('words', 0)}"
        ]
        
        x_pos = 20
        for text in metrics_text:
            cv2.putText(frame, text, (x_pos, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            x_pos += 200
        
        # Real-time tips (right side)
        if not metrics['eye_contact']:
            cv2.putText(frame, "Look at camera", (w - 250, h - 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        if not metrics['posture']:
            cv2.putText(frame, "Sit up straight", (w - 250, h - 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Recording indicator
        if metrics.get('recording', False):
            cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 70, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame
    
    def calculate_overall_confidence(self):
        """Advanced confidence calculation with weighted factors"""
        if self.total_frames == 0:
            return 0
        
        # Weight factors
        eye_contact_score = (self.eye_contact_frames / self.total_frames) * 30
        
        posture_penalty = (self.posture_alerts / max(1, self.total_frames / 30))
        posture_score = max(0, 25 - posture_penalty * 2)
        
        smile_score = min(15, (self.smile_count / max(1, self.total_frames / 60)) * 15)
        
        gesture_score = min(15, (self.hand_gesture_count / max(1, self.total_frames / 30)) * 15)
        excessive_gesture_penalty = min(10, self.excessive_hand_movement / 10)
        gesture_score = max(0, gesture_score - excessive_gesture_penalty)
        
        fidget_penalty = min(5, self.fidget_count / 50)
        head_tilt_penalty = min(5, self.head_tilt_count / 30)
        
        # Audio metrics
        audio_metrics = self.audio_analyzer.get_metrics()
        voice_score = (audio_metrics['clarity_score'] * 0.1) + (audio_metrics['energy_score'] * 0.05)
        
        total = eye_contact_score + posture_score + smile_score + gesture_score + voice_score - fidget_penalty - head_tilt_penalty
        
        return int(max(0, min(100, total)))
    
    def start_session(self, output_filename=None):
        """Main session with all features"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Error: Could not open camera")
            return
        
        # Setup video recording
        if output_filename:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 20
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.output_video = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
        
        # Start audio recording
        audio_file = f"audio_{self.session_id}.wav"
        self.audio_analyzer.start_recording(audio_file)
        
        self.session_start = time.time()
        print("🎥 AI Interview Coach Pro - Started!")
        print("🔊 Analyzing: Video + Audio + Body Language")
        print("Press 'q' to end and view detailed report")
        print("=" * 60)
        
        prev_nose_pos = None
        prev_hand_pos = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process all detection models
            face_results = self.face_mesh.process(rgb_frame)
            pose_results = self.pose.process(rgb_frame)
            hand_results = self.hands.process(rgb_frame)
            
            # Initialize frame metrics
            eye_contact = False
            good_posture = False
            is_smiling = False
            hands_count = 0
            
            # Face analysis
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                eye_contact = self.calculate_eye_contact(face_landmarks, frame.shape)
                is_smiling = self.detect_smile(face_landmarks)
                is_tilted = self.calculate_head_tilt(face_landmarks)
                
                if eye_contact:
                    self.eye_contact_frames += 1
                else:
                    self.looking_away_duration += 1
                    
                if is_smiling:
                    self.smile_count += 1
                    
                if is_tilted:
                    self.head_tilt_count += 1
            
            # Posture analysis
            if pose_results.pose_landmarks:
                good_posture = self.calculate_posture_score(pose_results.pose_landmarks)
                if not good_posture:
                    self.posture_alerts += 1
                
                # Fidgeting detection
                nose = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
                if prev_nose_pos:
                    movement = np.sqrt((nose.x - prev_nose_pos[0])**2 + (nose.y - prev_nose_pos[1])**2)
                    if movement > 0.015:
                        self.fidget_count += 1
                prev_nose_pos = (nose.x, nose.y)
            
            # Hand gesture analysis
            if hand_results.multi_hand_landmarks:
                hands_count = len(hand_results.multi_hand_landmarks)
                self.hand_gesture_count += hands_count
                
                if hands_count > 1:
                    self.excessive_hand_movement += 1
            
            self.total_frames += 1
            
            # Calculate confidence
            confidence_score = self.calculate_overall_confidence()
            self.confidence_over_time.append(confidence_score)
            self.eye_contact_over_time.append(1 if eye_contact else 0)
            
            # Get audio metrics
            audio_metrics = self.audio_analyzer.get_metrics()
            
            # Prepare metrics for display
            display_metrics = {
                'confidence': confidence_score,
                'eye_contact': eye_contact,
                'posture': good_posture,
                'smiles': self.smile_count,
                'gestures': hands_count,
                'words': audio_metrics['total_words'],
                'voice_clarity': audio_metrics['clarity_score'],
                'recording': output_filename is not None
            }
            
            # Draw enhanced feedback
            frame = self.draw_advanced_feedback(frame, display_metrics)
            
            # Save frame if recording
            if self.output_video:
                self.output_video.write(frame)
            
            cv2.imshow('AI Interview Coach Pro', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        if self.output_video:
            self.output_video.release()
        cv2.destroyAllWindows()
        
        # Stop audio recording
        self.audio_analyzer.stop_recording()
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
    
    def create_visualization_charts(self):
        """Create beautiful charts for the report"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Interview Performance Analytics', fontsize=16, fontweight='bold')
        
        # 1. Confidence over time
        ax1 = axes[0, 0]
        time_axis = np.arange(len(self.confidence_over_time)) / 20  # Convert frames to seconds
        ax1.plot(time_axis, self.confidence_over_time, color='#4CAF50', linewidth=2)
        ax1.fill_between(time_axis, self.confidence_over_time, alpha=0.3, color='#4CAF50')
        ax1.set_title('Confidence Score Over Time', fontweight='bold')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Confidence %')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 100])
        
        # 2. Performance metrics radar
        ax2 = axes[0, 1]
        audio_metrics = self.audio_analyzer.get_metrics()
        
        categories = ['Eye\nContact', 'Posture', 'Voice\nClarity', 'Energy', 'Gestures']
        values = [
            (self.eye_contact_frames / self.total_frames) * 100,
            max(0, 100 - (self.posture_alerts / max(1, self.total_frames / 30)) * 10),
            audio_metrics['clarity_score'],
            audio_metrics['energy_score'],
            min(100, (self.hand_gesture_count / max(1, self.total_frames / 30)) * 20)
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax2 = plt.subplot(222, projection='polar')
        ax2.plot(angles, values, 'o-', linewidth=2, color='#2196F3')
        ax2.fill(angles, values, alpha=0.25, color='#2196F3')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories)
        ax2.set_ylim(0, 100)
        ax2.set_title('Performance Metrics', fontweight='bold', pad=20)
        ax2.grid(True)
        
        # 3. Filler words breakdown
        ax3 = axes[1, 0]
        filler_data = audio_metrics['filler_count']
        if filler_data:
            fillers = list(filler_data.keys())[:5]  # Top 5
            counts = [filler_data[f] for f in fillers]
            colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(fillers)))
            ax3.barh(fillers, counts, color=colors)
            ax3.set_title('Most Used Filler Words', fontweight='bold')
            ax3.set_xlabel('Count')
        else:
            ax3.text(0.5, 0.5, 'No filler words detected\n✓ Great job!', 
                    ha='center', va='center', fontsize=14, color='green')
            ax3.set_title('Filler Words Analysis', fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Key metrics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        duration = time.time() - self.session_start
        final_score = self.calculate_overall_confidence()
        
        summary_text = f"""
        📊 SESSION SUMMARY
        
        Duration: {duration:.1f} seconds
        Final Score: {final_score}/100
        
        📹 Visual Metrics:
        • Eye Contact: {(self.eye_contact_frames/self.total_frames)*100:.1f}%
        • Posture Alerts: {self.posture_alerts}
        • Smiles: {self.smile_count}
        • Hand Gestures: {self.hand_gesture_count}
        
        🎤 Audio Metrics:
        • Words Spoken: {audio_metrics['total_words']}
        • Filler Words: {audio_metrics['total_fillers']}
        • Clarity Score: {audio_metrics['clarity_score']}/100
        • Energy Level: {audio_metrics['energy_score']}/100
        """
        
        ax4.text(0.1, 0.9, summary_text, fontsize=11, family='monospace',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        chart_filename = f'report_charts_{self.session_id}.png'
        plt.savefig(chart_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        return chart_filename
    
    def generate_comprehensive_report(self):
        """Generate detailed HTML report with visualizations"""
        duration = time.time() - self.session_start
        final_score = self.calculate_overall_confidence()
        audio_metrics = self.audio_analyzer.get_metrics()
        
        # Create charts
        chart_file = self.create_visualization_charts()
        
        # Prepare detailed analysis
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'duration_seconds': duration,
            'final_confidence_score': final_score,
            'visual_metrics': {
                'total_frames': self.total_frames,
                'eye_contact_percentage': (self.eye_contact_frames / self.total_frames) * 100,
                'eye_contact_frames': self.eye_contact_frames,
                'posture_alerts': self.posture_alerts,
                'smile_count': self.smile_count,
                'fidget_count': self.fidget_count,
                'hand_gestures': self.hand_gesture_count,
                'excessive_gestures': self.excessive_hand_movement,
                'head_tilt_count': self.head_tilt_count
            },
            'audio_metrics': audio_metrics,
            'confidence_timeline': self.confidence_over_time[::20]  # Sample every second
        }
        
        # Save JSON report
        json_filename = f'interview_report_{self.session_id}.json'
        with open(json_filename, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Generate HTML report
        html_content = self.create_html_report(analysis, chart_file)
        html_filename = f'interview_report_{self.session_id}.html'
        
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Print console summary
        self.print_console_summary(analysis)
        
        # Open report in browser
        print(f"\n✅ Reports saved:")
        print(f"   📄 JSON: {json_filename}")
        print(f"   🌐 HTML: {html_filename}")
        print(f"   📊 Charts: {chart_file}")
        print("\n🚀 Opening detailed report in browser...")
        
        time.sleep(1)
        webbrowser.open('file://' + os.path.abspath(html_filename))
    
    def create_html_report(self, analysis, chart_file):
        """Create beautiful HTML report"""
        vm = analysis['visual_metrics']
        am = analysis['audio_metrics']
        score = analysis['final_confidence_score']
        
        # Performance grade
        if score >= 90:
            grade = "A+"
            grade_color = "#4CAF50"
            feedback = "Outstanding! You're interview-ready!"
        elif score >= 80:
            grade = "A"
            grade_color = "#8BC34A"
            feedback = "Excellent performance! Minor tweaks needed."
        elif score >= 70:
            grade = "B"
            grade_color = "#FFC107"
            feedback = "Good job! Practice these areas for perfection."
        elif score >= 60:
            grade = "C"
            grade_color = "#FF9800"
            feedback = "Decent effort. Focus on improvement areas."
        else:
            grade = "D"
            grade_color = "#F44336"
            feedback = "Keep practicing! You'll get better."
        
        # Recommendations
        recommendations = []
        if vm['eye_contact_percentage'] < 60:
            recommendations.append("👁️ <strong>Eye Contact:</strong> Practice looking directly at the camera. Imagine you're talking to a friend.")
        if vm['posture_alerts'] > self.total_frames * 0.3:
            recommendations.append("🪑 <strong>Posture:</strong> Sit up straight with shoulders back. Good posture = confidence!")
        if am['total_fillers'] > am['total_words'] * 0.1:
            recommendations.append("🗣️ <strong>Filler Words:</strong> Pause instead of using 'um' or 'like'. Silence is powerful!")
        if am['energy_score'] < 50:
            recommendations.append("⚡ <strong>Energy:</strong> Speak with more enthusiasm! Show your passion for the role.")
        if vm['smile_count'] < 5:
            recommendations.append("😊 <strong>Smile More:</strong> A genuine smile makes you likeable and confident!")
        if vm['hand_gestures'] < 10:
            recommendations.append("✋ <strong>Hand Gestures:</strong> Use natural hand movements to emphasize points.")
        
        if not recommendations:
            recommendations.append("🎉 <strong>Perfect Performance!</strong> You nailed every aspect. Keep it up!")
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Interview Coach - Performance Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .score-section {{
            background: linear-gradient(135deg, {grade_color} 0%, {grade_color}dd 100%);
            color: white;
            padding: 60px 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        
        .score-section::before {{
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: pulse 3s ease-in-out infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.1); }}
        }}
        
        .score-display {{
            font-size: 6em;
            font-weight: bold;
            margin: 20px 0;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
            position: relative;
            z-index: 1;
        }}
        
        .grade {{
            font-size: 3em;
            font-weight: bold;
            margin: 10px 0;
            position: relative;
            z-index: 1;
        }}
        
        .feedback {{
            font-size: 1.5em;
            margin-top: 20px;
            position: relative;
            z-index: 1;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }}
        
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .metric-description {{
            font-size: 0.85em;
            color: #888;
            margin-top: 8px;
        }}
        
        .recommendations {{
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        
        .recommendations ul {{
            list-style: none;
            padding: 0;
        }}
        
        .recommendations li {{
            padding: 12px 0;
            border-bottom: 1px solid #ffe69c;
            font-size: 1.05em;
            line-height: 1.6;
        }}
        
        .recommendations li:last-child {{
            border-bottom: none;
        }}
        
        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}
        
        .chart-container img {{
            max-width: 100%;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .strengths {{
            background: #d4edda;
            border-left: 5px solid #28a745;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        
        .strengths ul {{
            list-style: none;
            padding: 0;
        }}
        
        .strengths li {{
            padding: 10px 0;
            font-size: 1.05em;
        }}
        
        .strengths li::before {{
            content: '✓ ';
            color: #28a745;
            font-weight: bold;
            font-size: 1.2em;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            border-top: 3px solid #667eea;
        }}
        
        .footer p {{
            color: #666;
            margin: 5px 0;
        }}
        
        .session-info {{
            background: #e9ecef;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            font-family: monospace;
        }}
        
        .progress-bar {{
            background: #e0e0e0;
            border-radius: 10px;
            height: 30px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 1s ease;
        }}
        
        @media print {{
            body {{
                background: white;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 AI Interview Coach</h1>
            <p>Professional Performance Analysis Report</p>
        </div>
        
        <div class="score-section">
            <div class="score-display">{score}</div>
            <div class="grade">Grade: {grade}</div>
            <div class="feedback">{feedback}</div>
        </div>
        
        <div class="content">
            <div class="session-info">
                <strong>Session ID:</strong> {analysis['session_id']}<br>
                <strong>Date:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br>
                <strong>Duration:</strong> {int(analysis['duration_seconds'] // 60)}m {int(analysis['duration_seconds'] % 60)}s
            </div>
            
            <div class="section">
                <h2 class="section-title">📊 Performance Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Eye Contact</div>
                        <div class="metric-value">{vm['eye_contact_percentage']:.1f}%</div>
                        <div class="metric-description">Maintained eye contact</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {vm['eye_contact_percentage']:.0f}%"></div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Voice Clarity</div>
                        <div class="metric-value">{am['clarity_score']}/100</div>
                        <div class="metric-description">Speech quality score</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {am['clarity_score']}%"></div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Energy Level</div>
                        <div class="metric-value">{am['energy_score']}/100</div>
                        <div class="metric-description">Voice enthusiasm</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {am['energy_score']}%"></div>
                        </div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Words Spoken</div>
                        <div class="metric-value">{am['total_words']}</div>
                        <div class="metric-description">Total word count</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Filler Words</div>
                        <div class="metric-value">{am['total_fillers']}</div>
                        <div class="metric-description">Um, uh, like, etc.</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Smile Count</div>
                        <div class="metric-value">{vm['smile_count']}</div>
                        <div class="metric-description">Positive expressions</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Hand Gestures</div>
                        <div class="metric-value">{vm['hand_gestures']}</div>
                        <div class="metric-description">Natural movements</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-label">Posture Alerts</div>
                        <div class="metric-value">{vm['posture_alerts']}</div>
                        <div class="metric-description">Slouching detected</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">📈 Visual Analytics</h2>
                <div class="chart-container">
                    <img src="{chart_file}" alt="Performance Charts">
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">💡 Personalized Recommendations</h2>
                <div class="recommendations">
                    <ul>
                        {''.join([f'<li>{rec}</li>' for rec in recommendations])}
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">✨ Your Strengths</h2>
                <div class="strengths">
                    <ul>
                        {self.generate_strengths_html(vm, am)}
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">📋 Detailed Breakdown</h2>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 10px;">
                    <h3 style="color: #667eea; margin-bottom: 15px;">🎥 Visual Analysis</h3>
                    <p><strong>Total Frames Analyzed:</strong> {vm['total_frames']} frames</p>
                    <p><strong>Eye Contact Frames:</strong> {vm['eye_contact_frames']} ({vm['eye_contact_percentage']:.1f}%)</p>
                    <p><strong>Posture Quality:</strong> {100 - (vm['posture_alerts'] / max(1, vm['total_frames'] / 30) * 10):.1f}% good posture</p>
                    <p><strong>Fidgeting:</strong> {vm['fidget_count']} movements detected</p>
                    <p><strong>Head Tilts:</strong> {vm['head_tilt_count']} instances</p>
                    
                    <h3 style="color: #667eea; margin: 25px 0 15px;">🎤 Audio Analysis</h3>
                    <p><strong>Total Words:</strong> {am['total_words']} words</p>
                    <p><strong>Filler Words:</strong> {am['total_fillers']} ({(am['total_fillers'] / max(1, am['total_words']) * 100):.1f}% of speech)</p>
                    <p><strong>Most Common Fillers:</strong> {', '.join([f"{k} ({v})" for k, v in list(am['filler_count'].items())[:3]]) if am['filler_count'] else 'None detected!'}</p>
                    <p><strong>Speech Segments:</strong> {am['speech_segments']} speaking intervals</p>
                    <p><strong>Average Volume:</strong> {am['avg_volume']} (normalized)</p>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <h3 style="color: #667eea; margin-bottom: 15px;">🚀 Next Steps</h3>
            <p>Practice makes perfect! Use this coach regularly to track your improvement.</p>
            <p>Share your progress with friends and help them prepare too!</p>
            <p style="margin-top: 20px; font-size: 0.9em; color: #999;">
                Generated by AI Interview Coach Pro | Powered by Computer Vision & AI
            </p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def generate_strengths_html(self, vm, am):
        """Generate HTML for strengths section"""
        strengths = []
        
        if vm['eye_contact_percentage'] >= 70:
            strengths.append("<li>Excellent eye contact - you connect well with the audience!</li>")
        if vm['posture_alerts'] < self.total_frames * 0.2:
            strengths.append("<li>Great posture - you appear confident and professional!</li>")
        if vm['smile_count'] >= 5:
            strengths.append("<li>Positive demeanor - your smiles make you approachable!</li>")
        if am['clarity_score'] >= 75:
            strengths.append("<li>Clear speech - minimal filler words shows preparation!</li>")
        if am['energy_score'] >= 60:
            strengths.append("<li>Good energy level - you sound enthusiastic and engaged!</li>")
        if vm['hand_gestures'] >= 10 and vm['excessive_gestures'] < 5:
            strengths.append("<li>Natural gesturing - your hand movements enhance communication!</li>")
        if am['total_words'] >= 50:
            strengths.append("<li>Articulate communicator - you expressed yourself well!</li>")
        
        if not strengths:
            strengths.append("<li>You completed the session - that's the first step to improvement!</li>")
        
        return ''.join(strengths)
    
    def print_console_summary(self, analysis):
        """Print beautiful console summary"""
        vm = analysis['visual_metrics']
        am = analysis['audio_metrics']
        score = analysis['final_confidence_score']
        
        print("\n" + "="*70)
        print("🎯 AI INTERVIEW COACH PRO - PERFORMANCE REPORT".center(70))
        print("="*70)
        print(f"\n{'FINAL CONFIDENCE SCORE':^70}")
        print(f"{'⭐' * (score // 20):^70}")
        print(f"{score}/100".center(70))
        print("\n" + "-"*70)
        
        print("\n📊 KEY METRICS:")
        print(f"  • Eye Contact: {vm['eye_contact_percentage']:.1f}%")
        print(f"  • Voice Clarity: {am['clarity_score']}/100")
        print(f"  • Energy Level: {am['energy_score']}/100")
        print(f"  • Words Spoken: {am['total_words']}")
        print(f"  • Filler Words: {am['total_fillers']}")
        print(f"  • Smiles: {vm['smile_count']}")
        print(f"  • Hand Gestures: {vm['hand_gestures']}")
        
        print("\n💡 QUICK FEEDBACK:")
        if score >= 80:
            print("  ✅ Outstanding! You're ready to ace that interview!")
        elif score >= 60:
            print("  ✅ Good job! A bit more practice and you'll be perfect!")
        else:
            print("  ✅ Keep practicing! Focus on the recommendations above.")
        
        print("\n" + "="*70)


def main():
    """Main entry point"""
    print("╔" + "="*68 + "╗")
    print("║" + "🎯 AI INTERVIEW COACH PRO".center(68) + "║")
    print("║" + "Professional Interview Preparation Tool".center(68) + "║")
    print("╚" + "="*68 + "╝")
    print("\n✨ Features:")
    print("  • Real-time video analysis (eye contact, posture, expressions)")
    print("  • Audio analysis (filler words, clarity, energy)")
    print("  • Body language tracking (gestures, fidgeting)")
    print("  • Comprehensive HTML report with charts")
    print("  • Session recording with feedback overlay")
    
    print("\n" + "-"*70)
    
    # Check dependencies
    try:
        import pyaudio
        import speech_recognition
        import matplotlib
        print("✅ All dependencies loaded successfully!")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\n📦 Install required packages:")
        print("   pip install pyaudio speechrecognition matplotlib")
        return
    
    coach = InterviewCoach()
    
    # Configuration
    print("\n⚙️  SESSION SETUP")
    record = input("Record this session with feedback overlay? (y/n): ").lower()
    
    output_file = None
    if record == 'y':
        output_file = f"interview_session_{coach.session_id}.mp4"
        print(f"✅ Video will be saved to: {output_file}")
    
    print("\n📋 TIPS FOR BEST RESULTS:")
    print("  • Sit in a well-lit area facing the camera")
    print("  • Position yourself centered in frame")
    print("  • Speak clearly as if in a real interview")
    print("  • Practice answering common interview questions")
    print("  • Press 'q' when finished to view your report")
    
    input("\n👉 Press ENTER to start your interview coaching session...")
    
    print("\n🎬 Starting in 3...")
    time.sleep(1)
    print("🎬 Starting in 2...")
    time.sleep(1)
    print("🎬 Starting in 1...")
    time.sleep(1)
    print("\n" + "="*70)
    
    coach.start_session(output_filename=output_file)


if __name__ == "__main__":
    main()