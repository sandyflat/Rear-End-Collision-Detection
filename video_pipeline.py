import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import cv2
import queue
import time

from vehicle_detection import detect_vehicles
from vehicle_tracking import VehicleTracker
from speed_estimation import SpeedEstimator
from lane_detection import draw_reverse_parking_lane


class VideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Collision Detection and Awareness System")
        self.root.geometry("800x600")

        self.video_path = None
        self.cap = None
        self.playing = False
        self.paused = False

        self.frame_width = 760
        self.frame_height = 480

        # Queues for threading
        self.frame_queue = queue.Queue(maxsize=5)
        self.output_queue = queue.Queue(maxsize=5)

        # UI Elements
        self.display_frame = tk.Frame(root, width=self.frame_width, height=self.frame_height, bg="black", bd=2, relief=tk.SOLID)
        self.display_frame.pack(pady=10)
        self.display_frame.pack_propagate(0)

        self.display_label = tk.Label(self.display_frame, bg="black", fg="white", text="No video selected", font=("Arial", 16))
        self.display_label.pack(expand=True, fill=tk.BOTH)

        self.bottom_frame = tk.Frame(root)
        self.bottom_frame.pack(side=tk.BOTTOM, anchor='w', padx=10, pady=10)

        self.btn_choose = tk.Button(self.bottom_frame, text="Choose Video", command=self.choose_video)
        self.btn_choose.pack(side=tk.LEFT, padx=5)

        self.btn_play = tk.Button(self.bottom_frame, text="Play", command=self.play_video, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT, padx=5)

        self.btn_pause_resume = tk.Button(self.bottom_frame, text="Pause", command=self.pause_resume_video, state=tk.DISABLED)
        self.btn_pause_resume.pack(side=tk.LEFT, padx=5)

        self.stop_event = threading.Event()

        # Vehicle pipeline components
        self.tracker = VehicleTracker()
        self.speed_estimator = SpeedEstimator(fps=30, my_speed_km_ph=30)

    def stop_video(self):
        if self.playing:
            self.stop_event.set()
            self.playing = False
            self.paused = False
            if self.cap:
                self.cap.release()
                self.cap = None
            with self.frame_queue.mutex:
                self.frame_queue.queue.clear()
            with self.output_queue.mutex:
                self.output_queue.queue.clear()
            self.display_label.config(image='', text="No video selected")
            self.btn_play.config(state=tk.NORMAL)
            self.btn_pause_resume.config(state=tk.DISABLED)
            self.btn_pause_resume.config(text="Pause")

    def choose_video(self):
        self.stop_video()
        file_path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.video_path = file_path
            self.btn_play.config(state=tk.NORMAL)
            self.display_label.config(text="")

    def play_video(self):
        if not self.video_path:
            return

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open video file.")
            return

        self.playing = True
        self.paused = False
        self.stop_event.clear()
        self.btn_play.config(state=tk.DISABLED)
        self.btn_pause_resume.config(state=tk.NORMAL)
        self.btn_pause_resume.config(text="Pause")


        self.tracker = VehicleTracker()
        self.speed_estimator = SpeedEstimator(fps=30, my_speed_km_ph=30)  # Fixed at 30 FPS

        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
        with self.output_queue.mutex:
            self.output_queue.queue.clear()

        threading.Thread(target=self.read_frames, daemon=True).start()
        threading.Thread(target=self.process_frames, daemon=True).start()
        self.update_frame_ui()

    def pause_resume_video(self):
        if self.playing:
            self.paused = not self.paused
            self.btn_pause_resume.config(text="Resume" if self.paused else "Pause")

    def read_frames(self):
        while not self.stop_event.is_set():
            if self.paused:
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                self.stop_event.set()
                break
            try:
                self.frame_queue.put(frame, timeout=1)
            except queue.Full:
                pass
            time.sleep(0.01)

        if self.cap:
            self.cap.release()
            self.cap = None

    def process_frames(self):
        while not self.stop_event.is_set() or not self.frame_queue.empty():
            if self.paused:
                time.sleep(0.1)
                continue

            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            frame = cv2.resize(frame, (self.frame_width, self.frame_height))

            # Draw lane zones
            lane_zones = draw_reverse_parking_lane(frame)

            # Vehicle detection
            annotated_frame, boxes = detect_vehicles(frame, lane_zones)

            # Tracking
            tracks = self.tracker.update_tracks(boxes, frame)

            for track_id, x1, y1, x2, y2 in tracks:
                self.speed_estimator.update(track_id, y1, y2)
                speed = self.speed_estimator.compute_speed(track_id)
                if speed:
                    cv2.putText(annotated_frame, f"Speed: {speed} km/h", (x1, y2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            self.speed_estimator.next_frame()

            try:
                self.output_queue.put(annotated_frame, timeout=1)
            except queue.Full:
                pass

    def update_frame_ui(self):
        if not self.playing and self.output_queue.empty():
            self.btn_play.config(state=tk.NORMAL)
            self.btn_pause_resume.config(state=tk.DISABLED)
            self.display_label.config(text="No video selected")
            return

        try:
            frame = self.output_queue.get_nowait()
        except queue.Empty:
            frame = None

        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.display_label.imgtk = imgtk
            self.display_label.config(image=imgtk)

        self.root.after(30, self.update_frame_ui)


def start_app():
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()