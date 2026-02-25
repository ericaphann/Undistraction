import cv2
import pyautogui
import tkinter as tk
import threading
import time
import mediapipe as mp
import warnings
from collections import deque

# Suppress protobuf deprecation warning (common with recent MediaPipe + protobuf)
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# ─── Setup MediaPipe Face Mesh ────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_faces=1,
    refine_landmarks=True           # Better eye/nose precision
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

# ─── Distraction Settings ─────────────────────────────────────
DISTRACTION_TIME = 4.0          # seconds before punishment (same for all directions)

# Thresholds — tune these after watching printed avg values!
LEFT_THRESHOLD  = 0.42          # nose x < this → looking left
RIGHT_THRESHOLD = 0.58          # nose x > this → looking right
UP_THRESHOLD    = 0.45          # nose y < this → looking up
DOWN_THRESHOLD  = 0.63          # nose y > this → looking down

# Smoothing: average last N frames to reduce noise/jitter
NOSE_X_HISTORY = deque(maxlen=10)
NOSE_Y_HISTORY = deque(maxlen=10)

# Timers for each direction (reset when looking center again)
left_timer_start  = None
right_timer_start = None
up_timer_start    = None
down_timer_start  = None

print("Running distraction blocker. Press ESC to quit.")

# ─── Punishment function ──────────────────────────────────────
def punish(direction="unknown"):
    print(f"→ Distracted ({direction}) → punishing!")
    
    # Quick mouse shake
    for _ in range(7):
        pyautogui.moveRel(70, 0, duration=0.07)
        pyautogui.moveRel(-70, 0, duration=0.07)
    
    # Annoying popup
    def show_popup():
        root = tk.Tk()
        root.title("HEY! FOCUS!")
        root.geometry("500x250")
        root.attributes('-topmost', True)
        root.configure(bg="#ffdddd")
        
        tk.Label(root, text=f"STOP LOOKING {direction.upper()}!",
                 font=("Arial", 24, "bold"), fg="darkred", bg="#ffdddd").pack(expand=True, pady=30)
        
        tk.Label(root, text="Get back to work!", 
                 font=("Arial", 16), fg="#333", bg="#ffdddd").pack(pady=10)
        
        tk.Button(root, text="Sorry... I'll focus", command=root.destroy,
                  font=("Arial", 14), bg="green", fg="white").pack(pady=20)
        
        root.mainloop()

    threading.Thread(target=show_popup, daemon=True).start()

# ─── Main loop ────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)  # Mirror for natural view

    # Prepare for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = face_mesh.process(frame_rgb)
    frame_rgb.flags.writeable = True

    distracted = False
    current_direction = "center"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose = face_landmarks.landmark[1]  # nose tip

            NOSE_X_HISTORY.append(nose.x)
            NOSE_Y_HISTORY.append(nose.y)

            if len(NOSE_X_HISTORY) < 6:
                # Not enough history yet → skip detection
                continue

            avg_x = sum(NOSE_X_HISTORY) / len(NOSE_X_HISTORY)
            avg_y = sum(NOSE_Y_HISTORY) / len(NOSE_Y_HISTORY)

            # Optional live tuning print (uncomment when adjusting thresholds)
            # print(f"Avg nose → x: {avg_x:.3f} | y: {avg_y:.3f}")

            current_time = time.time()

            # If mostly centered → reset all timers
            if (LEFT_THRESHOLD < avg_x < RIGHT_THRESHOLD) and (UP_THRESHOLD < avg_y < DOWN_THRESHOLD):
                left_timer_start = right_timer_start = up_timer_start = down_timer_start = None
                current_direction = "center"
            else:
                # Check directions one by one (prioritize if multiple barely hit)
                if avg_x < LEFT_THRESHOLD:
                    current_direction = "left"
                    if left_timer_start is None:
                        left_timer_start = current_time
                    if current_time - left_timer_start >= DISTRACTION_TIME:
                        distracted = True

                elif avg_x > RIGHT_THRESHOLD:
                    current_direction = "right"
                    if right_timer_start is None:
                        right_timer_start = current_time
                    if current_time - right_timer_start >= DISTRACTION_TIME:
                        distracted = True

                elif avg_y < UP_THRESHOLD:
                    current_direction = "up"
                    if up_timer_start is None:
                        up_timer_start = current_time
                    if current_time - up_timer_start >= DISTRACTION_TIME:
                        distracted = True

                elif avg_y > DOWN_THRESHOLD:
                    current_direction = "down"
                    if down_timer_start is None:
                        down_timer_start = current_time
                    if current_time - down_timer_start >= DISTRACTION_TIME:
                        distracted = True

            # Draw the face mesh (helps visualize)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Show direction label on frame
            color = (0, 255, 0) if current_direction == "center" else (0, 0, 255)
            cv2.putText(frame, f"Looking: {current_direction.upper()}", 
                        (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)

    # Trigger punishment only if distracted long enough
    if distracted:
        threading.Thread(target=punish, args=(current_direction,), daemon=True).start()
        # Reset all timers after punishment to avoid spam
        left_timer_start = right_timer_start = up_timer_start = down_timer_start = None

    # Show the result
    cv2.imshow("Distraction Blocker – Stay Focused!", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

# ─── Cleanup ──────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
face_mesh.close()

print("Session ended.")