import cv2
import mediapipe as mp
import math
import time
import collections
import numpy as np
import pandas as pd
from pathlib import Path
import threading
try:
    import winsound
except ImportError:
    winsound = None




# Initialize MediaPipe Pose and Drawing modules
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c, side="left"):
    if side == "right":
        a, c = c, a  # Flip for right arm to get comparable angle
    angle = math.degrees(math.atan2(c.y - b.y, c.x - b.x) - 
                         math.atan2(a.y - b.y, a.x - b.x))
    angle = angle % 360  # Ensure the angle is between 0 and 360.
    if angle > 180:
        angle = 360 - angle  # Fold into [0, 180] range.
    return angle


# Initialize webcam capture
cap = cv2.VideoCapture(0)
# ================= Ready Phase =================
# Goal: User must hold arms at ~90° (within 80-100°) for 3 consecutive seconds.
ready_duration_required = 3.0  # seconds required in the "ready" position
ready_start_time = None

print("Ready phase: Please hold your arms at ~90° (within 80-100°) for 3 seconds.")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror the frame if using a front-facing camera
        frame = cv2.flip(frame, 1)
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Default ready phase text
        cv2.putText(image, "Hold arms at ~90° for 3 seconds", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            # Get left and right arm angles
            left_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST])
            right_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])
            
            cv2.putText(image, f"Left Angle: {int(left_angle)} deg", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, f"Right Angle: {int(right_angle)} deg", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Check if both angles are within the desired range (80 to 100 degrees)
            if 80 <= left_angle <= 100 and 80 <= right_angle <= 100:
                if ready_start_time is None:
                    ready_start_time = time.time()
                elapsed = time.time() - ready_start_time
                remaining = int(max(0, ready_duration_required - elapsed))
                # Display countdown timer on screen
                cv2.putText(image, f"Hold for {remaining}s", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                # If the user holds the position long enough
                if elapsed >= ready_duration_required:
                    cv2.putText(image, "READY!", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
                    cv2.imshow("Ready Phase", image)
                    cv2.waitKey(500)  # Brief pause before proceeding
                    break  # Exit ready phase loop
            else:
                ready_start_time = None  # Reset the timer if condition is lost
        
        cv2.imshow("Ready Phase", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyWindow("Ready Phase")
print("Ready phase complete. Proceeding to calibration...")


# ================= Main Rep Counting Loop ===================

# Reinitialize tracking variables for both arms
reps = {"left": 0, "right": 0}
# Note: Check the initialization of 'directions'. If your movement starts from a high (lockout) position,
# you might want to initialize with "up". If not, adjust accordingly.
directions = {"left": "down", "right": "down"}
max_angle = {"left": 0, "right": 0}
min_angle = {"left": 180, "right": 180}
total_tut = {"left": 0, "right": 0}
start_time = {"left": None, "right": None}
eccentric_or_concentric = {"left": "eccentric", "right": "eccentric"}

TARGET_REPS = 12          # stop when either arm hits this

# ---------- AUDIO CUES (Windows "Beep") ----------
# Hz ,  ms      
BEEP_REP_FREQ  = 880;  BEEP_REP_DUR  = 120   # short high beep
BEEP_WARN_FREQ = 440;  BEEP_WARN_DUR = 300   # lower, longer beep

def play_rep_beep():
    if winsound:
        threading.Thread(
            target=winsound.Beep,
            args=(BEEP_REP_FREQ, BEEP_REP_DUR),
            daemon=True
        ).start()
def play_warn_beep():
    threading.Thread(
        target=winsound.Beep,
        args=(BEEP_WARN_FREQ, BEEP_WARN_DUR),
        daemon=True
    ).start()
# -------------------------------------------------



# Define desired targets for form corrections.
DESIRED_ROM = 100   # For example, each rep should span at least 100° of motion.
DESIRED_EXTENSION = 140  # Expected maximum angle at lockout.
warning_timeout = 3.0  # seconds for warning message to persist.
form_warning_time = {"left": None, "right": None}
form_warning_message = {"left": "", "right": ""}

# --- ROM stats -------------------------------------------------
best_rom   = {"left": 0, "right": 0}   # biggest ROM of any rep
rom_sum    = {"left": 0, "right": 0}   # accumulate for average
rom_count  = {"left": 0, "right": 0}   # how many reps logged

session_scores = {"left": [], "right": []}   # per-rep form scores

# ---------------------------------------------------------------



degree_symbol = chr(176)  # Alternative to Unicode for degrees

# ----------  (global-warning helper vars) ----------
SYMM_THRESHOLD = 15        # deg difference left vs right to flag asymmetry
WARNING_TIMEOUT = 3.0       # how long a global warning stays on-screen

angle_now = {"left": 0, "right": 0}   # current frame’s elbow angles
global_warn_msg  = ""                 # text to show
global_warn_time = None               # when that text was first triggered
# --------------------------------------------------------

# ----------  TEMPO / VELOCITY  set-up  ----------
# store last-frame angle & time stamp per arm
prev_angle     = {"left": None, "right": None}
prev_timestamp = {"left": None, "right": None}

# rolling list of frame-by-frame speeds for the current rep
speed_log = {"left": [], "right": []}

#– where we’ll store the finished averages
con_speed_avg = {"left": [], "right": []}   # one entry per rep
ecc_speed_avg = {"left": [], "right": []}

# speed targets  (deg/sec)  – tweak to taste
CON_MIN, CON_MAX =  60, 180      # concentric should be powerful but controlled
ECC_MIN, ECC_MAX =  30, 120      # eccentric should be slower
# -----------------------------------------------

# ----------------------score logic-------------------
def compute_form_score(rom, lockout_ok, avg_speed, phase, asymm_diff):
    score = 100.0
    # ROM
    score -= 0.5 * max(0, DESIRED_ROM - rom)
    # Lock-out
    if not lockout_ok:
        score -= 10
    # Speed (phase-specific limits)
    lo, hi = (CON_MIN, CON_MAX) if phase == "concentric" else (ECC_MIN, ECC_MAX)
    if avg_speed < lo:
        score -= 0.2 * (lo - avg_speed)
    elif avg_speed > hi:
        score -= 0.2 * (avg_speed - hi)
    # Symmetry (instantaneous diff)
    score -= 0.5 * (asymm_diff / SYMM_THRESHOLD)
    return max(0, min(100, score))
# ---------------------------------------------------------


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        break_outer = False        # reset each video-frame

        ret, frame = cap.read()
        if not ret:
            break
        
        # Fix flipped frame if using a front cam.
        frame = cv2.flip(frame, 1)
        
        # Convert image to RGB for processing and then back to BGR.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark

            # Iterate through both arms.
            for side in ["left", "right"]:
                if side == "left":
                    # For the displayed left side, use the right-side landmarks.
                    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
                    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                else:
                    # For the displayed right side, use the left-side landmarks.
                    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
                    wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

                # Calculate angle using the 'side' parameter.
                angle = calculate_angle(shoulder, elbow, wrist, side=side)
                angle_now[side] = angle      # <-- (keeps current frame angles)

                # ----- TEMPO computation -----
                now = time.time()

                # choose row for this arm
                y_offset = 100 if side == "left" else 140           

                if prev_angle[side] is not None:
                    dt = now - prev_timestamp[side]
                    if dt > 0:
                        speed = abs(angle - prev_angle[side]) / dt  # deg / s
                        speed_log[side].append(speed)

                        # live read-out
                        cv2.putText(image,
                                    f'{side[0].upper()} speed: {speed:5.1f} °/s',
                                    (350, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

                # update trackers for *next* frame (must run every iteration)
                prev_angle[side]     = angle
                prev_timestamp[side] = now
                # -----------------------------

                # Update max and min angles for ROM.
                max_angle[side] = max(max_angle[side], angle)
                min_angle[side] = min(min_angle[side], angle)

                # Rep Counting Logic:
                # If the arm is in the "up" state and the angle exceeds 80 deg, mark transition to "down" (eccentric phase).
                if angle > 80 and directions[side] == "up":
                    directions[side] = "down"
                    eccentric_or_concentric[side] = "eccentric"
                    # --- evaluate *concentric* speed we just finished ---
                    if speed_log[side]:
                        rep_avg_speed = sum(speed_log[side]) / len(speed_log[side])
                        con_speed_avg[side].append(rep_avg_speed)

                        if not (CON_MIN <= rep_avg_speed <= CON_MAX):
                            form_warning_time[side] = time.time()
                            form_warning_message[side] = (
                                f"{side.capitalize()} concentric too "
                                f"{'slow' if rep_avg_speed < CON_MIN else 'fast'}!")
                    speed_log[side].clear()           # start fresh for eccentric



                
                # When the arm is in the "down" state and the angle drops below 40 deg, count a rep.
                elif angle < 50 and directions[side] == "down":
                    directions[side] = "up"
                    reps[side] += 1
                    # --- auto-quit when target reached ---
                    if reps[side] >= TARGET_REPS:
                        print(f"Target {TARGET_REPS} reps reached on {side} arm → ending session.")
                        cap.release()
                        cv2.destroyAllWindows()
                        break_outer = True        # flag so it breaks out the outer while
                        break                     # break inner for-loop
                    finished_phase = eccentric_or_concentric[side]   # "eccentric" or "concentric"
                    eccentric_or_concentric[side] = "concentric"

                    # ---------- tempo evaluation on half-rep finish ----------
                    if speed_log[side]:                             # list still intact
                        rep_avg_speed = sum(speed_log[side]) / len(speed_log[side])

                        if finished_phase == "concentric":
                            con_speed_avg[side].append(rep_avg_speed)
                            if not (CON_MIN <= rep_avg_speed <= CON_MAX):
                                form_warning_time[side] = time.time()
                                form_warning_message[side] = (
                                    f"{side.capitalize()} concentric too "
                                    f"{'slow' if rep_avg_speed < CON_MIN else 'fast'}!")
                                play_warn_beep()                       #  <-- add beep here
                        else:  # finished_phase == "eccentric"
                            ecc_speed_avg[side].append(rep_avg_speed)
                            if not (ECC_MIN <= rep_avg_speed <= ECC_MAX):
                                form_warning_time[side] = time.time()
                                form_warning_message[side] = (
                                    f"{side.capitalize()} eccentric too "
                                    f"{'slow' if rep_avg_speed < ECC_MIN else 'fast'}!")
                                play_warn_beep()                       #  <-- add beep here

                    speed_log[side].clear()  # reset list for the *next* half-rep
                   # ---------------------------------------------------------


                    
                    # Update TUT.
                    if start_time[side]:
                        total_tut[side] += time.time() - start_time[side]
                    start_time[side] = time.time()
                    
                    play_rep_beep()           # <-- audio cue

                    print(f"{side.capitalize()} Arm Reps: {reps[side]}")
                    rep_rom = max_angle[side] - min_angle[side]
                    # >>> ROM bookkeeping
                    rom_sum[side]   += rep_rom
                    rom_count[side] += 1
                    best_rom[side]  = max(best_rom[side], rep_rom)
                    # <<< end bookkeeping

                    # ---- FORM SCORE bookkeeping (new) ----
                    asymm_diff = abs(angle_now["left"] - angle_now["right"])
                    lock_ok    = max_angle[side] >= DESIRED_EXTENSION
                    half_score = compute_form_score(
                        rom         = rep_rom,
                        lockout_ok  = lock_ok,
                        avg_speed   = rep_avg_speed,
                        phase       = finished_phase,
                        asymm_diff  = asymm_diff
                    )
                    session_scores.setdefault(side, []).append(half_score)
                    print(f"{side.capitalize()} form score: {half_score:.1f}")
                    # --------------------------------------

                    
                    # *** Form Correction Feedback (at the moment a rep is counted) ***
                    if rep_rom < DESIRED_ROM:
                        # Set warning time and message if not already set.
                        if form_warning_time[side] is None:
                            form_warning_time[side] = time.time()
                        form_warning_message[side] = f"Increase ROM on {side} arm!"
                        play_warn_beep()
                    # Check if the maximum (lockout) angle is below desired.
                    if max_angle[side] < DESIRED_EXTENSION:
                        if form_warning_time[side] is None:
                            form_warning_time[side] = time.time()
                        form_warning_message[side] = f"Extend fully on {side} arm!"
                        play_warn_beep()
        
                    # Reset current rep's max/min for the next rep.
                    max_angle[side] = angle
                    min_angle[side] = angle
    
                
                # Display Tracking Information.
                y_offset = 100 if side == "left" else 140
                cv2.putText(image, f'{side.capitalize()} Angle: {int(angle)} degrees',
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                rom = max_angle[side] - min_angle[side]
                cv2.putText(image, f'{side.capitalize()} ROM: {int(rom)} degrees',
                            (10, 300 if side == "left" else 340),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                if start_time[side]:
                    tut = time.time() - start_time[side]
                    cv2.putText(image, f'{side.capitalize()} TUT: {tut:.2f}s ({eccentric_or_concentric[side]})',
                                (10, 380 if side == "left" else 420),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # end of the inner for-side loop
            if break_outer:                 
                break                       

            # ----------  NEW  asymmetry global warning ----------
            diff = abs(angle_now["left"] - angle_now["right"])
            if diff > SYMM_THRESHOLD:
                # start / refresh the warning timer
                global_warn_msg  = "⚠  Arm asymmetry detected!"
                global_warn_time = time.time()
            elif diff <= SYMM_THRESHOLD:
                # clear immediately when symmetry restored
                global_warn_msg  = ""
                global_warn_time = None
            # -----------------------------------------------------


            # ----------  per-arm warning balloons (tempo / ROM) ----------
            for arm in ("left", "right"):
                msg = form_warning_message[arm]
                if msg:                                        # only if something to show
                    # still within timeout?
                    if time.time() - form_warning_time[arm] < warning_timeout:
                        y_pos = 430 if arm == "left" else 460  # separate rows for clarity
                        cv2.putText(image, msg,
                                    (10, y_pos),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        # timeout expired → clear
                        form_warning_message[arm] = ""
            # --------------------------------------------------------------


        
        # --- Display Overall Rep Counter ---
        # This line is added to display rep counts on the video feed.
        cv2.putText(image, f"Left Reps: {reps['left']} | Right Reps: {reps['right']}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # --- Display global asymmetry banner (if active) ---
        if global_warn_msg:
            cv2.putText(image, global_warn_msg,
                    (10, 470),                      #  adjust Y-pos if you like
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            
        
        cv2.imshow('Shoulder Press Tracker - Bilateral', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Console Summary Output
rom_left = max_angle["left"] - min_angle["left"]
rom_right = max_angle["right"] - min_angle["right"]
average_tut_left = total_tut["left"] / reps["left"] if reps["left"] > 0 else 0
average_tut_right = total_tut["right"] / reps["right"] if reps["right"] > 0 else 0

print("\n--- Session Summary ---")
print(f"Total Left Arm Reps: {reps['left']}")
print(f"Total Right Arm Reps: {reps['right']}")
print(f"Left ROM: {rom_left:.2f} degrees | Right ROM: {rom_right:.2f} degrees")
print(f"Left Average TUT: {average_tut_left:.2f}s | Right Average TUT: {average_tut_right:.2f}s")
print("Avg concentric speed (°/s):",
      {k: (sum(v)/len(v) if v else 0) for k, v in con_speed_avg.items()})
print("Avg eccentric  speed (°/s):",
      {k: (sum(v)/len(v) if v else 0) for k, v in ecc_speed_avg.items()})
print(f"Best ROM  (deg): Left {best_rom['left']:.1f} | Right {best_rom['right']:.1f}")
avg_left_rom  = rom_sum['left']  / rom_count['left']  if rom_count['left']  else 0
avg_right_rom = rom_sum['right'] / rom_count['right'] if rom_count['right'] else 0
print(f"Avg  ROM  (deg): Left {avg_left_rom:.1f} | Right {avg_right_rom:.1f}")
avg_score_left  = sum(session_scores["left"])  / len(session_scores["left"])  if session_scores["left"]  else 0
avg_score_right = sum(session_scores["right"]) / len(session_scores["right"]) if session_scores["right"] else 0
print(f"Avg Form Score   : Left {avg_score_left:.1f} | Right {avg_score_right:.1f}")
print("-----------------------")

# Collect the numbers you just printed
session_dict = {
    "date_time"          : time.strftime("%Y-%m-%d %H:%M:%S"),
    "left_reps"          : reps["left"],
    "right_reps"         : reps["right"],
    "left_rom_deg"       : round(rom_left,   1),
    "right_rom_deg"      : round(rom_right,  1),
    "left_avg_TUT_s"     : round(average_tut_left,  2),
    "right_avg_TUT_s"    : round(average_tut_right, 2),
    "left_con_speed"     : round(sum(con_speed_avg["left"])/len(con_speed_avg["left"]) if con_speed_avg["left"] else 0, 1),
    "right_con_speed"    : round(sum(con_speed_avg["right"])/len(con_speed_avg["right"]) if con_speed_avg["right"] else 0, 1),
    "left_ecc_speed"     : round(sum(ecc_speed_avg["left"])/len(ecc_speed_avg["left"]) if ecc_speed_avg["left"] else 0, 1),
    "right_ecc_speed"    : round(sum(ecc_speed_avg["right"])/len(ecc_speed_avg["right"]) if ecc_speed_avg["right"] else 0, 1),
    "left_rom_best_deg"  : round(best_rom["left"],  1),
    "right_rom_best_deg" : round(best_rom["right"], 1),
    "left_rom_avg_deg"   : round(avg_left_rom,  1),
    "right_rom_avg_deg"  : round(avg_right_rom, 1),
    "left_form_score"  : round(avg_score_left, 1),
    "right_form_score" : round(avg_score_right,1),


}
print("About to save log...")
print("Working directory:", Path.cwd())

# ----- SAVE LOG -----
script_dir = Path(__file__).resolve().parent
data_dir = script_dir / "data"
data_dir.mkdir(exist_ok=True)

log_path = data_dir / "shoulder_press_log.csv"
df = pd.DataFrame([session_dict])

if log_path.exists():
    df.to_csv(log_path, mode="a", index=False, header=False)
else:
    df.to_csv(log_path, index=False)

print(f"✅ Session saved to: {log_path}")
# Append-or-create
if log_path.exists():
    df.to_csv(log_path, mode="a", index=False, header=False)
else:
    df.to_csv(log_path, index=False)

print(f"Session saved to {log_path.resolve()}")
# -----------------------------------------