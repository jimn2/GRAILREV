import zmq
import time
import os
import shutil
from PIL import Image
import numpy as np
from openai import OpenAI
import base64
import customtkinter as ctk
import tkinter as tk
from tkinter import scrolledtext

API_KEY = "YOUR_API_KEY_HERE"
experiment_running = False

current_des_x = None
current_des_y = None
current_des_heading = None

# -----------------------------
# Image encoding
# -----------------------------
def encode_image_base64(image_path):
    if not image_path or not isinstance(image_path, (str, bytes, os.PathLike)):
        return None
    if not os.path.isfile(image_path):
        return None
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


# -----------------------------
# GPT (text + images)
# -----------------------------
def query_gpt_with_images(prompt, before_img_path, after_img_path):
    client = OpenAI(api_key=API_KEY)

    before_b64 = encode_image_base64(before_img_path)
    after_b64 = encode_image_base64(after_img_path)

    if not before_b64 or not after_b64:
        return "[Error loading images]"

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Before:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{before_b64}"}},
            {"type": "text", "text": "After:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{after_b64}"}},
            {"type": "text", "text": prompt},
        ]
    }]

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        max_completion_tokens=200
    )
    return response.choices[0].message.content


# -----------------------------
# GPT (text only)
# -----------------------------
def query_gpt_text(prompt):
    client = OpenAI(api_key=API_KEY)
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=200
    )
    return response.choices[0].message.content


# -----------------------------
# Parse returned action label
# -----------------------------
def parse_action_label(text):
    text = text.strip().lower()
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return None


# -----------------------------
# Socket setup
# -----------------------------
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

SEND_INTERVAL = 0.05
SIMVIS_DIR = "simvis"

last_experiment_data = {
    "images": ("", ""),
    "odometry": {},
    "prompt": ""
}
recording = False

# Cleanup simvis dir
if os.path.exists(SIMVIS_DIR):
    shutil.rmtree(SIMVIS_DIR)
os.makedirs(SIMVIS_DIR, exist_ok=True)


# -----------------------------
# GUI Setup
# -----------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Husky Agent Controller")
app.geometry("1200x800")

# Sliders
sliders = []
for i in range(2):
    ctk.CTkLabel(app, text=f"Wheel {i+1} velocity").grid(row=i, column=0, padx=10, pady=5)
    slider = ctk.CTkSlider(app, from_=-5.0, to=5.0, width=400)
    slider.set(0.0)
    slider.grid(row=i, column=1, padx=10, pady=5)
    sliders.append(slider)

# Logs
live_log = scrolledtext.ScrolledText(app, width=80, height=10)
live_log.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

exp_log = scrolledtext.ScrolledText(app, width=80, height=20)
exp_log.grid(row=7, column=0, columnspan=2, padx=10, pady=10)

# Legend
legend_box = scrolledtext.ScrolledText(app, width=50, height=20)
legend_box.grid(row=0, column=2, rowspan=12, padx=10, pady=10)
legend_box.insert("end",
    "x  = world x position\n"
    "y  = world y position\n"
    "yaw = heading (rad)\n"
    "vx  = forward velocity (body)\n"
    "vy  = sideways velocity (body)\n"
    "wz  = yaw rate (body)\n"
)


# -----------------------------
# Live update loop (sliders)
# -----------------------------
def update_loop():
    global experiment_running
    global current_des_x, current_des_y, current_des_heading

    if not experiment_running:
        left_speed = sliders[0].get()
        right_speed = sliders[1].get()

        socket.send_json({
            "velocities": [left_speed, right_speed],
            "record": recording
        })
        reply = socket.recv_json()
        rs = reply["robot_state"]

    else:
        # During experiments, just poll simulator state
        socket.send_json({})
        reply = socket.recv_json()
        rs = reply["robot_state"]

        # We are not using wheel velocities during experiments
        left_speed = 0.0
        right_speed = 0.0

    # Compute errors (safe wrapping for heading)
    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    x_err = None if current_des_x is None else (current_des_x - rs["x"])
    y_err = None if current_des_y is None else (current_des_y - rs["y"])
    yaw_err = None if current_des_heading is None else wrap(current_des_heading - rs["yaw"])

    text = ""
    text += f"ACTUAL:  x={rs['x']:.3f},  y={rs['y']:.3f},  yaw={rs['yaw']:.3f}\n"
    text += f"DESIRED: x={current_des_x},  y={current_des_y},  head={current_des_heading}\n"
    text += f"ERROR:   dx={x_err},  dy={y_err},  dheading={yaw_err}\n"

    live_log.delete("1.0", tk.END)
    live_log.insert(tk.END, text)

    app.after(int(SEND_INTERVAL * 1000), update_loop)


# ============================================================
# Waypoint experiment
# ============================================================

WAYPOINTS = [
    # Start: cell1102 mapped to (0.0, 0.0)

    # Move right: 1102 → 1103 → 1104 → 1105
    (1.5, 0.0, 0.0),
    (3.0, 0.0, 0.0),
    (4.5, 0.0, 0.0),

    # Move up: 1105 → 1005 → 0905 → 0805
    (4.5, 1.5, 1.5708),
    (4.5, 3.0, 1.5708),
    (4.5, 4.5, 1.5708),

    # Move right: 0805 → 0806
    (6.0, 4.5, 0.0),

    # Move up: 0806 → 0706 → 0606 → 0506
    (6.0, 6.0, 1.5708),
    (6.0, 7.5, 1.5708),
    (6.0, 9.0, 1.5708),

    # Move right: 0506 → 0507 → 0508 → 0509
    (7.5, 9.0, 0.0),
    (9.0, 9.0, 0.0),
    (10.5, 9.0, 0.0),

    # Move up: 0509 → 0409
    (10.5, 10.5, 1.5708),

    # Move right: 0409 → 0410 → 0411
    (12.0, 10.5, 0.0),
    (13.5, 10.5, 0.0),

    # Move up: 0411 → 0311 → 0211
    (13.5, 12.0, 1.5708),
    (13.5, 13.5, 1.5708)
]


POS_TOL = 0.05
HEADING_TOL = 0.03


def goto_heading(theta, odom):
    """Turn to heading theta using the simulator's built-in turn controller."""
    while True:
        socket.send_json({"go_to_heading": theta, "record": True})
        reply = socket.recv_json()

        rs = reply["robot_state"]

        odom["x"].append(rs["x"])
        odom["y"].append(rs["y"])
        odom["yaw"].append(rs["yaw"])
        odom["vx"].append(rs["vx"])
        odom["vy"].append(rs["vy"])
        odom["wz"].append(rs["wz"])

        diff = (theta - rs["yaw"] + np.pi) % (2*np.pi) - np.pi
        if abs(diff) < HEADING_TOL:
            return




def goto_xy(x_target, y_target, odom):
    """Drive using wheel speeds until we get close enough."""
    while True:
        socket.send_json({"velocities": [2.0, 2.0], "record": True})
        reply = socket.recv_json()

        rs = reply["robot_state"]

        odom["x"].append(rs["x"])
        odom["y"].append(rs["y"])
        odom["yaw"].append(rs["yaw"])
        odom["vx"].append(rs["vx"])
        odom["vy"].append(rs["vy"])
        odom["wz"].append(rs["wz"])

        dx = x_target - rs["x"]
        dy = y_target - rs["y"]
        dist = np.sqrt(dx*dx + dy*dy)

        if dist < POS_TOL:
            socket.send_json({"velocities": [0, 0], "record": True})
            socket.recv_json()
            return




def run_waypoint_experiment():
    global experiment_running
    experiment_running = True
    print("RUN EXPERIMENT BUTTON PRESSED")

    exp_log.delete("1.0", tk.END)

    # Reset screenshots
    socket.send_json({"reset_screenshots": True})
    socket.recv_json()

    # BEFORE screenshot
    socket.send_json({"velocities": [0,0], "record": True})
    before_reply = socket.recv_json()
    img_before = before_reply.get("screenshot_path", "")

    # ODOM ARRAY
    odom = {"x": [], "y": [], "yaw": [], "vx": [], "vy": [], "wz": []}

    # Process each waypoint
    for (wx, wy, wh) in WAYPOINTS:
        global current_des_x, current_des_y, current_des_heading
        current_des_x = wx
        current_des_y = wy
        current_des_heading = wh

        # ---- Send waypoint to simulator ----
        socket.send_json({"waypoint": {"x": wx, "y": wy, "heading": wh}})
        socket.recv_json()   # {"ack": True}

        # ---- Step until done ----
        while True:
            socket.send_json({})
            reply = socket.recv_json()

            rs = reply["robot_state"]
            done = reply["done"]

            # store odometry
            odom["x"].append(rs["x"])
            odom["y"].append(rs["y"])
            odom["yaw"].append(rs["yaw"])
            odom["vx"].append(rs["vx"])
            odom["vy"].append(rs["vy"])
            odom["wz"].append(rs["wz"])

            if done:
                break



    # AFTER screenshot
    socket.send_json({"velocities": [0,0], "record": True})
    after_reply = socket.recv_json()
    img_after = after_reply.get("screenshot_path", "")

    last_experiment_data["images"] = (img_before, img_after)
    last_experiment_data["odometry"] = odom

    # === Build GPT prompt ===
    prompt = "You output the following at 5Hz over this experiment:\n\n"
    prompt += f"x: {odom['x']}\n"
    prompt += f"y: {odom['y']}\n"
    prompt += f"yaw: {odom['yaw']}\n"
    prompt += f"vx: {odom['vx']}\n"
    prompt += f"vy: {odom['vy']}\n"
    prompt += f"wz: {odom['wz']}\n\n"

    prompt += (
        "Based on the odometry time-series and the before/after images, "
        "please name the action the robot performed.\n"
        "Return exactly one short action label (1–4 words), all lowercase.\n"
    )

    last_experiment_data["prompt"] = prompt
    exp_log.insert(tk.END, prompt)
    experiment_running = False

# -----------------------------
# Query LLM: text only
# -----------------------------
def query_text():
    prompt = last_experiment_data["prompt"]
    if not prompt:
        exp_log.insert(tk.END, "[Error: Run experiment first]\n")
        return

    resp = query_gpt_text(prompt)
    label = parse_action_label(resp)

    exp_log.delete("1.0", tk.END)
    exp_log.insert(tk.END, prompt + "\n\n=== GPT RESPONSE ===\n" + resp)

    if label:
        exp_log.insert(tk.END, f"\n\n[Action label]\n{label}\n")
    else:
        exp_log.insert(tk.END, "\n\n[Error parsing label]\n")


# -----------------------------
# Query LLM: text + images
# -----------------------------
def query_image():
    prompt = last_experiment_data["prompt"]
    img_before, img_after = last_experiment_data["images"]

    if not prompt:
        exp_log.insert(tk.END, "[Error: Run experiment first]\n")
        return

    resp = query_gpt_with_images(prompt, img_before, img_after)
    label = parse_action_label(resp)

    exp_log.delete("1.0", tk.END)
    exp_log.insert(tk.END, prompt + "\n\n=== GPT RESPONSE ===\n" + resp)

    if label:
        exp_log.insert(tk.END, f"\n\n[Action label]\n{label}\n")
    else:
        exp_log.insert(tk.END, "\n\n[Error parsing label]\n")


# -----------------------------
# Recording toggles
# -----------------------------
def record_on():
    global recording
    recording = True
    exp_log.insert(tk.END, "[Recording ENABLED]\n")

def record_off():
    global recording
    recording = False
    exp_log.insert(tk.END, "[Recording DISABLED]\n")

# -----------------------------
# Buttons
# -----------------------------
ctk.CTkButton(app, text="Start Recording", command=record_on, fg_color="green").grid(row=3, column=0)
ctk.CTkButton(app, text="Stop Recording", command=record_off, fg_color="red").grid(row=3, column=1)

ctk.CTkButton(app, text="Run Waypoint Experiment", command=run_waypoint_experiment, fg_color="blue").grid(row=4, column=0)

ctk.CTkButton(app, text="Query LLM (Text Only)", command=query_text, fg_color="orange").grid(row=6, column=0)
ctk.CTkButton(app, text="Query LLM (Text + Image)", command=query_image, fg_color="purple").grid(row=6, column=1)

# -----------------------------
# Start main loop
# -----------------------------
update_loop()
app.mainloop()
