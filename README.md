# Face Recognition Based Attendance System

This project uses **OpenCV** and **face_recognition** in Python to take attendance automatically using face recognition from a webcam.

## Features

- Detects faces from webcam in real time  
- Recognizes registered students using face recognition  
- Marks attendance in `Attendance.csv` with:
  - Registration Number
  - Name
  - Time (HH:MM:SS)
- Each student is marked only once per run (based on Registration Number)

## How It Works

1. All known faces are stored in the `imageData/` folder.
2. Each image file is named as:

   ```text
   Name.RegNo.jpg
<img width="1920" height="1080" alt="Screenshot 2025-12-07 120028" src="https://github.com/user-attachments/assets/85d58d2e-af76-46b2-88a4-11b854106a1e" />
<img width="1879" height="144" alt="Screenshot 2025-12-07 120142" src="https://github.com/user-attachments/assets/77ed6be5-30a2-40cc-9408-4c6b2e93697e" />
