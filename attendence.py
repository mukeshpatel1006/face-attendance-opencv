# THIS IS THE FINAL CODE FOR VIDEO

import cv2
import face_recognition
import os
import math
import numpy
from datetime import datetime

# ensure Attendance.csv exists (empty file)
if not os.path.exists("Attendance.csv"):
    open("Attendance.csv", "w").close()

# OPEN WEBCAM
videoCapture = cv2.VideoCapture(0)

# FOR MARKING ATTENDANCE (only once per regNumber)
def markAttendance(regNumber, name):
    with open("Attendance.csv", "r+") as f:
        data = f.readlines()
        regNumberList = []
        for line in data:
            line = line.strip()
            if not line:
                continue
            entry = line.split(',')
            regNumberList.append(entry[0])

        # write only if this regNumber not already present
        if regNumber not in regNumberList:
            timeFormat = datetime.now().strftime('%H:%M:%S')
            f.write(f'\n{regNumber},{name},{timeFormat}')
            print(f"Attendance Marked: {regNumber}, {name}, {timeFormat}")

# FOR CHECKING THE ACCURACY
def getAccuracy(faceDistance, faceMatchThreshold = 0.6):
    if faceDistance > faceMatchThreshold:
        range_val = (1.0 - faceMatchThreshold)
        linearValue = (1.0 - faceDistance) / (range_val * 2.0)
        return linearValue
    else:
        range_val = faceMatchThreshold
        linearValue = 1.0 - (faceDistance / (range_val * 2.0))
        return linearValue + ((1.0 - linearValue) * math.pow((linearValue - 0.5) * 2, 0.2))

# FOR GETTING PATH, NAME, REGISTRATION NO. AND ENCODINGS OF EACH PERSON
allPaths = os.listdir("imageData")
allNames = []
allRegNumbers = []
allEncodings = []

for index in range(len(allPaths)):
    file_name = allPaths[index]
    # skip non-image files
    if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    allNames.append(file_name.split(".")[0])        # Name
    allRegNumbers.append(file_name.split(".")[1])   # RegNumber

    image = face_recognition.load_image_file(os.path.join("imageData", file_name))
    encodings = face_recognition.face_encodings(image)
    if len(encodings) == 0:
        print("No face found in image:", file_name)
        continue

    temp = encodings[0]
    allEncodings.append(temp)

print("Loaded known faces:", list(zip(allNames, allRegNumbers)))

while True:
    ret, frame = videoCapture.read()
    if not ret:
        print("Failed to read from webcam.")
        break

    # resize for better view
    frame = cv2.resize(frame, (0, 0), fx=2, fy=1.6)

    # smaller frame for faster processing
    resizedFrame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)

    requiredFrame = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2RGB)

    faceLocation = face_recognition.face_locations(requiredFrame)
    faceEncoding = face_recognition.face_encodings(requiredFrame, faceLocation)

    faceNames = []
    accuracy = 0.0  # default

    for encoding in faceEncoding:
        ismatched = face_recognition.compare_faces(allEncodings, encoding)
        matchedName = "Unknown"

        faceDistance = face_recognition.face_distance(allEncodings, encoding)

        # ✅ FIXED: use minimum distance (works even if only 1 known face)
        minimumFaceDistance = min(faceDistance)

        accuracy = getAccuracy(minimumFaceDistance) * 100

        bestMatchIndex = numpy.argmin(faceDistance)

        #faceCoordinates = list(i*5 for i in faceLocation[0])

        if ismatched[bestMatchIndex] and accuracy > 80:
            matchedName = allNames[bestMatchIndex]
            regNo = allRegNumbers[bestMatchIndex]
            markAttendance(regNo, matchedName)
            #cv2.putText(frame, "%.2f"%accuracy + "%", (faceCoordinates[3] + 6, faceCoordinates[2]+ 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)

        faceNames.append(matchedName)

    # DRAW RESULTS ON FRAME
    for (top, right, bottom, left), name in zip(faceLocation, faceNames):
        top *= 5
        right *= 5
        bottom *= 5
        left *= 5

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(frame, name, (left + 6, bottom - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)
        if accuracy > 80:
            cv2.putText(frame, "%.2f" % accuracy + "%", (left + 6, bottom + 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)

        #cv2.rectangle(frame, (faceCoordinates[3], faceCoordinates[0]), (faceCoordinates[1], faceCoordinates[2]), (0, 255, 0), 3)
        #cv2.putText(frame, matchedName, (faceCoordinates[3] + 6, faceCoordinates[2] - 10), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow("Recording video", frame)

    # ✅ Press 'q' to quit cleanly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()
print("Video stopped.")
