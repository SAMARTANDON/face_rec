import face_recognition
import os
import  cv2
import numpy as np
import pandas as pd
from datetime import datetime

def load_known_faces(folder_path):
    
    known_encodings = []
    known_names = []
 
    for filename in os.listdir(folder_path):
        
        image_path = os.path.join(folder_path, filename)
        image = face_recognition.load_image_file(image_path)
 
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            print(f"Warning: no face found in {filename}, skipping.")
            continue
 
        # take the first face found in the image
        known_encodings.append(encodings[0])
        # use the filename (without extension) as the person's name
        name = os.path.splitext(filename)[0]
        known_names.append(name)
 
        print(f"Loaded: {name}")
 
    return known_encodings, known_names

known_face_encodings,known_face_rolls=load_known_faces('/home/samar/face_rec/pics')

# my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!
face_locations = []
face_encodings = []
face_rolls = []
process_this_frame = True

video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_rolls[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_rolls[best_match_index]
            face_rolls.append(name)

        # Draw a box around the face
        cv2.rectangle(frame, (left-30, top-30), (right+30, bottom+30), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left-30, bottom -5), (right+30, bottom+30), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left - 24, bottom + 24), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) == 27:  
        break

video_capture.release()
cv2.destroyAllWindows()
df = pd.read_csv("attendence_list.csv")

time = datetime.now().strftime("%Y-%m-%d %H")
if time not in df.columns:
    df[time] = 0

face_rolls=list(set(face_rolls))
print(face_rolls)

for i, roll in enumerate(df["roll"]):
    if roll in face_rolls:
        df.at[i, time] = 1

df.to_csv("attendence_list.csv",index=False)