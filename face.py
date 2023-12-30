#import libraries
import cv2
import face_recognition

# Load an image for face recognition
image_of_person = face_recognition.load_image_file("me.jpg")
person_face_encoding = face_recognition.face_encodings(image_of_person)[0]

# Create arrays of known face encodings and their corresponding labels
known_face_encodings = [person_face_encoding]
known_face_labels = ["Myriam"]

# Open the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture each frame from the webcam
    ret, frame = video_capture.read()

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # If no faces are found, display a message
    if not face_locations:
        cv2.putText(frame, "Where is your face? Are you a ghost?", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Check if the face matches any known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            # If a match is found, use the label of the known face
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_labels[first_match_index]
                cv2.putText(frame, f"{name}", (int((left + right) / 2), bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Alert! stranger", (left + 6, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw rectangles around the faces
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()
