import face_recognition
import cv2


def compare_faces_shlee_v(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    """
    Compare a list of face encodings against a candidate encoding to see if they match.

    :param known_face_encodings: A list of known face encodings
    :param face_encoding_to_check: A single face encoding to compare against the list
    :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
    :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
    """
    return list(face_recognition.face_distance(known_face_encodings, face_encoding_to_check) <= tolerance), list(face_recognition.face_distance(known_face_encodings, face_encoding_to_check))

# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file
input_movie = cv2.VideoCapture("qwer_example_video.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output_qwer.avi', fourcc, 29.97, (1280, 720))

# Load some sample pictures and learn how to recognize them.
#lmm_image = face_recognition.load_image_file("lin-manuel-miranda.png")
chodan_image = face_recognition.load_image_file("chodan.jpg")
chodan_face_encoding = face_recognition.face_encodings(chodan_image, known_face_locations=None, num_jitters=5, model="large")[0]

majenta_image = face_recognition.load_image_file("majenta.jpg")
majenta_face_encoding = face_recognition.face_encodings(majenta_image, known_face_locations=None, num_jitters=5, model="large")[0]

hina_image = face_recognition.load_image_file("hina.jpg")
hina_face_encoding = face_recognition.face_encodings(hina_image, known_face_locations=None, num_jitters=5, model="large")[0]

siyeon_image = face_recognition.load_image_file("siyeon.jpg")
siyeon_face_encoding = face_recognition.face_encodings(siyeon_image, known_face_locations=None, num_jitters=5, model="large")[0]

known_faces = [
    chodan_face_encoding,
    majenta_face_encoding,
    hina_face_encoding,
    siyeon_face_encoding
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    #rgb_frame = frame[:, :, ::-1]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match, score = compare_faces_shlee_v(known_faces, face_encoding, tolerance=0.55)

        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo
        name = None
        if any(match):
            print(score)
            index_of_largest = score.index(min(score))
            
            if index_of_largest == 0:
                name = "Chodan"
            elif index_of_largest == 1:
                name = "Majenta"
            elif index_of_largest == 2:
                name = "Hina"
            elif index_of_largest == 3:
                name = "Siyeon"      
        
        # if match[0]:
        #     #name = "Lin-Manuel Miranda"
        #     name = "Chodan"
        # elif match[1]:
        #     #name = "Alex Lacamoire"
        #     name = "Majenta"
        # elif match[2]:
        #     #name = "Alex Lacamoire"
        #     name = "Hina"
        # elif match[3]:
        #     #name = "Alex Lacamoire"
        #     name = "Siyeon"
        print(name)
        face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()
