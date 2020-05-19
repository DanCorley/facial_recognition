import cv2

file = '/Users/dcorley/Desktop/facial_recognition/files/IMG_7733.MOV'

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture(file)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
dimensions = (height, width)

# Check if camera opened successfully
if cap.isOpened() is False:
    print("Error opening video file")

xml = '/Users/dcorley/Desktop/facial_recognition/files/\
haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(xml)
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter('output.mp4', fourcc, fps, dimensions)

frame_num = 0
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    if ret is True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            (x, y, w, h) = face_cascade.detectMultiScale(
                gray, 1.3, minNeighbors=3, minSize=(500, 500),
                maxSize=(800, 800)
            )[0]
        except IndexError:
            x = 0
        if x == 0:
            pass
        elif not frame_num % 2:
            (x2, y2, w2, h2) = (x, y, w, h)
            img = cv2.rectangle(
                frame, (x2, y2), (x2+w2, y2+h2),
                (255, 0, 0), 3
            )
            cv2.putText(
                frame, 'PERSON', (x2, y2 - 10), font,
                1.5, (0, 0, 0), 2, cv2.LINE_AA
            )
        else:
            img = cv2.rectangle(
                frame, (x2, y2), (x2+w2, y2+h2),
                (255, 0, 0), 3
            )
            cv2.putText(
                frame, 'PERSON', (x2, y2 - 10), font,
                1.5, (0, 0, 0), 2, cv2.LINE_AA
            )
            # for (ex, ey, ew, eh) in eyes:
            #     cv2.rectangle(
            #         roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2
            #     )
        # # Display the resulting frame
        cv2.imshow('Frame', frame)
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        out.write(frame)
        frame_num += 1

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()
