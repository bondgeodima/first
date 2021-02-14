import cv2

VIDEO_SOURCE = "C://Users//Administrator//Mask_RCNN//car_video//out.mp4"

video_capture = cv2.VideoCapture(VIDEO_SOURCE)

while True:
    success, frame = video_capture.read()
    if not success:
        print('break')
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_image = frame[:, :, ::-1]

    # Show the frame of video on the screen
    cv2.imshow('Video', rgb_image)

    # Hit 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

