import dlib
import cv2
import numpy as np
from constants import WIDTH, HEIGHT, FRAME_WIDTH, LIP_MARGIN, LIP_CROP_SIZE, DETECTOR_PATH
from sampleutils import sample


faceDetector = dlib.get_frontal_face_detector()
landmarkDetector = dlib.shape_predictor(
    DETECTOR_PATH
)


def shape2List(landmark):
    points = []
    for i in range(48, 68):
        points.append((landmark.part(i).x, landmark.part(i).y))
    return points


def detectLandmark(frames):
    landmarks = []
    for i, frame in enumerate(frames):
        faces = faceDetector(frame, 1)  # must put 1 for grayscale
        if len(faces) < 1:
            if len(landmarks) > 0:
                landmarks.append(landmarks[-1])
            continue
        for face in faces:
            landmark = landmarkDetector(frame, face)
            landmark = shape2List(landmark)
            landmarks.append(landmark)
    # assert len(landmarks) > 0, "No landmark found"
    return landmarks


def extractLip(frames, landmarks):
    lips = []
    for i, landmark in enumerate(landmarks):
        if(i >= len(frames)):
            break
        lip = landmark

        lip_x = sorted(lip, key=lambda pointx: pointx[0])
        lip_y = sorted(lip, key=lambda pointy: pointy[1])

        # print(len(lip_x), len(lip_y))

        x_add = int((-lip_x[0][0]+lip_x[-1][0]) * LIP_MARGIN)
        y_add = int((-lip_y[0][1]+lip_y[-1][1]) * LIP_MARGIN)

        crop_pos = (lip_x[0][0]-x_add, lip_x[-1][0]+x_add,
                    lip_y[0][1]-y_add, lip_y[-1][1]+y_add)

        cropped = frames[i][crop_pos[2]:crop_pos[3], crop_pos[0]:crop_pos[1]]
        if(cropped.shape[0] == 0 or cropped.shape[1] == 0 or cropped.shape[2] == 0):
            continue
        cropped = cv2.resize(
            cropped, (LIP_CROP_SIZE[0], LIP_CROP_SIZE[1]), interpolation=cv2.INTER_CUBIC)
        lips.append(cropped)
    return lips


def detectLips(frames):
    landmarks = detectLandmark(frames)
    lips = extractLip(frames, landmarks)
    for i in range(len(lips)):
        lips[i] = cv2.resize(lips[i], (WIDTH, HEIGHT))
        lips[i] = np.expand_dims(lips[i], axis=-1)
    return lips


def transform(frames):
    landmarks = detectLandmark(frames)
    lips = extractLip(frames, landmarks)

    # if len(lips) == 0:
    #     print("Error detecting lips")

    lips = sample(lips)
    for i in range(len(lips)):
        lips[i] = cv2.resize(lips[i], (HEIGHT, WIDTH))
        lips[i] = np.expand_dims(lips[i], axis=-1)
    return np.array(lips)
