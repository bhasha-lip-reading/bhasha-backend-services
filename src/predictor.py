from os import abort
from augment import resizeVideo
import tensorflow as tf
import numpy as np
import cv2
from constants import ASSET_DIR, MODEL_PATH, FRAME_HEIGHT, FRAME_WIDTH, TOP_K, MEAN, STD
from faceutils import detectLandmark, detectLips, extractLip, transform
from sampleutils import sample
from utils import read, to_gif, get_top_k_predictions, load_words

predictor = tf.keras.models.load_model(MODEL_PATH)


def predict(file):
    frames = read(file)
    lips = detectLips(frames)
    # print(len(lips))
    # if(len(lips) == 0):
    #     abort(500)
    lips = sample(np.array(lips))
    # print(len(lips), lips[0].shape)
    lips = resizeVideo(lips, (FRAME_HEIGHT, FRAME_WIDTH))
    to_gif(lips, "utos.gif")

    for i in range(len(lips)):
        lips[i] = lips[i] / 255.0
        lips[i] = lips[i] - MEAN
        lips[i] = lips[i] / STD

    print(lips[0])

    lips = np.array(lips)
    batch = np.expand_dims(lips, axis=0)

    print('Prediction input shape:', batch.shape)

    predictions = predictor.predict(batch)[0]

    top_k_values, top_k_indices = get_top_k_predictions(predictions)

    words = load_words()

    response = {}
    for i in range(TOP_K):
        top_k_indices[i] += 400
        phrase = words[top_k_indices[i]]
        confidence = top_k_values[i]

        response["sentence{}".format(i + 1)] = str(phrase)

        response["confidence{}".format(
            i + 1)] = str('{:.2f}%'.format(confidence * 100.0))

        response["audioFile{}".format(
            i + 1)] = "{:03d}".format(top_k_indices[i] + 1)

    return response


if __name__ == '__main__':
    print(predict(ASSET_DIR + '/input.mp4'))
