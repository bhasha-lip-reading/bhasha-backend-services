import cv2
import imageio
from tensorflow_docs.vis import embed
from constants import ASSET_DIR, DATA_DIR, INIT_WIDTH, INIT_HEIGHT, MEAN, STD, UPLOAD_DIR, WORD_PATH, TOP_K, LIP_CROP_SIZE
import numpy as np
import os


def read(filePath):
    frames = []
    capture = cv2.VideoCapture(filePath)

    # print(capture.get(cv2.CAP_PROP_FPS))
    # print(capture.isOpened())
    while True:
        success, frame = capture.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (INIT_HEIGHT, INIT_WIDTH),
                           interpolation=cv2.INTER_NEAREST)

        frame = np.expand_dims(frame, axis=-1)
        frames.append(frame)

    capture.release()

    return frames


def writeVideo(frames, size, file, save_to):
    capture = cv2.VideoCapture(file)
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    # print(fps)
    fourcc = int(cv2.VideoWriter_fourcc('m', 'p', '4', 'v'))
    writer = cv2.VideoWriter(save_to, fourcc, fps, size, 0)
    for frame in frames:
        writer.write(frame)
    writer.release()


def load_words():
    words = []
    with open(WORD_PATH, mode='r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            words.append(line.split(',')[-1][:-1])
    return words


def get_top_k_predictions(predictions, top_k=TOP_K):
    predictions = np.array(predictions)
    top_k_indices = predictions.argsort()[-top_k:][::-1]
    top_k_values = [predictions[x] for x in top_k_indices]

    return top_k_values, top_k_indices


def to_gif(images, filename):
    images = np.array(images)
    converted_images = images.astype(np.uint8)
    path = os.path.join(ASSET_DIR, '{}'.format(filename))
    imageio.mimsave(path, converted_images, fps=20)
    return embed.embed_file(path)


if __name__ == '__main__':
    words = load_words()

    with open(DATA_DIR + "/words.txt", mode='w', encoding='utf-8') as file:
        for i in range(len(words)):
            words[i] = "\"" + words[i] + "\","
            file.write(words[i])
