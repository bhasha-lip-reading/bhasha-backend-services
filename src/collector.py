from augment import augment, resizeVideo
from constants import DATA_DIR, FRAME_HEIGHT, FRAME_WIDTH, UPLOAD_DIR
from faceutils import detectLips
from utils import read, to_gif, writeVideo
from sampleutils import sample
from tfrecordutils import createExample, parseTFrecord
import os
from time import time
import tensorflow as tf
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--speaker')


def collect(filePath):
    frames = read(filePath)
    frames = detectLips(frames)

    print("Lip frames: ", len(frames))

    videos = augment(frames)

    sampleVideos = []
    for video in videos:
        sampleVideo = sample(video)
        sampleVideo = resizeVideo(sampleVideo, (FRAME_HEIGHT, FRAME_WIDTH))
        sampleVideo = np.array(sampleVideo)
        sampleVideos.append(sampleVideo)

    for i in range(len(sampleVideos)):
        to_gif(sampleVideos[i], "{:02d}.gif".format(i + 1))

    label = filePath.split('/')[-1].split('-')[0]
    speaker = filePath.split('-')[-1].split('.')[0]

    print(speaker, label)

    path = os.path.join(
        DATA_DIR, "speaker-{}-label-{}-sample-{}.tfrecords".format(speaker, label, len(sampleVideos)))

    with tf.io.TFRecordWriter(path) as writer:
        for video in sampleVideos:
            example = createExample(video, int(label))
            writer.write(example.SerializeToString())


def generateDataset(speaker):
    speaker = int(speaker)
    filePaths = sorted(glob(
        # UPLOAD_DIR + "/S{:03d}/*/*.mp4".format(speaker)
        UPLOAD_DIR + "/{:03d}/*.mp4".format(speaker)
    ))

    path = os.path.join(
        DATA_DIR, "sentence-speaker-{:03d}.tfrecords".format(speaker))

    with tf.io.TFRecordWriter(path) as writer:
        sampleCount = 0
        for filePath in tqdm(filePaths):

            # sentence dataset
            # batch = int(filePath.split('/')[-1].split('-')[1])
            # print(filePath.split('/')[-1].split('.')[0])
            label = int(filePath.split('/')[-1].split('-')[-1][:-4])
            if(label <= 400):
                continue

            frames = read(filePath)
            frames = detectLips(frames)

            if(len(frames) == 0):
                with open(DATA_DIR + '/sentence-missing.txt', mode='a+') as missfile:
                    missfile.write("\n" + filePath.split('/')[-1])
                continue

            videos = augment(frames)
            sampleVideos = []
            for video in videos:
                sampleVideo = sample(video)
                sampleVideo = resizeVideo(
                    sampleVideo, (FRAME_HEIGHT, FRAME_WIDTH))
                sampleVideo = np.array(sampleVideo)
                sampleVideos.append(sampleVideo)

            # label = (int(filePath.split('/')[-1].split('-')[1]) - 1) * \
            #     25 + int(filePath.split('/')[-1].split('-')[-1][:-4])
            # print(label)
            sampleCount += len(sampleVideos)

            # print(label)
            for video in sampleVideos:
                example = createExample(video, int(label))
                writer.write(example.SerializeToString())

    print("total samples: {}".format(sampleCount))


def fromFrame2TFRecord(speaker):
    import cv2
    from constants import INIT_WIDTH, INIT_HEIGHT

    speaker = int(speaker)
    filePaths = sorted(glob(
        UPLOAD_DIR + "/032f/*"
    ))

    path = os.path.join(
        DATA_DIR, "sentence-speaker-{:03d}.tfrecords".format(speaker))

    with tf.io.TFRecordWriter(path) as writer:
        sampleCount = 0
        for filePath in tqdm(filePaths):
            framePaths = sorted(glob(filePath + "/*.jpg"))

            frames = []
            for framePath in framePaths:
                frame = cv2.imread(framePath)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (INIT_HEIGHT, INIT_WIDTH),
                                   interpolation=cv2.INTER_NEAREST)
                frame = np.expand_dims(frame, axis=-1)
            frames.append(frame)

            label = int(filePath.split('/')[-1])
            # print(label)
            if(label <= 400):
                continue
            frames = detectLips(frames)

            if(len(frames) == 0):
                with open(DATA_DIR + '/sentence-missing.txt', mode='a+') as missfile:
                    missfile.write("\n" + filePath.split('/')[-1])
                continue

            videos = augment(frames)
            sampleVideos = []
            for video in videos:
                sampleVideo = sample(video)
                sampleVideo = resizeVideo(
                    sampleVideo, (FRAME_HEIGHT, FRAME_WIDTH))
                sampleVideo = np.array(sampleVideo)
                sampleVideos.append(sampleVideo)

            # label = (int(filePath.split('/')[-1].split('-')[1]) - 1) * \
            #     25 + int(filePath.split('/')[-1].split('-')[-1][:-4])
            # print(label)
            sampleCount += len(sampleVideos)

            # print(label)
            for video in sampleVideos:
                example = createExample(video, int(label))
                writer.write(example.SerializeToString())

        print("total samples: {}".format(sampleCount))


def extract(filePath, speaker, label):
    # filename = filePath.split('/')[-1]
    # filename = "speaker-{}-label-{}.mp4".format(speaker, label)
    # print(filename)
    # frames = read(filePath)
    # frames = detectLips(frames)
    # if(len(frames) == 0):
    #     return
    # sampleVideo = sample(np.array(frames))
    # sampleVideo = resizeVideo(sampleVideo, (FRAME_HEIGHT, FRAME_WIDTH))

    # writeVideo(frames=sampleVideo, size=(FRAME_HEIGHT, FRAME_WIDTH),
    #            file=filePath, save_to=DATA_DIR + '/testcases/' + filename)

    path = os.path.join(
        DATA_DIR, "sentence-camera-app-speaker-{}.tfrecords".format(speaker))

    with tf.io.TFRecordWriter(path) as writer:
        for file in files:
            frames = read(file)
            label = file.split('/')[-1].split('-')[-1][:-4]
            example = createExample(frames, int(label))
            writer.write(example.SerializeToString())


def video2jpg(speaker):
    files = sorted(glob(UPLOAD_DIR + '/{}/*'.format(speaker)))

    flag = False
    for i in tqdm(range(len(files))):
        label = int(files[i].split('/')[-1].split('.')[0])
        Path("./upload/032f/{:03d}".format(label)
             ).mkdir(parents=True, exist_ok=True)

        if flag == False and label != i + 1:
            print("Missing ", i + 1)
            flag = True
        subprocess.run(
            'ffmpeg -loglevel quiet -i {} ./upload/032f/{:03d}/%04d.jpg -hide_banner'.format(files[i], label), shell=True)


if __name__ == '__main__':
    # fromFrame2TFRecord('032')
    # video2jpg('032')
    # args = parser.parse_args()

    # import cv2
    # from constants import INIT_WIDTH, INIT_HEIGHT

    # frames = []
    # for file in files:
    #     frame = cv2.imread(file)
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #     frame = cv2.resize(frame, (INIT_HEIGHT, INIT_WIDTH),
    #                        interpolation=cv2.INTER_NEAREST)
    #     frame = np.expand_dims(frame, axis=-1)
    # frames.append(frame)

    # lips = detectLips(frames)
    # to_gif(lips, "utus-putus.gif")
    # extract(files, '032', 476)

    # generateDataset(27)

    # t = time()

    # speaker = 1
    # generateDataset(int(args.speaker))

    # files = sorted(glob(
    #     DATA_DIR + "/*.tfrecords"
    # ))
    # print(files)

    # raw_dataset = tf.data.TFRecordDataset(files)
    # parsed_dataset = raw_dataset.map(parseTFrecord)

    # _ = None
    # __ = None

    # for video, label in parsed_dataset.take(9):
    #     _ = video
    #     __ = label

    # print(__.numpy())
    # print(_.numpy().shape)
    # to_gif(_.numpy(), "utos.gif")

    # print('Execution time: ', time() - t)
