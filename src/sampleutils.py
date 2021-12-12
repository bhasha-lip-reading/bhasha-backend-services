from constants import FRAMES_PER_VIDEO
import numpy as np


def downsample(framesOriginal):
    totalFrames = len(framesOriginal)
    framesToBeDeleted = totalFrames - FRAMES_PER_VIDEO
    takes = totalFrames // framesToBeDeleted
    if takes == 1:
        takes += 1

    if totalFrames == FRAMES_PER_VIDEO:
        return framesOriginal

    frames = []
    lastFrame = None
    for i in range(totalFrames):
        if (i + 1) % takes == 0:
            continue
        frames.append(framesOriginal[i])
        lastFrame = framesOriginal[i]
        if len(frames) == FRAMES_PER_VIDEO:
            break

    while len(frames) < FRAMES_PER_VIDEO:
        frames.append(lastFrame)

    if(len(frames) > FRAMES_PER_VIDEO):
        downsample(frames)

    assert len(frames) == FRAMES_PER_VIDEO, 'Downsampling fails'
    return frames


def upsample(framesOriginal):
    totalFrames = len(framesOriginal)
    upsample_rate = FRAMES_PER_VIDEO // totalFrames + 1

    frames = []
    for i in range(totalFrames):
        for j in range(upsample_rate):
            frames.append(framesOriginal[i])
    assert len(frames) >= FRAMES_PER_VIDEO, 'upsampling fails'
    return downsample(frames)


def sample(npFrames):
    frames = []
    for i in range(npFrames.shape[0]):
        frames.append(npFrames[i])

    n = len(frames)
    if n > FRAMES_PER_VIDEO:
        frames = downsample(frames)
    elif n < FRAMES_PER_VIDEO:
        frames = upsample(frames)
    assert len(frames) == FRAMES_PER_VIDEO, "Sampling fails."
    return frames
