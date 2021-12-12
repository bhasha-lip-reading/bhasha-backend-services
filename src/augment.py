import random
import numpy as np
import cv2
from constants import CROP_SIZE
from skimage.util import random_noise

noises = ['gaussian', 's&p', 'speckle']


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__+'(mean={0}, std={1})'.format(self.mean, self.std)


class Rgb2Gray(object):
    """Convert image to grayscale.
    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a numpy.ndarray of shape (H x W x C) in the range [0.0, 1.0].
    """

    def rgb2gray(self, rgb):

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return np.array(gray)

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Image to be converted to gray.
        Returns:
            numpy.ndarray: grey image
        """
        frames = np.stack([self.rgb2gray(_) for _ in frames], axis=0)
        return np.expand_dims(frames, axis=-1)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        frames = np.array(frames)
        if(len(frames.shape) < 4):
            frames = np.expand_dims(frames, axis=-1)
        t, h, w, c = frames.shape
        crop_h, crop_w = self.size
        delta_w = int(round((w - crop_w))/2.)
        delta_h = int(round((h - crop_h))/2.)
        frames = frames[:, delta_h:delta_h+crop_h, delta_w:delta_w+crop_w, :]
        return frames


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        frames = np.array(frames)
        if(len(frames.shape) < 4):
            frames = np.expand_dims(frames, axis=-1)
        t, h, w, c = frames.shape
        crop_h, crop_w = self.size
        delta_w = random.randint(0, w - crop_w)
        delta_h = random.randint(0, h - crop_h)
        # print('delta_w, delta_h', delta_w, delta_h)
        frames = frames[:, delta_h:delta_h+crop_h, delta_w:delta_w+crop_w, :]
        return frames


class HorizontalFlip():
    def __init__(self, flip_ratio):
        self.flip_ratio = flip_ratio

    def __call__(self, frames):
        frames = np.array(frames)
        if(len(frames.shape) < 4):
            frames = np.expand_dims(frames, axis=-1)

        t, h, w, c = frames.shape
        if random.random() < self.flip_ratio:
            for index in range(t):
                ___ = frames[index]
                frames[index, :, :, 0] = cv2.flip(frames[index, :, :, 0], 1)
        return frames


def resizeVideo(frames, size):
    resized = []
    for i in range(len(frames)):
        frame = cv2.resize(frames[i], (size[0], size[1]),
                           interpolation=cv2.INTER_NEAREST)
        frame = np.expand_dims(frame, axis=-1)
        resized.append(frame)
    return resized


"""
Please consider adding noise and quality degradition factors
accross devices for augmentation"""


def addNoise(frames):
    noiseIndex = random.randint(0, len(noises) - 1)
    noisy = []
    for i in range(len(frames)):
        frame = frames[i]
        frame = random_noise(frame, mode=noises[noiseIndex])
        frame = (255 * frame).astype(np.uint8)
        noisy.append(frame)
    return np.array(noisy)


"""
Return converting to numpy array"""


def augment(frames):
    __ = frames
    centerCropped = CenterCrop(CROP_SIZE).__call__(frames)
    horizonRandom = RandomCrop(CROP_SIZE).__call__(
        HorizontalFlip(1.0).__call__(frames))
    randomCropped = RandomCrop(CROP_SIZE).__call__(frames)
    resized = resizeVideo(frames, CROP_SIZE)
    horizonFlipped = HorizontalFlip(1.0).__call__(resized)

    return [
        np.array(resized),
        horizonFlipped,
        horizonRandom,
        centerCropped,
        randomCropped,
        addNoise(resized),
        addNoise(horizonFlipped),
        addNoise(horizonRandom),
        addNoise(centerCropped),
        addNoise(randomCropped),
    ]
