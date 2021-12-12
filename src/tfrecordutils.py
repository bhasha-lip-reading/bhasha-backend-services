from constants import DATA_DIR, FRAME_COUNT
import tensorflow as tf
import os


def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def createExample(frames, label):
    feature = {}
    for i in range(FRAME_COUNT):
        feature['frame-%02d' % i] = image_feature(frames[i])
    feature['label'] = int64_feature(label)

    return tf.train.Example(features=tf.train.Features(feature=feature))


def parseTFrecord(example):
    feature_description = {}
    for i in range(FRAME_COUNT):
        feature_description['frame-%02d' %
                            i] = tf.io.FixedLenFeature([], tf.string)
    feature_description['label'] = tf.io.FixedLenFeature([], tf.int64)

    example = tf.io.parse_single_example(example, feature_description)

    frames = []
    for i in range(FRAME_COUNT):
        frames.append(tf.io.decode_jpeg(example['frame-%02d' % i], channels=1))
    return frames, example['label'] - 1
