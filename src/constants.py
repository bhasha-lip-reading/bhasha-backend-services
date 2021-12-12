import os

ROOT = '/Users/bhuiyans/Documents/bhasha-backend-services'
ASSET_DIR = os.path.join(ROOT, 'asset')
DATA_DIR = os.path.join(ROOT, 'data')
MODELS_DIR = os.path.join(ROOT, 'models')
UPLOAD_DIR = os.path.join(ROOT, 'upload')

MODEL_NAME = 'sentence-simple-cnn-normalized-2'

DETECTOR_PATH = os.path.join(ASSET_DIR, 'landmarks.dat')
WORD_PATH = os.path.join(ASSET_DIR, 'vocabs.txt')
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)
GIF_PATH = os.path.join(ASSET_DIR, 'utos.gif')

LIP_MARGIN = 0.3
(HEIGHT, WIDTH) = (320, 320)
(INIT_WIDTH, INIT_HEIGHT) = (480, 360)
LIP_CROP_SIZE = (480, 360)
FRAME_HEIGHT = 48
FRAME_WIDTH = 48
FRAMES_PER_VIDEO = 35
TOP_K = 10
CROP_SIZE = (300, 300)
FRAME_COUNT = 35
(MEAN, STD) = (0.421, 0.165)
ALLOWED_EXTENSIONS = {'mp4'}
