import ctypes

SCENE_IMAGES = "test_images/scene_images"
SCENE_DATA = "test_data/scene_data"

SHOW = False
SHOW_DELAY = None

if __name__ == '__main__':
    recognizer = ctypes.CDLL("cmake-build-debug/Debug/recognizer-sharedd.dll")
    recognizer.recognize()
