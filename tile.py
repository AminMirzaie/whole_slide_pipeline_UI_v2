from PyQt5 import QtCore, QtGui
import math
import os
import PIL.Image
import io
import xml.dom.minidom
import time
import numpy as np
import concurrent.futures
import cv2
import tifffile
NS_DEEPZOOM = 'http://schemas.microsoft.com/deepzoom/2008'

DEFAULT_RESIZE_FILTER = PIL.Image.ANTIALIAS
DEFAULT_IMAGE_FORMAT = 'jpg'

RESIZE_FILTERS = {
    'cubic': PIL.Image.CUBIC,
    'bilinear': PIL.Image.BILINEAR,
    'bicubic': PIL.Image.BICUBIC,
    'nearest': PIL.Image.NEAREST,
    'antialias': PIL.Image.ANTIALIAS,
    }
IMAGE_FORMATS = {
    'jpg': 'jpg',
    'png': 'png',
    }
def _clamp(val, min, max):
    if val < min:
        return min
    elif val > max:
        return max
    return val

def safe_open(path):
    with open(path, 'rb') as f:
        return io.BytesIO(f.read())

def _get_or_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def _get_files_path(path):
    return os.path.splitext(path)[0] + '_files'

class DeepZoomImageDescriptor(object):
    def __init__(self, width=None, height=None,
                 tile_size=254, tile_overlap=1, tile_format='jpg'):
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self._num_levels = None

    def save(self, destination):
        with open(destination, 'w') as f:
            doc = xml.dom.minidom.Document()
            image = doc.createElementNS(NS_DEEPZOOM, 'Image')
            image.setAttribute('xmlns', NS_DEEPZOOM)
            image.setAttribute('TileSize', str(self.tile_size))
            image.setAttribute('Overlap', str(self.tile_overlap))
            image.setAttribute('Format', 'jpg')
            size = doc.createElementNS(NS_DEEPZOOM, 'Size')
            size.setAttribute('Width', str(self.height))
            size.setAttribute('Height', str(self.width))
            image.appendChild(size)
            doc.appendChild(image)
            descriptor = doc.toxml()
            f.write(descriptor)

    @property
    def num_levels(self):
        if self._num_levels is None:
            max_dimension = max(self.width, self.height)
            self._num_levels = int(math.ceil(math.log(max_dimension, 2))) + 1
        return self._num_levels

    def get_scale(self, level):
        assert 0 <= level and level < self.num_levels, 'Invalid pyramid level'
        max_level = self.num_levels - 1
        return math.pow(0.5, max_level - level)

    def get_dimensions(self, level):
        assert 0 <= level and level < self.num_levels, 'Invalid pyramid level'
        scale = self.get_scale(level)
        width = int(math.ceil(self.width * scale))
        height = int(math.ceil(self.height * scale))
        return (width, height)

    def get_num_tiles(self, level):
        assert 0 <= level and level < self.num_levels, 'Invalid pyramid level'
        w, h = self.get_dimensions( level )
        return (int(math.ceil(float(w) / self.tile_size)),
                int(math.ceil(float(h) / self.tile_size)))

    def get_tile_bounds(self, level, row, column):
        assert 0 <= level and level < self.num_levels, 'Invalid pyramid level'
        offset_x = 0 if column == 0 else self.tile_overlap
        offset_y = 0 if row == 0 else self.tile_overlap
        x = (row * self.tile_size) - offset_y
        y = (column * self.tile_size) - offset_x
        level_width, level_height = self.get_dimensions(level)
        w = self.tile_size + (1 if row == 0 else 2) * self.tile_overlap
        h = self.tile_size + (1 if column == 0 else 2) * self.tile_overlap
        w = min(w, level_width  - x)
        h = min(h, level_height - y)
        return (x, y, x + w, y + h)


class ImageCreator(object):
    def __init__(self, tile_size=254, tile_overlap=1):
        self.tile_size = int(tile_size)
        self.tile_overlap = _clamp(int(tile_overlap), 0, 10)

    def tiles(self, level):
        rows, columns = self.descriptor.get_num_tiles(level)
        for row in range(rows):
            for column in range(columns):
                yield (row, column)

    def get_image_first(self, level):
        assert 0 <= level and level < self.descriptor.num_levels, 'Invalid pyramid level'
        width, height = self.descriptor.get_dimensions(level)
        if self.descriptor.width == width and self.descriptor.height == height:
            return self.image
        return cv2.resize(self.image, (height, width), interpolation=cv2.INTER_AREA)

    def get_image(self, level, old_img, first):
        assert 0 <= level and level < self.descriptor.num_levels, 'Invalid pyramid level'
        width, height = self.descriptor.get_dimensions(level)
        if self.descriptor.width == width and self.descriptor.height == height:
            return self.image
        if first:
            return self.get_image_first(level)
        else:
            return cv2.resize(old_img, (height, width), interpolation=cv2.INTER_AREA)

    def create(self, image, source, destination, thread_number):
        self.parts = []
        self.image = image
        width, height, channels = self.image.shape
        self.descriptor = DeepZoomImageDescriptor(width=width,
                                                  height=height,
                                                  tile_size=self.tile_size,
                                                  tile_overlap=self.tile_overlap)
        self.image_files = _get_or_create_path(_get_files_path(destination))
        for level in range(self.descriptor.num_levels):
            for (row, column) in self.tiles(level):
                self.parts.append([level, row, column])
        self.descriptor.save(destination)
        self.parts.reverse()
        self.processes = np.array_split(np.array(self.parts), thread_number)
        img_nums = list(range(thread_number))
        t1 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_number) as executor:
            results = executor.map(self.process_thread, img_nums)
        concurrent.futures.as_completed(results)
        t2 = time.perf_counter()
        print(t2 - t1)

    def process_thread(self, img_num):
        data = self.processes[img_num]
        for d in range(data.shape[0]):
            level = data[d, 0]
            row = data[d, 1]
            column = data[d, 2]
            if d == 0:
                old_level = level
                level_dir = _get_or_create_path(os.path.join(self.image_files, str(level)))
                level_image = self.get_image(level, None, True)
                old_image = level_image

                bounds = self.descriptor.get_tile_bounds(level, row, column)
                tile = level_image[bounds[0]:bounds[2], bounds[1]:bounds[3], :]
                format = 'jpg'
                tile_path = os.path.join(level_dir,
                                         '{}_{}.{}'.format(column, row, format))
                cv2.imwrite(tile_path, tile[:, :, ::-1])
            else:
                if old_level != level:
                    level_dir = _get_or_create_path(os.path.join(self.image_files, str(level)))
                    level_image = self.get_image(level, old_image, False)
                    old_image = level_image

                    bounds = self.descriptor.get_tile_bounds(level, row, column)
                    tile = level_image[bounds[0]:bounds[2], bounds[1]:bounds[3], :]
                    format = 'jpg'
                    tile_path = os.path.join(level_dir,
                                             '{}_{}.{}'.format(column, row, format))
                    cv2.imwrite(tile_path, tile[:, :, ::-1])
                else:
                    bounds = self.descriptor.get_tile_bounds(level, row, column)
                    tile = level_image[bounds[0]:bounds[2], bounds[1]:bounds[3], :]
                    format = 'jpg'
                    tile_path = os.path.join(level_dir,
                                             '{}_{}.{}'.format(column, row, format))
                    cv2.imwrite(tile_path, tile[:, :, ::-1])

            old_level = level


class tile_worker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    finished2 = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(str)
    def __init__(self, source_path, des):
        super().__init__()
        self.des = des
        self.source_path = source_path

    def run(self):
        self.progress.emit("processing tile of image ... \n")
        self.progress.emit("please wait!\n")
        source_image = tifffile.imread(self.source_path)
        creator = ImageCreator(tile_size=256, tile_overlap=1.0)
        creator.create(source_image, source_image, self.des, 20)
        self.progress.emit("tile images created!\n")
        self.progress.emit("tile image directory: "+self.des+"\n")
        self.finished.emit()
        self.finished2.emit()