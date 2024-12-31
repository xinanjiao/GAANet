# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""
from PIL import Image
import glob
import hashlib
import math
import json
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from zipfile import ZipFile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective,resample_segments
from utils.general import (LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,segment2box,
                           segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first
from utils.rboxs_utils import poly_filter, poly2rbox

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # DPP

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {2: Image.FLIP_LEFT_RIGHT,
                  3: Image.ROTATE_180,
                  4: Image.FLIP_TOP_BOTTOM,
                  5: Image.TRANSPOSE,
                  6: Image.ROTATE_270,
                  7: Image.TRANSVERSE,
                  8: Image.ROTATE_90,
                  }.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


def create_dataloader_rgb_ir(path1,path2, imgsz, batch_size, stride, names, single_cls=False, hyp=None, augment=False, cache=False, pad=0.0,
                      rect=False, rank=-1, workers=8, image_weights=False, quad=False, prefix='', shuffle=False):
    if rect and shuffle:
        LOGGER.warning('WARNING: --rect is incompatible with DataLoader shuffle, setting shuffle=False')
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = LoadMultiModalImagesAndLabels(path1,path2, names, imgsz, batch_size,
                                      augment=augment,  # augmentation
                                      hyp=hyp,  # hyperparameters
                                      rect=rect,  # rectangular batches
                                      cache_images=cache,
                                      single_cls=single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix)

    # batch_size = min(batch_size, len(dataset))
    # nw = min([os.cpu_count() // WORLD_SIZE, batch_size if batch_size > 1 else 0, workers])  # number of workers
    # sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    # loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // WORLD_SIZE, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader


    
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=True,
                  collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn), dataset


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            # 裁剪白边
            # img0 = img0[100:612, 100:740]  # 裁剪坐标为[y0:y1, x0:x1]

            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap, s

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    # YOLOv5 local webcam dataloader, i.e. `python detect.py --source 0`
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.pipe = eval(pipe) if pipe.isnumeric() else pipe
        self.cap = cv2.VideoCapture(self.pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        ret_val, img0 = self.cap.read()
        img0 = cv2.flip(img0, 1)  # flip left-right

        # Print
        assert ret_val, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        s = f'webcam {self.count}: '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None, s

    def __len__(self):
        return 0


class LoadStreams:
    # YOLOv5 streamloader, i.e. `python detect.py --source 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP streams`
    def __init__(self, sources='streams.txt', img_size=640, stride=32, auto=True):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources) as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        self.auto = auto
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            st = f'{i + 1}/{n}: {s}... '
            if 'youtube.com/' in s or 'youtu.be/' in s:  # if source is YouTube video
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                s = pafy.new(s).getbest(preftype="mp4").url  # YouTube URL
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f"{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        LOGGER.info('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride, auto=self.auto)[0].shape for x in self.imgs])
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            LOGGER.warning('WARNING: Stream shapes differ. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap, stream):
        # Read stream `i` frames in daemon thread
        n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n % read == 0:
                success, im = cap.retrieve()
                if success:
                    self.imgs[i] = im
                else:
                    LOGGER.warning('WARNING: Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)  # re-open stream if signal was lost
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        img = [letterbox(x, self.img_size, stride=self.stride, auto=self.rect and self.auto)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None, ''

    def __len__(self):
        return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]
    # sb, t = 'labels', []  # /images/, /labels/
    # for x in img_paths:
    #     if 'visible' in x.split(os.sep):
    #         sa = 'visible'
    #     elif 'infrared' in x.split(os.sep):
    #         sa = 'infrared'
    #     t.append('txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)))
    # return t


class LoadMultiModalImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version

    """
    FQY  载入多模态数据 （RGB 和 IR）
    """

    def __init__(self, path_rgb,path_ir, cls_names, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        """
        Returns:
            Dataset.labels (list): n_imgs * array(num_gt_perimg, [cls_id, poly])
            Dataset.shapes (array): (n_imgs, [ori_img_width, ori_img_height])

            Dataset.batch_shapes (array): (n_batches, [h_rect, w_rect])
        """
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path_rgb = path_rgb
        self.path_ir = path_ir
        self.path_rgb = path_rgb

        self.albumentations = Albumentations() if augment else None
        self.cls_names = cls_names

        # if isinstance(cls_names,dict):
        #     self.cls_names = list(cls_names.values())



        try:
            f_rgb = []  # image files
            f_ir = []
            # -----------------------------  rgb   -----------------------------
            for p_rgb in path_rgb if isinstance(path_rgb, list) else [path_rgb]:
                p_rgb = Path(p_rgb)  # os-agnostic
                if p_rgb.is_dir():  # dir
                    f_rgb += glob.glob(str(p_rgb / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p_rgb.is_file():  # file
                    with open(p_rgb, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p_rgb.parent) + os.sep
                        f_rgb += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{path_rgb} does not exist')

                # -----------------------------  ir   -----------------------------
                for p_ir in path_ir if isinstance(path_ir, list) else [path_ir]:
                    p_ir = Path(p_ir)  # os-agnostic
                    if p_ir.is_dir():  # dir
                        f_ir += glob.glob(str(p_ir / '**' / '*.*'), recursive=True)
                        # f = list(p.rglob('**/*.*'))  # pathlib
                    elif p_ir.is_file():  # file
                        with open(p_ir, 'r') as t:
                            t = t.read().strip().splitlines()
                            parent = str(p_ir.parent) + os.sep
                            f_ir += [x.replace('./', parent) if x.startswith('./') else x for x in
                                      t]  # local to global path
                            # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                    else:
                        raise Exception(f'{prefix}{p_ir} does not exist')

     
            self.img_files_rgb = sorted([x.replace('/', os.sep) for x in f_rgb if x.split('.')[-1].lower() in IMG_FORMATS])
            self.img_files_ir = sorted([x.replace('/', os.sep) for x in f_ir if x.split('.')[-1].lower() in IMG_FORMATS])

            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert (self.img_files_rgb, self.img_files_ir), (f'{prefix}No images found', f'{prefix}No images found')
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path_rgb,path_ir}: {e}\nSee {HELP_URL}')

   
   
        # Check cache

        # Check cache
        # Check rgb cache
        self.label_files_rgb = img2label_paths(self.img_files_rgb)  # labels
        # print(self.label_files)
        cache_rgb_path = (p_rgb if p_rgb.is_file() else Path(self.label_files_rgb[0]).parent).with_suffix('.cache')  # cached labels
        if cache_rgb_path.is_file():
            cache_rgb, exists_rgb = torch.load(cache_rgb_path), True  # load
            if cache_rgb['hash'] != get_hash(self.label_files_rgb + self.img_files_rgb) or 'version' not in cache_rgb:  # changed
                cache_rgb, exists_rgb = self.cache_labels(self.img_files_rgb,self.label_files_rgb,
                                                          cache_rgb_path, prefix), False  # re-cache
        else:
            cache_rgb, exists_rgb = self.cache_labels(self.img_files_rgb,self.label_files_rgb,
                                                      cache_rgb_path, prefix), False  # cache

        # Check ir cache
        self.label_files_ir = img2label_paths(self.img_files_ir)  # labels
        # print(self.label_files)
        cache_ir_path = (p_ir if p_ir.is_file() else Path(self.label_files_ir[0]).parent).with_suffix('.cache')  # cached labels
        if cache_ir_path.is_file():
            cache_ir, exists_ir = torch.load(cache_ir_path), True  # load
            if cache_ir['hash'] != get_hash(self.label_files_ir + self.img_files_ir) or 'version' not in cache_ir:  # changed
                cache_ir, exists_ir = self.cache_labels(self.img_files_ir, self.label_files_ir,
                                                        cache_ir_path, prefix), False  # re-cache
        else:
            cache_ir, exists_ir = self.cache_labels(self.img_files_ir, self.label_files_ir,
                                                    cache_ir_path, prefix), False  # cache




        # Display cache
        nf_rgb, nm_rgb, ne_rgb, nc_rgb, n_rgb = cache_rgb.pop('results')  # found, missing, empty, corrupted, total
        nf_ir, nm_ir, ne_ir, nc_ir, n_ir = cache_ir.pop('results')  # found, missing, empty, corrupted, total
        if exists_rgb:
            d = f"Scanning RGB '{cache_rgb_path}' images and labels... {nf_rgb} found, {nm_rgb} missing, {ne_rgb} empty, {nc_rgb} corrupted"
            tqdm(None, desc=prefix + d, total=n_rgb, initial=n_rgb)  # display cache results
        if exists_ir:
            d = f"Scanning IR '{cache_rgb_path}' images and labels... {nf_ir} found, {nm_ir} missing, {ne_ir} empty, {nc_ir} corrupted"
            tqdm(None, desc=prefix + d, total=n_ir, initial=n_ir)  # display cache results

        assert nf_rgb > 0 or not augment, f'{prefix}No labels in {cache_rgb_path}. Can not train without labels. See {HELP_URL}'


        # Read cache
        # Read RGB cache
        [cache_rgb.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels_rgb, shapes_rgb, self.segments_rgb = zip(*cache_rgb.values())
        self.labels_rgb = list(labels_rgb)

        
        # np.savetxt('shapes_rgb.txt',shapes_rgb,fmt='%s')

        self.shapes_rgb = np.array(shapes_rgb, dtype=np.float64)
        self.img_files_rgb = list(cache_rgb.keys())  # update
        self.label_files_rgb = img2label_paths(cache_rgb.keys())  # update
        if single_cls:
            for x in self.labels_rgb:
                x[:, 0] = 0

        n_rgb = len(shapes_rgb)  # number of images
        bi_rgb = np.floor(np.arange(n_rgb) / batch_size).astype(int)  # batch index
        nb_rgb = bi_rgb[-1] + 1  # number of batches
        self.batch_rgb = bi_rgb  # batch index of image
        self.n_rgb = n_rgb
        self.indices_rgb = range(n_rgb)

        # Read IR cache
        [cache_ir.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels_ir, shapes_ir, self.segments_ir = zip(*cache_ir.values())
        self.labels_ir = list(labels_ir)
        self.shapes_ir = np.array(shapes_ir, dtype=np.float64)
        self.img_files_ir = list(cache_ir.keys())  # update
        self.label_files_ir = img2label_paths(cache_ir.keys())  # update
        if single_cls:
            for x in self.labels_ir:
                x[:, 0] = 0

        n_ir = len(shapes_ir)  # number of images
        bi_ir = np.floor(np.arange(n_ir) / batch_size).astype(int)  # batch index
        nb_ir = bi_ir[-1] + 1  # number of batches
        self.batch_ir = bi_ir  # batch index of image
        self.n_ir = n_ir
        self.indices_ir = range(n_ir)

        # # Update labels
        # include_class = []  # filter labels to include only these classes (optional)
        # include_class_array = np.array(include_class).reshape(1, -1)
        # for i, (label, segment) in enumerate(zip(self.labels_ir, self.segments)):
        #     if include_class:
        #         j = (label[:, 0:1] == include_class_array).any(1)
        #         self.labels[i] = label[j]
        #         if segment:
        #             self.segments[i] = segment[j]
        #     if single_cls:  # single-class training, merge all classes into 0
        #         self.labels[i][:, 0] = 0
        #         if segment:
        #             self.segments[i][:, 0] = 0

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s_rgb = self.shapes_rgb  # wh
            ar_rgb = s_rgb[:, 1] / s_rgb[:, 0]  # aspect ratio
            irect_rgb = ar_rgb.argsort()
            self.img_files_rgb = [self.img_files_rgb[i] for i in irect_rgb]
            self.label_files_rgb = [self.label_files_rgb[i] for i in irect_rgb]
            self.labels_rgb = [self.labels_rgb[i] for i in irect_rgb]
            self.shapes_rgb = s_rgb[irect_rgb]  # wh
            ar_rgb = ar_rgb[irect_rgb]

            # Set training image shapes
            shapes_rgb = [[1, 1]] * nb_rgb
            for i in range(nb_rgb):
                ari_rgb = ar_rgb[bi_rgb == i]
                mini, maxi = ari_rgb.min(), ari_rgb.max()
                if maxi < 1:
                    shapes_rgb[i] = [maxi, 1]
                elif mini > 1:
                    shapes_rgb[i] = [1, 1 / mini]

            self.batch_shapes_rgb = np.ceil(np.array(shapes_rgb) * img_size / stride + pad).astype(int) * stride


            # IR
            # Sort by aspect ratio
            s_ir = self.shapes_ir  # wh
            ar_ir = s_ir[:, 1] / s_ir[:, 0]  # aspect ratio
            irect_ir = ar_ir.argsort()
            self.img_files_ir = [self.img_files_ir[i] for i in irect_ir]
            self.label_files_ir = [self.label_files_ir[i] for i in irect_ir]
            self.labels_ir = [self.labels_ir[i] for i in irect_ir]
            self.shapes_ir = s_ir[irect_ir]  # wh
            ar_ir = ar_ir[irect_ir]

            # Set training image shapes
            shapes_ir = [[1, 1]] * nb_ir
            for i in range(nb_ir):
                ari_ir = ar_ir[bi_ir == i]
                mini, maxi = ari_ir.min(), ari_ir.max()
                if maxi < 1:
                    shapes_ir[i] = [maxi, 1]
                elif mini > 1:
                    shapes_ir[i] = [1, 1 / mini]

            self.batch_shapes_ir = np.ceil(np.array(shapes_ir) * img_size / stride + pad).astype(int) * stride

        self.imgs_rgb = [None] * n_rgb
        self.imgs_ir = [None] * n_ir

        self.labels = self.labels_rgb
        self.shapes = self.shapes_rgb
        self.indices = self.indices_rgb

        self.imgs_rgb,  self.imgs_ir ,self.img_npy = [None] * n_rgb, [None] * n_ir , [None] * n_rgb

        # # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        # self.imgs, self.img_npy = [None] * n, [None] * n
        # if cache_images:
        #     if cache_images == 'disk':
        #         self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
        #         self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
        #         self.im_cache_dir.mkdir(parents=True, exist_ok=True)
        #     gb = 0  # Gigabytes of cached images
        #     self.img_hw0, self.img_hw = [None] * n, [None] * n
        #     results = ThreadPool(NUM_THREADS).imap(lambda x: load_image_label(*x), zip(repeat(self), range(n)))
        #     pbar = tqdm(enumerate(results), total=n)
        #     for i, x in pbar:
        #         if cache_images == 'disk':
        #             if not self.img_npy[i].exists():
        #                 np.save(self.img_npy[i].as_posix(), x[0])
        #             gb += self.img_npy[i].stat().st_size
        #         else:
        #             self.imgs[i], self.img_hw0[i], self.img_hw[i], self.labels[i] = x  # im, hw_orig, hw_resized, label_resized = load_image_label(self, i)
        #             gb += self.imgs[i].nbytes
        #         pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'
        #     pbar.close()

    def cache_labels(self, imgfiles,labelfiles,path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        img_files = imgfiles
        label_files = labelfiles
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(img_files, label_files, repeat(prefix), repeat(self.cls_names))),
                        desc=desc, total=len(img_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments] 
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(label_files + img_files)
        x['results'] = nf, nm, ne, nc, len(img_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x

    def __len__(self):
        return len(self.img_files_rgb)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        '''
        Augment the [clsid poly] labels and trans label format to rbox.
        Returns:
            img (tensor): (3, height, width), RGB
            labels_out (tensor): (n, [None clsid cx cy l s theta gaussian_θ_labels]) θ∈[-pi/2, pi/2)
            img_file (str): img_dir 
            shapes : None or [(h_raw, w_raw), (hw_ratios, wh_paddings)], for COCO mAP rescaling
        '''
        index_rgb = self.indices_rgb[index]  # linear, shuffled, or image_weights
        index_ir = self.indices_ir[index]  # linear, shuffled, or image_weights


        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic



            img_rgb, labels_rgb, img_ir, labels_ir = load_mosaic_RGB_IR(self, index_rgb, index_ir)
          

            if index == 3509:
                X = Image.fromarray(img_rgb)
                X.save("HH.jpeg")


            # img, labels = load_mosaic(self, index)
            shapes = None

            # MixUp augmentation
            # if random.random() < hyp['mixup']:
            #     img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1)))

        else:
            # Load image and label
            img_rgb,img_ir, (h0, w0), (h, w), img_label = load_image_rgb_ir(self, index) 





            # Letterbox
            shape = self.batch_shapes_rgb[self.batch_rgb[index]] if self.rect else self.img_size  # final letterboxed shape
            img_rgb, ratio, pad = letterbox(img_rgb, shape, auto=False, scaleup=self.augment)
            img_ir, ratio, pad = letterbox(img_ir, shape, auto=False, scaleup=self.augment)

            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling


            labels = self.labels_rgb[index].copy() # labels (array): (num_gt_perimg, [cls_id, poly])
            if labels.size:  
                # labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                labels[:, [1, 3, 5, 7]] = img_label[:, [1, 3, 5, 7]] * ratio[0] + pad[0]
                labels[:, [2, 4, 6, 8]] = img_label[:, [2, 4, 6, 8]] * ratio[1] + pad[1]

            labels_rgb = labels
            labels_ir = labels

            # img4_rgb, img4_ir, labels4_rgb, labels4_ir = random_perspective_rgb_ir(img4_rgb, img4_ir, labels4_rgb, labels4_ir,
            #                                                                     segments4_rgb, segments4_ir,
            #                                                 degrees=self.hyp['degrees'],
            #                                                 translate=self.hyp['translate'],
            #                                                 scale=self.hyp['scale'],
            #                                                 shear=self.hyp['shear'],
            #                                                 perspective=self.hyp['perspective'],
            #                                                 border=self.mosaic_border)  # border to remove

        nL = len(labels_rgb)  # number of labels
        # if nl:
        #     labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)


        if self.augment:
            # Albumentations
            # img, labels = self.albumentations(img, labels)
            # nl = len(labels)  # update after albumentations

            # HSV color-space

            if index == 3509:
                X = Image.fromarray(img_ir)
                X.save("HH.jpeg")
                
            augment_hsv(img_rgb, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
            augment_hsv(img_ir, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            if index == 3509:
                X = Image.fromarray(img_ir)
                X.save("H11H.jpeg")

            img_h, img_w = img_rgb.shape[0], img_rgb.shape[1]
            # Flip up-down
            if random.random() < hyp['flipud']:
                img_rgb = np.flipud(img_rgb)
                img_ir = np.flipud(img_ir)
                if nL:
                    # labels[:, 2] = 1 - labels[:, 2]
                    labels_rgb[:, 2::2] = img_h - labels_rgb[:, 2::2] - 1

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img_rgb = np.fliplr(img_rgb)
                img_ir = np.fliplr(img_ir)

                if nL:
                    # labels[:, 1] = 1 - labels[:, 1]
                    labels_rgb[:, 1::2] = img_w - labels_rgb[:, 1::2] - 1

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout
        if nL:
        # *[clsid poly] to *[clsid cx cy l s theta gaussian_θ_labels] θ∈[-pi/2, pi/2) non-normalized
            rboxes, csl_labels  = poly2rbox(polys=labels_rgb[:, 1:], 
                                            num_cls_thata=hyp['cls_theta'] if hyp else 180, 
                                            radius=hyp['csl_radius'] if hyp else 6.0, 
                                            use_pi=True, use_gaussian=True)
            labels_obb = np.concatenate((labels_rgb[:, :1], rboxes, csl_labels), axis=1)
            labels_mask = (rboxes[:, 0] >= 0) & (rboxes[:, 0] < img_rgb.shape[1]) \
                        & (rboxes[:, 1] >= 0) & (rboxes[:, 0] < img_rgb.shape[0]) \
                        & (rboxes[:, 2] > 5) | (rboxes[:, 3] > 5)
            labels_obb = labels_obb[labels_mask]
            nL = len(labels_obb)  # update after filter
        
        if hyp:
            c_num = 7 + hyp['cls_theta'] # [index_of_batch clsid cx cy l s theta gaussian_θ_labels]
        else:
            c_num = 187

        # labels_out = torch.zeros((nl, 6))
        labels_out = torch.zeros((nL, c_num))
        if nL:
            # labels_out[:, 1:] = torch.from_numpy(labels)
            labels_out[:, 1:] = torch.from_numpy(labels_obb)

        # Convert
        img_rgb = img_rgb[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_rgb = np.ascontiguousarray(img_rgb)
        img_ir = img_ir[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_ir = np.ascontiguousarray(img_ir)
        img_all = np.concatenate((img_rgb, img_ir), axis=0)

        return torch.from_numpy(img_all), labels_out, self.img_files_rgb[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed; (tupe(b*tensor))
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image_rgb_ir(self, i):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    img_rgb = self.imgs_rgb[i]
    img_ir = self.imgs_ir[i]

    label = self.labels[i].copy() # labels (array): (num_gt_perimg, [cls_id, poly])
    if (img_rgb is None) and (img_ir is None):  # not cached

        # npy = self.img_npy[i]

        # if npy and npy.exists():  # load npy

        #     img_rgb = np.load(npy)
        #     img_ir  = np.load(npy)

        # else:  # read image


        path_rgb = self.img_files_rgb[i]
        path_ir = self.img_files_ir[i]

        img_rgb = cv2.imread(path_rgb)  # BGR
        img_ir = cv2.imread(path_ir)  # BGR


        assert img_rgb is not None, 'Image RGB Not Found ' + path_rgb
        assert img_ir is not None, 'Image IR Not Found ' + path_ir


        h0, w0 = img_rgb.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img_rgb = cv2.resize(img_rgb, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
            img_ir = cv2.resize(img_ir, (int(w0 * r), int(h0 * r)),
                             interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
            label[:, 1:] *= r
        return img_rgb, img_ir,(h0, w0), img_rgb.shape[:2], label  # im, hw_original, hw_resized, resized_label
    else:
        return self.imgs_rgb[i], self.imgs_ir[i], self.img_hw0_rgb[i], self.img_hw_rgb[i]  , self.labels[i]



def load_image_label(self, i):
    # loads 1 image from dataset index 'i', returns im, original hw, resized hw
    im = self.imgs[i]
    label = self.labels[i].copy() # labels (array): (num_gt_perimg, [cls_id, poly])
    if im is None:  # not cached in ram
        npy = self.img_npy[i]
        if npy and npy.exists():  # load npy
            im = np.load(npy)
        else:  # read image
            path = self.img_files[i]
            im = cv2.imread(path)  # BGR
            assert im is not None, f'Image Not Found {path}'
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
            label[:, 1:] *= r
        return im, (h0, w0), im.shape[:2], label  # im, hw_original, hw_resized, resized_label
    else:
        return self.imgs[i], self.img_hw0[i], self.img_hw[i], self.labels[i]  # im, hw_original, hw_resized, resized_label




def load_mosaic(self, index):
    # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w), img_label = load_image_label(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = img_label.copy(), self.segments[index].copy() # labels (array): (num_gt_perimg, [cls_id, poly])
        if labels.size:
            # labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            labels[:, [1, 3, 5, 7]] = img_label[:, [1, 3, 5, 7]] + padw
            labels[:, [2, 4, 6, 8]] = img_label[:, [2, 4, 6, 8]] + padh
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    # for x in (labels4[:, 1:], *segments4):
    for x in (segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    h_filter = 2 * s
    w_filter = 2 * s
    labels_mask = poly_filter(polys=labels4[:, 1:].copy(), h=h_filter, w=w_filter)
    labels4 = labels4[labels_mask]
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4

# def load_mosaic_RGB_IR(self, index1, index2):

def load_mosaic_RGB_IR(self, index1, index2):


    index_rgb = index1
    index_ir = index2

    labels4_rgb, segments4_rgb = [], []
    labels4_ir, segments4_ir = [], []

    s = self.img_size
    yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y

    assert index_rgb == index_ir, 'INDEX RGB 不等于 INDEX IR'

    indices = [index_rgb] + random.choices(self.indices, k=3)  # 3 additional image indices
    
    # random.shuffle(indices)

    for i, index in enumerate(indices):
        # Load image
        img_rgb, img_ir, _, (h, w),img_label = load_image_rgb_ir(self, index)


        # place img in img4
        if i == 0:  # top left
            img4_rgb = np.full((s * 2, s * 2, img_rgb.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            img4_ir = np.full((s * 2, s * 2, img_ir.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)


        img4_rgb[y1a:y2a, x1a:x2a] = img_rgb[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        img4_ir[y1a:y2a, x1a:x2a] = img_ir[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b


        # Labels

        labels_rgb, segments_rgb = img_label.copy(), self.segments_rgb[index].copy() # labels (array): (num_gt_perimg, [cls_id, poly])
        labels_ir, segments_ir = img_label.copy(), self.segments_ir[index].copy() # labels (array): (num_gt_perimg, [cls_id, poly])
        
        
        
        if labels_rgb.size:
            # labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            labels_rgb[:, [1, 3, 5, 7]] = labels_rgb[:, [1, 3, 5, 7]] + padw
            labels_rgb[:, [2, 4, 6, 8]] = labels_rgb[:, [2, 4, 6, 8]] + padh

            labels_ir[:, [1, 3, 5, 7]] = labels_ir[:, [1, 3, 5, 7]] + padw
            labels_ir[:, [2, 4, 6, 8]] = labels_ir[:, [2, 4, 6, 8]] + padh

            segments_rgb = [xyn2xy(x, w, h, padw, padh) for x in segments_rgb]
            segments_ir = [xyn2xy(x, w, h, padw, padh) for x in segments_ir]

        labels4_rgb.append(labels_rgb)
        segments4_rgb.extend(segments_rgb)
        labels4_ir.append(labels_ir)
        segments4_ir.extend(segments_ir)

    # Concat/clip labels
    labels4_rgb = np.concatenate(labels4_rgb, 0)
    labels4_ir = np.concatenate(labels4_ir, 0)

    for x in (segments4_rgb):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    for x in (segments4_ir):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()

    h_filter = 2 * s
    w_filter = 2 * s
    labels_mask = poly_filter(polys=labels4_rgb[:, 1:].copy(), h=h_filter, w=w_filter)
    labels4_rgb = labels4_rgb[labels_mask]
    labels_mask = poly_filter(polys=labels4_ir[:, 1:].copy(), h=h_filter, w=w_filter)
    labels4_ir = labels4_ir[labels_mask]

    # Augment
    img4_rgb, labels4_rgb, segments4_rgb = copy_paste(img4_rgb, labels4_rgb, segments4_rgb, p=self.hyp['copy_paste'])
    img4_ir, labels4_ir, segments4_ir = copy_paste(img4_ir, labels4_ir, segments4_ir, p=self.hyp['copy_paste'])



    img4_rgb, img4_ir, labels4_rgb, labels4_ir = random_perspective_rgb_ir(img4_rgb, img4_ir, labels4_rgb, labels4_ir,
                                                                           segments4_rgb, segments4_ir,
                                                    degrees=self.hyp['degrees'],
                                                    translate=self.hyp['translate'],
                                                    scale=self.hyp['scale'],
                                                    shear=self.hyp['shear'],
                                                    perspective=self.hyp['perspective'],
                                                    border=self.mosaic_border)  # border to remove


    # img4_rgb, labels4_rgb = random_perspective(img4_rgb, labels4_rgb, segments4_rgb,
    #                                    degrees=self.hyp['degrees'],
    #                                    translate=self.hyp['translate'],
    #                                    scale=self.hyp['scale'],
    #                                    shear=self.hyp['shear'],
    #                                    perspective=self.hyp['perspective'],
    #                                    border=self.mosaic_border)  # border to remove

    # img4_ir, labels4_ir = random_perspective(img4_ir, labels4_ir, segments4_ir,
    #                                    degrees=self.hyp['degrees'],
    #                                    translate=self.hyp['translate'],
    #                                    scale=self.hyp['scale'],
    #                                    shear=self.hyp['shear'],
    #                                    perspective=self.hyp['perspective'],
    #                                    border=self.mosaic_border)  # border to remove

    return img4_rgb, labels4_rgb, img4_ir, labels4_ir



def load_mosaic(self, index):
    # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w), img_label = load_image_label(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = img_label.copy(), self.segments[index].copy() # labels (array): (num_gt_perimg, [cls_id, poly])
        if labels.size:
            # labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            labels[:, [1, 3, 5, 7]] = img_label[:, [1, 3, 5, 7]] + padw
            labels[:, [2, 4, 6, 8]] = img_label[:, [2, 4, 6, 8]] + padh
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    # for x in (labels4[:, 1:], *segments4):
    for x in (segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    h_filter = 2 * s
    w_filter = 2 * s
    labels_mask = poly_filter(polys=labels4[:, 1:].copy(), h=h_filter, w=w_filter)
    labels4 = labels4[labels_mask]
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4


def load_mosaic9(self, index):
    # YOLOv5 9-mosaic loader. Loads 1 image + 8 random images into a 9-image mosaic
    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w), img_label = load_image_label(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

        # Labels
        labels, segments = img_label.copy(), self.segments[index].copy() # labels (array): (num_gt_perimg, [cls_id, poly])
        if labels.size:
            # labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            labels_ = labels.clone() if isinstance(labels, torch.Tensor) else np.copy(labels)
            labels_[:, [1, 3, 5, 7]] = labels[:, [1, 3, 5, 7]] + padx
            labels_[:, [2, 4, 6, 8]] = labels[:, [2, 4, 6, 8]] + pady
            labels = labels_

        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous

    # Offset
    yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    # labels9[:, [1, 3]] -= xc
    # labels9[:, [2, 4]] -= yc
    labels9[:, [1, 3, 5, 7]] -= xc
    labels9[:, [2, 4, 6, 8]] -= yc

    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    # for x in (labels9[:, 1:], *segments9):
    for x in (segments9):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    h_filter = 2 * s
    w_filter = 2 * s
    labels_mask = poly_filter(polys=labels9[:, 1:].copy(), h=h_filter, w=w_filter)
    labels9 = labels9[labels_mask]
    # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
    img9, labels9 = random_perspective(img9, labels9, segments9,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img9, labels9


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path='../datasets/coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='../datasets/coco128'):  # from utils.datasets import *; extract_boxes()
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file) as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path='../datasets/coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path.parent / x).unlink(missing_ok=True) for x in txt]  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write('./' + img.relative_to(path.parent).as_posix() + '\n')  # add image to txt file


def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix, cls_name_list = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                labels = [x.split() for x in f.read().strip().splitlines() if len(x)]

                # Yolov5-obb does not support segment labels yet
                # if any([len(x) > 8 for x in l]):  # is segment
                #     classes = np.array([x[0] for x in l], dtype=np.float32)
                #     segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  # (cls, xy1...)
                #     l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                l_ = []
                for label in labels:
                    if label[-1] == "2": # diffcult
                        continue
                    cls_id = cls_name_list.index(label[8])
                    l_.append(np.concatenate((str(cls_id), label[:8]), axis=None))
                l = np.array(l_, dtype=np.float32)
            nl = len(l)
            if nl:
                assert len(label) == 10, f'Yolov5-OBB labels require 10 columns, which same as DOTA Dataset, {len(label)} columns detected'
                assert (l >= 0).all(), f'negative label values {l[l < 0]}, please check your dota format labels'
                #assert (l[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {l[:, 1:][l[:, 1:] > 1]}'
                _, i = np.unique(l, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    l = l[i]  # remove duplicates
                    if segments:
                        segments = segments[i]
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                # l = np.zeros((0, 5), dtype=np.float32)
                l = np.zeros((0, 9), dtype=np.float32)
        else:
            nm = 1  # label missing
            # l = np.zeros((0, 5), dtype=np.float32)
            l = np.zeros((0, 9), dtype=np.float32)
        return im_file, l, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


def dataset_stats(path='coco128.yaml', autodownload=False, verbose=False, profile=False, hub=False):
    """ Return dataset statistics dictionary with images and instances counts per split per class
    To run in parent directory: export PYTHONPATH="$PWD/yolov5"
    Usage1: from utils.datasets import *; dataset_stats('coco128.yaml', autodownload=True)
    Usage2: from utils.datasets import *; dataset_stats('../datasets/coco128_with_yaml.zip')
    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally
        verbose:        Print stats dictionary
    """

    def round_labels(labels):
        # Update labels to integer class and 6 decimal place floats
        return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]

    def unzip(path):
        # Unzip data.zip TODO: CONSTRAINT: path/to/abc.zip MUST unzip to 'path/to/abc/'
        if str(path).endswith('.zip'):  # path is data.zip
            assert Path(path).is_file(), f'Error unzipping {path}, file not found'
            ZipFile(path).extractall(path=path.parent)  # unzip
            dir = path.with_suffix('')  # dataset directory == zip name
            return True, str(dir), next(dir.rglob('*.yaml'))  # zipped, data_dir, yaml_path
        else:  # path is data.yaml
            return False, None, path

    def hub_ops(f, max_dim=1920):
        # HUB ops for 1 image 'f': resize and save at reduced quality in /dataset-hub for web/app viewing
        f_new = im_dir / Path(f).name  # dataset-hub image filename
        try:  # use PIL
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)  # ratio
            if r < 1.0:  # image too large
                im = im.resize((int(im.width * r), int(im.height * r)))
            im.save(f_new, 'JPEG', quality=75, optimize=True)  # save
        except Exception as e:  # use OpenCV
            print(f'WARNING: HUB ops PIL failure {f}: {e}')
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im_width)  # ratio
            if r < 1.0:  # image too large
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(f_new), im)

    zipped, data_dir, yaml_path = unzip(Path(path))
    with open(check_yaml(yaml_path), errors='ignore') as f:
        data = yaml.safe_load(f)  # data dict
        if zipped:
            data['path'] = data_dir  # TODO: should this be dir.resolve()?
    check_dataset(data, autodownload)  # download dataset if missing
    hub_dir = Path(data['path'] + ('-hub' if hub else ''))
    stats = {'nc': data['nc'], 'names': data['names']}  # statistics dictionary
    for split in 'train', 'val', 'test':
        if data.get(split) is None:
            stats[split] = None  # i.e. no test set
            continue
        x = []
        dataset = LoadImagesAndLabels(data[split])  # load dataset
        for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics'):
            x.append(np.bincount(label[:, 0].astype(int), minlength=data['nc']))
        x = np.array(x)  # shape(128x80)
        stats[split] = {'instance_stats': {'total': int(x.sum()), 'per_class': x.sum(0).tolist()},
                        'image_stats': {'total': dataset.n, 'unlabelled': int(np.all(x == 0, 1).sum()),
                                        'per_class': (x > 0).sum(0).tolist()},
                        'labels': [{str(Path(k).name): round_labels(v.tolist())} for k, v in
                                   zip(dataset.img_files, dataset.labels)]}

        if hub:
            im_dir = hub_dir / 'images'
            im_dir.mkdir(parents=True, exist_ok=True)
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(hub_ops, dataset.img_files), total=dataset.n, desc='HUB Ops'):
                pass

    # Profile
    stats_path = hub_dir / 'stats.json'
    if profile:
        for _ in range(1):
            file = stats_path.with_suffix('.npy')
            t1 = time.time()
            np.save(file, stats)
            t2 = time.time()
            x = np.load(file, allow_pickle=True)
            print(f'stats.npy times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

            file = stats_path.with_suffix('.json')
            t1 = time.time()
            with open(file, 'w') as f:
                json.dump(stats, f)  # save stats *.json
            t2 = time.time()
            with open(file) as f:
                x = json.load(f)  # load hyps dict
            print(f'stats.json times: {time.time() - t2:.3f}s read, {t2 - t1:.3f}s write')

    # Save, print and return
    if hub:
        print(f'Saving {stats_path.resolve()}...')
        with open(stats_path, 'w') as f:
            json.dump(stats, f)  # save stats.json
    if verbose:
        print(json.dumps(stats, indent=2, sort_keys=False))
    return stats






class LoadImagesAndLabels(Dataset):
    # YOLOv5 train_loader/val_loader, loads images and labels for training and validation
    cache_version = 0.6  # dataset labels *.cache version

    def __init__(self, path, cls_names, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        """
        Returns:
            Dataset.labels (list): n_imgs * array(num_gt_perimg, [cls_id, poly])
            Dataset.shapes (array): (n_imgs, [ori_img_width, ori_img_height])

            Dataset.batch_shapes (array): (n_batches, [h_rect, w_rect])
        """
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        self.albumentations = Albumentations() if augment else None

        self.cls_names = cls_names
        # if isinstance(cls_names,dict):
        #     self.cls_names = list(cls_names.values())

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {HELP_URL}')

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # same version
            assert cache['hash'] == get_hash(self.label_files + self.img_files)  # same hash
        except:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {HELP_URL}'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels) # labels(list[array]): n_imgs * array(num_gt_perimg, [cls_id, poly])
        self.shapes = np.array(shapes, dtype=np.float64) # img_ori shape
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0
                if segment:
                    self.segments[i][:, 0] = 0

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1: # batch图像高宽比均小于1时, shape=[h/w, 1] = [h_ratio, w_ratio]
                    shapes[i] = [maxi, 1] 
                elif mini > 1: # batch图像高宽比均大于1时, shape=[1, w/h] = [h_ratio, w_ratio]
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride # (nb, [h_rect, w_rect])

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs, self.img_npy = [None] * n, [None] * n
        if cache_images:
            if cache_images == 'disk':
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(NUM_THREADS).imap(lambda x: load_image_label(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i], self.labels[i] = x  # im, hw_orig, hw_resized, label_resized = load_image_label(self, i)
                    gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB {cache_images})'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap(verify_image_label, zip(self.img_files, self.label_files, repeat(prefix), repeat(self.cls_names))),
                        desc=desc, total=len(self.img_files))
            for im_file, l, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [l, shape, segments] 
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{prefix}WARNING: No labels found in {path}. See {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, len(self.img_files)
        x['msgs'] = msgs  # warnings
        x['version'] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
            LOGGER.info(f'{prefix}New cache created: {path}')
        except Exception as e:
            LOGGER.warning(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')  # not writeable
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        '''
        Augment the [clsid poly] labels and trans label format to rbox.
        Returns:
            img (tensor): (3, height, width), RGB
            labels_out (tensor): (n, [None clsid cx cy l s theta gaussian_θ_labels]) θ∈[-pi/2, pi/2)
            img_file (str): img_dir 
            shapes : None or [(h_raw, w_raw), (hw_ratios, wh_paddings)], for COCO mAP rescaling
        '''
        index = self.indices[index]  # linear, shuffled, or image_weights

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            img, labels = load_mosaic(self, index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp['mixup']:
                img, labels = mixup(img, labels, *load_mosaic(self, random.randint(0, self.n - 1)))

        else:
            # Load image and label
            img, (h0, w0), (h, w), img_label = load_image_label(self, index) 

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape [h_rect, w_rect]
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment) # ratio[w_ratio, h_ratio], pad[w_padding, h_padding]
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling [(h_raw, w_raw), (hw_ratios, wh_paddings)]

            labels = img_label.copy() # labels (array): (num_gt_perimg, [cls_id, poly])
            if labels.size:  
                # labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
                labels[:, [1, 3, 5, 7]] = img_label[:, [1, 3, 5, 7]] * ratio[0] + pad[0]
                labels[:, [2, 4, 6, 8]] = img_label[:, [2, 4, 6, 8]] * ratio[1] + pad[1]

            if self.augment:
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

        nl = len(labels)  # number of labels
        # if nl:
        #     labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)


        if self.augment:
            # Albumentations
            # img, labels = self.albumentations(img, labels)
            # nl = len(labels)  # update after albumentations

            # HSV color-space
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            img_h, img_w = img.shape[0], img.shape[1]
            # Flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nl:
                    # labels[:, 2] = 1 - labels[:, 2]
                    labels[:, 2::2] = img_h - labels[:, 2::2] - 1

            # Flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nl:
                    # labels[:, 1] = 1 - labels[:, 1]
                    labels[:, 1::2] = img_w - labels[:, 1::2] - 1

            # Cutouts
            # labels = cutout(img, labels, p=0.5)
            # nl = len(labels)  # update after cutout
        if nl:
        # *[clsid poly] to *[clsid cx cy l s theta gaussian_θ_labels] θ∈[-pi/2, pi/2) non-normalized
            rboxes, csl_labels  = poly2rbox(polys=labels[:, 1:], 
                                            num_cls_thata=hyp['cls_theta'] if hyp else 180, 
                                            radius=hyp['csl_radius'] if hyp else 6.0, 
                                            use_pi=True, use_gaussian=True)
            labels_obb = np.concatenate((labels[:, :1], rboxes, csl_labels), axis=1)
            labels_mask = (rboxes[:, 0] >= 0) & (rboxes[:, 0] < img.shape[1]) \
                        & (rboxes[:, 1] >= 0) & (rboxes[:, 0] < img.shape[0]) \
                        & (rboxes[:, 2] > 5) | (rboxes[:, 3] > 5)
            labels_obb = labels_obb[labels_mask]
            nl = len(labels_obb)  # update after filter
        
        if hyp:
            c_num = 7 + hyp['cls_theta'] # [index_of_batch clsid cx cy l s theta gaussian_θ_labels]
        else:
            c_num = 187

        # labels_out = torch.zeros((nl, 6))
        labels_out = torch.zeros((nl, c_num))
        if nl:
            # labels_out[:, 1:] = torch.from_numpy(labels)
            labels_out[:, 1:] = torch.from_numpy(labels_obb)

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed; (tupe(b*tensor))
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4

def random_perspective_rgb_ir(img_rgb, img_ir, targets_rgb=(),targets_ir=(), segments_rgb=(), segments_ir=(),
                              degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxyxyxy]

    im = img_rgb
    targets = targets_rgb
    segments = segments_rgb

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img_rgb = cv2.warpPerspective(img_rgb, M, dsize=(width, height), borderValue=(114, 114, 114))
            img_ir = cv2.warpPerspective(img_ir, M, dsize=(width, height), borderValue=(114, 114, 114))
   
        else:  # affine
            img_rgb = cv2.warpAffine(img_rgb, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            img_ir = cv2.warpAffine(img_ir, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            # xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy[:, :2] = targets[:, 1:].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # # create new boxes
            # x = xy[:, [0, 2, 4, 6]]
            # y = xy[:, [1, 3, 5, 7]]
            # new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # # clip
            # new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            # new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
            # clip boxes 不启用，保留预测完整物体的能力

        # filter candidates
        # i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        # targets = targets[i]
        # targets[:, 1:5] = new[i]
        targets_mask = poly_filter(polys=xy, h=height, w=width)
        targets[:, 1:] = xy
        targets = targets[targets_mask]

    return img_rgb, img_ir, targets, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


