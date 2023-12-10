
import os
import re
import utm
import cv2
import math
import shutil
from tqdm import tqdm
from mega import Mega
from skimage import io
from os.path import join


THRESHOLD_IN_METERS = 5
RETRY_SECONDS = 2


def get_distance(coords_A, coords_B):
    return math.sqrt((float(coords_B[0])-float(coords_A[0]))**2 + (float(coords_B[1])-float(coords_A[1]))**2)


def is_valid_timestamp(timestamp):
    """Return True if it's a valid timestamp, in format YYYYMMDD_hhmmss,
        with all fields from left to right optional.
    >>> is_valid_timestamp('')
    True
    >>> is_valid_timestamp('201901')
    True
    >>> is_valid_timestamp('20190101_123000')
    True
    """
    return bool(re.match("^(\d{4}(\d{2}(\d{2}(_(\d{2})(\d{2})?(\d{2})?)?)?)?)?$", timestamp))


def format_coord(num, left=2, right=5):
    """Return the formatted number as a string with (left) int digits 
            (including sign '-' for negatives) and (right) float digits.
    >>> format_coord(1.1, 3, 3)
    '001.100'
    >>> format_coord(-0.123, 3, 3)
    '-00.123'
    """
    sign = "-" if float(num) < 0 else ""
    num = str(abs(float(num))) + "."
    integer, decimal = num.split(".")[:2]
    left -= len(sign)
    return f"{sign}{int(integer):0{left}d}.{decimal[:right]:<0{right}}"


def format_location_info(latitude, longitude):
    easting, northing, zone_number, zone_letter = utm.from_latlon(float(latitude), float(longitude))
    easting = format_coord(easting, 7, 2)
    northing = format_coord(northing, 7, 2)
    latitude = format_coord(latitude, 3, 5)
    longitude = format_coord(longitude, 4, 5)
    return easting, northing, zone_number, zone_letter, latitude, longitude


def get_dst_image_name(latitude, longitude, pano_id=None, tile_num=None, heading=None,
                       pitch=None, roll=None, height=None, timestamp=None, note=None, extension=".jpg"):
    easting, northing, zone_number, zone_letter, latitude, longitude = format_location_info(latitude, longitude)
    tile_num  = f"{int(float(tile_num)):02d}" if tile_num  is not None else ""
    heading   = f"{int(float(heading)):03d}"  if heading   is not None else ""
    pitch     = f"{int(float(pitch)):03d}"    if pitch     is not None else ""
    timestamp = f"{timestamp}"                if timestamp is not None else ""
    note      = f"{note}"                     if note      is not None else ""
    assert is_valid_timestamp(timestamp), f"{timestamp} is not in YYYYMMDD_hhmmss format"
    if roll is None: roll = ""
    else: raise NotImplementedError()
    if height is None: height = ""
    else: raise NotImplementedError()
    
    return f"@{easting}@{northing}@{zone_number:02d}@{zone_letter}@{latitude}@{longitude}" + \
           f"@{pano_id}@{tile_num}@{heading}@{pitch}@{roll}@{height}@{timestamp}@{note}@{extension}"


class VideoReader:
    def __init__(self, video_name, size=None):
        if not os.path.exists(video_name):
            raise FileNotFoundError(f"{video_name} does not exist")
        self.video_name = video_name
        self.size = size
        self.vc = cv2.VideoCapture(f"{video_name}")
        self.frames_per_second = self.vc.get(cv2.CAP_PROP_FPS)
        self.frame_duration_millis = 1000 / self.frames_per_second
        self.frames_num = int(self.vc.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_length_in_millis = int(self.frames_num * 1000 / self.frames_per_second)

    def get_frame_at_frame_num(self, frame_num):
        self.vc.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        frame = self.vc.read()[1]
        if frame is None: return None  # In case of corrupt videos
        if self.size is not None:
            frame = cv2.resize(frame, self.size[::-1], cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def __del__(self):
        self.vc.release()


dataset_name = "st_lucia"
raw_data_folder = join(dataset_name, "raw_data")
dst_database_folder = join(dataset_name, "train_set")
dst_queries_folder = join(dataset_name, "test_set")
os.makedirs(dataset_name, exist_ok=True)
os.makedirs(dst_database_folder, exist_ok=True)
os.makedirs(dst_queries_folder, exist_ok=True)
os.makedirs(raw_data_folder, exist_ok=True)

# Use the first pass for the database, and the last one for the queries
urls = ['https://mega.nz/file/nE4g0LzZ#c8eL_H3ZfXElqEukw38i32p5cjwusTuNJYYeEP1d5Pg',
        'https://mega.nz/file/PAgWSIhD#UeeA6knWL3pDh_IczbYkcA1R1MwSZ2vhEg2DTr1_oNw']
login = Mega().login()

for sequence_num, url in enumerate(urls):
    print(f"{sequence_num:>2} / {len(urls)} ) downloading {url}")
    zip_path = login.download_url(url, raw_data_folder)
    zip_path = str(zip_path)
    subset_name = os.path.basename(zip_path).replace(".zip", "")
    shutil.unpack_archive(zip_path, raw_data_folder)
    
    vr = VideoReader(join(raw_data_folder, subset_name, "webcam_video.avi"))
    
    with open(join(raw_data_folder, subset_name, "fGPS.txt"), "r") as file:
        lines = file.readlines()
    
    last_coordinates = None
    for frame_num, line in zip(tqdm(range(vr.frames_num)), lines):
        latitude, longitude = line.split(",")
        latitude = "-" + latitude  # Given latitude is positive, real latitude is negative (in Australia)
        easting, northing = format_location_info(latitude, longitude)[:2]
        if last_coordinates is None:
            last_coordinates = (easting, northing)
        else:
            distance_in_meters = get_distance(last_coordinates, (easting, northing))
            if distance_in_meters < THRESHOLD_IN_METERS:
                continue  # If this frame is too close to the previous one, skip it
            else:
                last_coordinates = (easting, northing)
        
        frame = vr.get_frame_at_frame_num(frame_num)
        image_name = get_dst_image_name(latitude, longitude, pano_id=f"{subset_name}_{frame_num:05d}")
        if sequence_num == 0:  # The first sequence is the database
            io.imsave(join(dst_database_folder, image_name), frame)
        else:
            io.imsave(join(dst_queries_folder, image_name), frame)

shutil.rmtree(raw_data_folder)
