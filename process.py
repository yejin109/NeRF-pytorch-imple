import cv2
import os

# TODO
# 1. ffmpeg 설치(7z로 시스템 환경 변수 추가 및 python package 설치 포함)

# Process
# 1. Video -> Image
# 2. (TBD) Image -> Preprocess(Resize and etd)

video_path = "./dataset/custom/sunset/raw/sunset.mp4"

# Adjust `max_scale` to something smaller for faster processing.
max_scale = 1.0

# A smaller FPS will be much faster for bundle adjustment,
# but at the expensive of a lower sampling density for training.

# For the paper we used ~15 fps but we default to something lower here to get you started faster.
# If given an fps of -1 we will try to auto-compute it.
fps = -1
target_num_frames = 100


cap = cv2.VideoCapture(video_path)
input_fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


if num_frames < target_num_frames:
  raise RuntimeError(
      'The video is too short and has fewer frames than the target.')

if fps == -1:
  fps = int(target_num_frames / num_frames * input_fps)
  print(f"Auto-computed FPS = {fps}")

overwrite = False
filters = f"mpdecimate,setpts=N/FRAME_RATE/TB,scale=iw*{max_scale}:ih*{max_scale}"
tmp_rgb_raw_dir = 'rgb-raw'
out_pattern = str('./dataset/custom/sunset/rgb-raw/%06d.png')
# os.system(f'cmd /c "ffmpeg -i {video_path} -r {fps} -vf {filters}  {out_pattern}"')
print()

