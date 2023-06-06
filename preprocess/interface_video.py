import cv2
import os
from preprocess._utils import get_parent_dir, run_cmd, log

# Process
# 1. Video -> Image
# 2. (TBD) Image -> Preprocess(Resize and etd)


def video_to_png(dataset_path, video_name, max_scale=1.0, fps=-1, target_num_frames=100):
    """
    :param dataset_path
    :param video_name: Video file path from project directory
    :param max_scale: Adjust `max_scale` to something smaller for faster processing.
    :param fps:
        A smaller FPS will be much faster for bundle adjustment,
        but at the expensive of a lower sampling density for training.
        For the paper we used ~15 fps but we default to something lower here to get you started faster.
        if given an fps of -1 we will try to auto-compute it.
    :param target_num_frames: 결과적으로 사용할 frame 수. 즉 실제 영상 fps에서 우리가 뽑을 frame 수에 해당
    """
    log("="*100, add_time=False)
    log(f"Video to Image Conversion started")

    output_dir = 'img'
    video_path = f"{dataset_path}/{video_name}"
    os.makedirs(f"{dataset_path}/{output_dir}", exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if num_frames < target_num_frames:
        raise RuntimeError('The video is too short and has fewer frames than the target.')

    if fps == -1:
        fps = int(target_num_frames / num_frames * input_fps)
        log(f"Auto-computed FPS = {fps}")

    filters = f"mpdecimate,setpts=N/FRAME_RATE/TB,scale=iw*{max_scale}:ih*{max_scale}"
    out_pattern = str(f'{get_parent_dir(video_path)}/{output_dir}/%06d.png')
    cmd = f'ffmpeg -i {video_path} -r {fps} -vf {filters}  {out_pattern}'
    out = run_cmd(cmd)
    log(f"FFMPEG cmd : {cmd}")
    log(f"FFMPEG DONE : {out}")


if __name__ == '__main__':
    path = "C:/Users/user/github/nerf/dataset/custom/test/cup.mp4"
    video_to_png(path)

