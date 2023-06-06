from preprocess import video_to_png, img_to_colmap, colmap_to_json
from preprocess._utils import log

CAMERA_TYPE = 'perspective'

TYPE_TO_MODEL = {
    "perspective": "OPENCV",
    "fisheye": "OPENCV_FISHEYE",
    "equirectangular": "EQUIRECTANGULAR",
}

dataset_dir = "./dataset/custom/cup"
video_name = 'cup.mp4'
COLMAP_VERSION = 3.8

log('*'*100)
log(f'Data set : {dataset_dir} / Video : {video_name} / Colmap version : {COLMAP_VERSION}')

video_to_png(dataset_dir, video_name)

img_to_colmap(dataset_dir, TYPE_TO_MODEL[CAMERA_TYPE], COLMAP_VERSION, gpu=True, vocab_tree_filename='C:/Users/ay011/nerfbackend/vocab-tree/vocab_tree_flickr100K_words32K.bin')

reconstruction_dir = './dataset/custom/cup/colmap/sparse/0'
colmap_to_json(reconstruction_dir, dataset_dir, image_id_to_depth_path=None, image_rename_map=None)