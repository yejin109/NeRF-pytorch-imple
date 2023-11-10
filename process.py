import argparse

from preprocess import video_to_png, img_to_colmap, colmap_to_json, resize
from preprocess._utils import log
from preprocess.interface_model import preprocess_neus, preprocess_nerf


parser = argparse.ArgumentParser()

# Model
parser.add_argument('--model', type=str, default='neus')

# Data
parser.add_argument('--camera', type=str, default='perspective')
parser.add_argument('--dir', type=str, default="./dataset/custom/cat")
parser.add_argument('--file', type=str, default="cat.mp4")
parser.add_argument('--v_colmap', type=str, default=3.8)


TYPE_TO_MODEL = {
    "perspective": "OPENCV",
    "fisheye": "OPENCV_FISHEYE",
    "equirectangular": "EQUIRECTANGULAR",
}


def preprocess_default(camera, file_dir, file_name, colmap_version):
    log('*'*100)
    log(f'Data set : {file_dir} / Video : {file_name} / Colmap version : {colmap_version}')

    # Resize
    resize(file_dir, [1, 2, 4, 8])

    img_to_colmap(file_dir, TYPE_TO_MODEL[camera], colmap_version, gpu=True,
                  vocab_tree_filename='C:/Users/ay011/nerfbackend/vocab-tree/vocab_tree_flickr100K_words32K.bin')

    reconstruction_dir = f'{file_dir}/colmap/sparse/0'
    colmap_to_json(reconstruction_dir, file_dir, image_id_to_depth_path=None, image_rename_map=None)


if __name__ == '__main__':
    args = parser.parse_args()

    # Video -> Image
    video_to_png(args.dir, args.file)

    if args.model == 'default':
        preprocess_default(args.camera, args.dir, args.file, args.v_colmap)
    elif args.model == 'nerf':
        preprocess_nerf(args.dir, args.v_colmap)
    elif args.model == 'neus':
        preprocess_neus(args.dir, args.v_colmap)
    else:
        raise AssertionError(f'Cannot support {args.model}')
