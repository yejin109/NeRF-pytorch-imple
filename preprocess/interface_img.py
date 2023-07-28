import os
import cv2
import json
import tqdm
import struct
import imageio
import numpy as np
from glob import glob
from preprocess._utils import run_cmd, log
from preprocess.camera import CameraModels, Camera, parse_colmap_camera_params
from preprocess.image import Image

from PIL import Image as PIL_Image


def img_to_colmap(dataset_dir, camera_model, colmap_version,
                  vocab_tree_filename=None, matching_method='vocab_tree', gpu=False):
    """
    :param gpu
    :param vocab_tree_filename
    :param colmap_version
    :param dataset_dir:
    :param camera_model: C
    :param matching_method:
        Feature matching method to use. Vocab tree is recommended for a balance of speed
        and accuracy. Exhaustive is slower but more accurate. Sequential is faster but
        should only be used for videos.
    :return:
    """

    log("="*100, add_time=False)
    log(f"Image to Colmap Conversion started")
    img_dir = f"{dataset_dir}/images"
    colmap_dir = f"{dataset_dir}/colmap"
    os.makedirs(colmap_dir, exist_ok=True)

    feature_extractor_cmd = [
        f"colmap feature_extractor",
        f"--database_path {colmap_dir}/database.db",
        f"--image_path {img_dir}",
        "--ImageReader.single_camera 1",
        f"--ImageReader.camera_model {camera_model}",
        f"--SiftExtraction.use_gpu {int(gpu)}",
    ]
    feature_extractor_cmd = " ".join(feature_extractor_cmd)
    out = run_cmd(feature_extractor_cmd)
    log(f"Feature extractor cmd : {feature_extractor_cmd}")
    log(f"Feature extractor DONE : {out}")

    feature_matcher_cmd = [
        f"colmap {matching_method}_matcher",
        f"--database_path {colmap_dir}/database.db",
        f"--SiftMatching.use_gpu {int(gpu)}",
    ]
    if matching_method == "vocab_tree":
        feature_matcher_cmd.append(f"--VocabTreeMatching.vocab_tree_path {vocab_tree_filename}")

    feature_matcher_cmd = " ".join(feature_matcher_cmd)
    run_cmd(feature_matcher_cmd)
    log(f"Feature matcher cmd : {feature_matcher_cmd}")
    log(f"Feature matcher DONE : {out}")

    # Bundle adjustment
    sparse_dir = f"{colmap_dir}/sparse"
    os.makedirs(sparse_dir, exist_ok=True)
    mapper_cmd = [
        f"colmap mapper",
        f"--database_path {colmap_dir}/database.db",
        f"--image_path {img_dir}",
        f"--output_path {sparse_dir}",
    ]
    if colmap_version >= 3.7:
        mapper_cmd.append("--Mapper.ba_global_function_tolerance 1e-6")

    mapper_cmd = " ".join(mapper_cmd)
    out = run_cmd(mapper_cmd)
    log(f"Mapper cmd : {mapper_cmd}")
    log(f"Mapper DONE : {out}")

    bundle_adjuster_cmd = [
        f"colmap bundle_adjuster",
        f"--input_path {sparse_dir}/0",
        f"--output_path {sparse_dir}/0",
        "--BundleAdjustment.refine_principal_point 1",
    ]
    bundle_adjuster_cmd = " ".join(bundle_adjuster_cmd)
    out = run_cmd(bundle_adjuster_cmd)
    log(f"Bundle Adjustment DONE : {bundle_adjuster_cmd}")
    log(f"Bundle Adjustment DONE : {out}")


def colmap_to_json(recon_dir, output_dir, camera_mask_path=None, image_id_to_depth_path=None, image_rename_map=None):
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.

    Args:
        recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
        output_dir: Path to the output directory.
        camera_mask_path: Path to the camera mask.
        image_id_to_depth_path: When including sfm-based depth, embed these depth file paths in the exported json
        image_rename_map: Use these image names instead of the names embedded in the COLMAP db

    Returns:
        The number of registered images.
    """

    cam_id_to_camera = read_cameras_binary(f"{recon_dir}/cameras.bin")
    im_id_to_image = read_images_binary(f"{recon_dir}/images.bin")

    frames = []
    for im_id, im_data in im_id_to_image.items():
        # NB: COLMAP uses Eigen / scalar-first quaternions
        # * https://colmap.github.io/format.html
        # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
        # the `rotation_matrix()` handles that format for us.

        rotation = im_data.qvec2rotmat()

        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1

        name = im_data.name
        if image_rename_map is not None:
            name = image_rename_map[name]
        name = f"./{output_dir}/images/{name}"

        frame = {
            "file_path": name,
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": im_id,
        }
        if camera_mask_path is not None:
            frame["mask_path"] = camera_mask_path.relative_to(camera_mask_path.parent.parent).as_posix()
        if image_id_to_depth_path is not None:
            depth_path = image_id_to_depth_path[im_id]
            frame["depth_file_path"] = str(depth_path)
        frames.append(frame)

    if set(cam_id_to_camera.keys()) != {1}:
        raise RuntimeError("Only single camera shared for all images is supported.")
    out = parse_colmap_camera_params(cam_id_to_camera[1])
    out["frames"] = frames

    applied_transform = np.eye(4)[:3, :]
    applied_transform = applied_transform[np.array([1, 0, 2]), :]
    applied_transform[2, :] *= -1
    out["applied_transform"] = applied_transform.tolist()

    with open(f"{output_dir}/transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    return len(frames)


def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CameraModels[model_id].name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CameraModels[model_id].param_num
            params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(
                idx=camera_id, model=model_name, width=width, height=height, params=np.array(params)
            )
        assert len(cameras) == num_cameras
    return cameras


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points_2d = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points_2d, format_char_sequence="ddq" * num_points_2d)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))])
            point_3d_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                idx=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_idx=camera_id,
                name=image_name,
                xys=xys,
                point3D_id=point_3d_ids,
            )
    return images


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def resize(dataset_dir, image_scales):
    img_dir = f"{dataset_dir}/images"
    for image_path in tqdm.tqdm(glob(f"{img_dir}/*.png"), desc=f'Resize Image of {",".join([str(i) for i in image_scales])}'):
        image_path = image_path.replace('\\', '/')
        image = make_divisible(imageio.imread(image_path), max(image_scales))
        for scale in image_scales:
            output_path = f"{img_dir}_{scale}"
            os.makedirs(output_path, exist_ok=True)
            save_image(f'{output_path}/{image_path.split("/")[-1]}', image_to_uint8(downsample_image(image, scale)))


def save_image(path, image: np.ndarray, extension='png') -> None:
    with open(path, 'wb') as f:
        image = PIL_Image.fromarray(np.asarray(image))
        image.save(f, format=extension)


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert the image to a uint8 array."""
    if image.dtype == np.uint8:
        return image
    if not issubclass(image.dtype.type, np.floating):
        raise ValueError(f'Input image should be a floating type but is of type {image.dtype!r}')
    return (image * 255).clip(0.0, 255).astype(np.uint8)


def make_divisible(image: np.ndarray, divisor: int) -> np.ndarray:
    """Trim the image if not divisible by the divisor."""
    height, width = image.shape[:2]
    if height % divisor == 0 and width % divisor == 0:
        return image

    new_height = height - height % divisor
    new_width = width - width % divisor

    return image[:new_height, :new_width]


def downsample_image(image: np.ndarray, scale: int) -> np.ndarray:
    """Downsamples the image by an integer factor to prevent artifacts."""
    if scale == 1:
        return image

    height, width = image.shape[:2]
    if height % scale > 0 or width % scale > 0:
        raise ValueError(f'Image shape ({height},{width}) must be divisible by the scale ({scale}).')
    out_height, out_width = height // scale, width // scale
    resized = cv2.resize(image, (out_width, out_height), cv2.INTER_AREA)
    return resized
