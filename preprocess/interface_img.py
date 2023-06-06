import os
from preprocess._utils import run_cmd


def img_colmap(dataset_dir, camera_model, colmap_version, matching_method='vocab_tree', gpu=False):
    """
    :param dataset_dir:
    :param camera_model: C
    :param matching_method:
        Feature matching method to use. Vocab tree is recommended for a balance of speed
        and accuracy. Exhaustive is slower but more accurate. Sequential is faster but
        should only be used for videos.
    :return:
    """
    img_dir = f"{dataset_dir}/img"
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
    print(f"Feature extractor DONE : {out}")

    feature_matcher_cmd = [
        f"colmap {matching_method}_matcher",
        f"--database_path {colmap_dir}/database.db",
        f"--SiftMatching.use_gpu {int(gpu)}",
    ]
    if matching_method == "vocab_tree":
        vocab_tree_filename = "C:/vocab-tree/vocab_tree_flickr100K_words32K.bin"
        feature_matcher_cmd.append(f"--VocabTreeMatching.vocab_tree_path {vocab_tree_filename}")

    feature_matcher_cmd = " ".join(feature_matcher_cmd)
    run_cmd(feature_matcher_cmd)
    print(f"Feature matcher DONE : {out}")

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
    print(f"Mapper DONE : {out}")

    bundle_adjuster_cmd = [
        f"colmap bundle_adjuster",
        f"--input_path {sparse_dir}/0",
        f"--output_path {sparse_dir}/0",
        "--BundleAdjustment.refine_principal_point 1",
    ]
    out = run_cmd(" ".join(bundle_adjuster_cmd))
    print(f"Bundle Adjustment DONE : {out}")

