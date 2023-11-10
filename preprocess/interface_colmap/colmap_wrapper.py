import os
import subprocess
from pathlib import Path

# $ DATASET_PATH=/path/to/dataset

# $ colmap feature_extractor \
#    --database_path $DATASET_PATH/database.db \
#    --image_path $DATASET_PATH/images

# $ colmap exhaustive_matcher \
#    --database_path $DATASET_PATH/database.db

# $ mkdir $DATASET_PATH/sparse

# $ colmap mapper \
#     --database_path $DATASET_PATH/database.db \
#     --image_path $DATASET_PATH/images \
#     --output_path $DATASET_PATH/sparse


# $ mkdir $DATASET_PATH/dense
def run_colmap(basedir, match_type):
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')

    basedir = Path(basedir)
    colmap_path = basedir / 'colmap'
    db_path = '.\\' + str(colmap_path / 'database.db')
    db_path = db_path.replace('\\', '/')
    sparse_path = '.\\' + str(colmap_path / 'sparse')
    sparse_path = sparse_path.replace('\\', '/')
    img_path = '.\\' + str(basedir / 'images')
    img_path = img_path.replace('\\', '/')

    feature_extractor_args = [
        'colmap', 'feature_extractor',
        '--database_path', db_path,
        '--image_path', img_path,
        '--ImageReader.single_camera', '1',
        # '--SiftExtraction.use_gpu', '0',
    ]
    # feat_output = (subprocess.check_output(feature_extractor_args, universal_newlines=True))
    feat_output = (subprocess.run(feature_extractor_args, universal_newlines=True))
    logfile.write(feat_output)
    print('Features extracted')

    exhaustive_matcher_args = [
        'colmap', match_type,
        '--database_path', db_path,
    ]

    match_output = (subprocess.check_output(exhaustive_matcher_args, universal_newlines=True))
    logfile.write(match_output)
    print('Features matched')

    if not os.path.exists(sparse_path):
        os.makedirs(sparse_path)

    # mapper_args = [
    #     'colmap', 'mapper',
    #         '--database_path', os.path.join(basedir, 'database.db'),
    #         '--image_path', os.path.join(basedir, 'images'),
    #         '--output_path', os.path.join(basedir, 'sparse'),
    #         '--Mapper.num_threads', '16',
    #         '--Mapper.init_min_tri_angle', '4',
    # ]
    mapper_args = [
        'colmap', 'mapper',
        '--database_path', db_path,
        '--image_path', img_path,
        '--output_path', sparse_path,  # --export_path changed to --output_path in colmap 3.6
        '--Mapper.num_threads', '16',
        '--Mapper.init_min_tri_angle', '4',
        '--Mapper.multiple_models', '0',
        '--Mapper.extract_colors', '0',
    ]

    map_output = (subprocess.check_output(mapper_args, universal_newlines=True))
    logfile.write(map_output)
    logfile.close()
    print('Sparse map created')

    print('Finished running COLMAP, see {} for logs'.format(logfile_name))
