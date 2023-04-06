import os
import json
import imageio
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

os.chdir('..')

# datatype = 'nerf_llff_data'
# dataset = 'room'

datatype = 'nerf_synthetic'
dataset = 'chair'

datadir = f'./dataset/{datatype}/{dataset}'
factor = 8
bd_factor=.75
path_zflat=False


# imgdir = os.path.join(datadir, 'images')
imgdir = os.path.join(datadir, 'train')

img_paths = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

# sh = imageio.imread(img_paths[0]).shape


def load_pose_bd():
    if datatype == 'nerf_llff_data':
        poses_arr = np.load(os.path.join(datadir, 'poses_bounds.npy'))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])
        poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        bds = poses_arr[:, -2:]
    else:
        with open(f'{datadir}/transforms_train.json') as f:
            raw =json.load(f)
        focal_length = raw['camera_angle_x']
        frames = raw['frames']
        bds = None
        poses = []
        for frame in frames:
            poses.append(frame['transform_matrix'])
        poses = np.array(poses)
        poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    return poses, bds


def normalize(x):
    return x / np.linalg.norm(x)


poses, bds = load_pose_bd()
# Sample Case : llff data

case = 99
img_path = [i for i in img_paths if f'_{case}' in i][0]
print(img_path)
# img = imageio.imread(img_paths[case])
img = cv2.imread(img_path)
font = cv2.FONT_HERSHEY_DUPLEX
pose = poses[case]
# bd = bds[case]
for i in range(3):
    cv2.putText(img, ' '.join([str(j)[:5] for j in pose[i]]), (10, 90+i*30), font, 0.5,(0,0,155), 1, cv2.LINE_AA)
cv2.imshow('img', img)


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

trans = poses[:, :3, 3]
print(poses.shape)
print(trans.shape)
ax.scatter(trans[:, 0], trans[:, 1], trans[:, 2])
# ax.scatter(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3])
for i in range(len(poses)):
    pose = poses[i, :3, :3]
    tran = trans[i]
    direct = - pose @ np.array([0, 0, 1])
    direct = normalize(direct)*1
    ray = tran+direct
    x,y,z = zip(*np.array([tran, ray]))
    color ='red'
    if i == case:
        color = 'blue'
    ax.plot3D(x, y, z, color)
    ax.text(tran[0], tran[1], tran[2], str(i))
plt.show()

