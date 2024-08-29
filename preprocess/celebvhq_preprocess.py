# parsing labels, segment and crop raw videos.
import argparse
import os
import sys
import glob
from pathlib import Path
sys.path.append(os.getcwd())


def crop_face(root: str):
    from util.face_sdk.face_crop import process_videos
    source_dir = os.path.join(root, "video")
    target_dir = os.path.join(root, "cropped")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    exps = glob.glob(os.path.join(source_dir) + '/*')
    print(len(exps), ' exps in total')
    for exp in exps:
        print('Processing for ', exp)
        # file_name = file.split('/')[-1]
        # single_viddeo_path = file.replace(file_name, '')
        Path(exp.replace('video', 'cropped')).mkdir(parents=True, exist_ok=True)

        process_videos(exp, exp.replace('video', 'cropped'), ext="mp4")


# def gen_split(root: str):
#     videos = list(filter(lambda x: x.endswith('.mp4'), os.listdir(os.path.join(root, 'cropped'))))
#     total_num = len(videos)

#     with open(os.path.join(root, "train.txt"), "w") as f:
#         for i in range(int(total_num * 1)):
#             f.write(videos[i][:-4] + "\n")

    # with open(os.path.join(root, "val.txt"), "w") as f:
    #     for i in range(int(total_num * 0.8), int(total_num * 0.9)):
    #         f.write(videos[i][:-4] + "\n")

    # with open(os.path.join(root, "test.txt"), "w") as f:
    #     for i in range(int(total_num * 0.9), total_num):
    #         f.write(videos[i][:-4] + "\n")


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="Root directory of CelebV-HQ")
args = parser.parse_args()

if __name__ == '__main__':
    data_root = args.data_dir
    crop_face(data_root)

    # if not os.path.exists(os.path.join(data_root, "train.txt")) or \
    #     not os.path.exists(os.path.join(data_root, "val.txt")) or \
    #     not os.path.exists(os.path.join(data_root, "test.txt")):
    #     gen_split(data_root)
    # if not os.path.exists(os.path.join(data_root, "train.txt")):
    #     gen_split(data_root)
