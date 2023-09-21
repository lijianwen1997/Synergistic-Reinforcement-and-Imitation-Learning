import torch
import torchvision
from torch.utils.data import DataLoader

import csv
import os.path as osp

from dataset import FluvialDataset, InputChannelConfig
from build_dataset import abs_path
from vae import VAE


IMG_HEIGHT, IMG_WIDTH = 128, 128

"""
Specify the demo id you want to encode
"""
demo_id = 0

demo_num = 84

CHANNEL_CONFIG = InputChannelConfig.RGB_ONLY

MODEL_LOAD_NAME = 'vae-sim-rgb-all.pth'
MODEL_LOAD_PATH = abs_path('models/' + MODEL_LOAD_NAME)

# dataset path for initing FluvialDataset class
TEST_DATASET_RELA_PATH = f'../riverine_simulation/demonstration/demo{demo_id}/demo{demo_id}.csv'
# trajectory path. each trajectory is a sequence of [image_path, action, reward, done]
SRC_CSV_PATH = abs_path(f'../riverine_simulation/demonstration/demo{demo_id}/traj.csv')
# target trajectory path where all image paths are replaced with encoded vectors
DST_CSV_PATH = osp.join(osp.dirname(SRC_CSV_PATH), CHANNEL_CONFIG.name + '.csv')

# VAE constants
latent_dim = 1024
hidden_dims = [32, 64, 128, 256, 512, 1024]
batch_size = 1


def update_path(demo_id: int):
    global TEST_DATASET_RELA_PATH, SRC_CSV_PATH, DST_CSV_PATH
    TEST_DATASET_RELA_PATH = f'../riverine_simulation/demonstration/demo{demo_id}/demo{demo_id}.csv'
    SRC_CSV_PATH = abs_path(f'../riverine_simulation/demonstration/demo{demo_id}/traj.csv')
    DST_CSV_PATH = osp.join(osp.dirname(SRC_CSV_PATH), CHANNEL_CONFIG.name + '.csv')


def resize(image_tensor, h=IMG_HEIGHT, w=IMG_WIDTH):
    new_image_tensor = torchvision.transforms.functional.resize(image_tensor, size=[h, w])
    return new_image_tensor


if __name__ == '__main__':
    # init model
    model = VAE(in_channels=CHANNEL_CONFIG.value, latent_dim=latent_dim, hidden_dims=hidden_dims)
    model.eval()

    # load state dict
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=torch.device('cpu')))
    print(f'Model {MODEL_LOAD_NAME} is loaded!')

    for idx in range(demo_num):
        update_path(idx)  # update input and output trajectory paths

        if not osp.exists(SRC_CSV_PATH):
            print(f'{SRC_CSV_PATH} does not exist!')
            continue

        # load test dataset
        test_dataset = FluvialDataset(TEST_DATASET_RELA_PATH, transform=resize, target_transform=resize,
                                      channel_config=CHANNEL_CONFIG)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # read actions to a list
        all_actions, all_rewards, all_dones = [], [], []
        with open(SRC_CSV_PATH, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=',')
            for line in reader:
                all_actions.append(line[1])
                all_rewards.append(line[2])
                all_dones.append(line[3])
        # print(f'{all_actions=}')

        assert len(all_actions) == len(test_dataloader), f'action len {len(all_actions)} does not match dataloader len {len(test_dataloader)}'

        # get encoded latent vector and save it to csv file
        with open(DST_CSV_PATH, 'w+', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            for i, img in enumerate(test_dataloader):
                latent_vec = model.encode(img)[0].tolist()[0]
                writer.writerow([latent_vec, all_actions[i], all_rewards[i], all_dones[i]])
        print(f'Image encoding and vec-act-rew-done writing are finished!')

