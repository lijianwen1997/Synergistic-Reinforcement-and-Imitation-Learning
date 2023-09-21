import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import FluvialDataset, InputChannelConfig
from build_dataset import abs_path
from vae import VAE

IMG_HEIGHT, IMG_WIDTH = 128, 128

# specify any csv file that contains image paths
TEST_DATASET_RELA_PATH = '../riverine_simulation/demonstration/demo0/demo0.csv'

CHANNEL_CONFIG = InputChannelConfig.RGB_ONLY

MODEL_LOAD_NAME = 'vae-sim-rgb-all.pth'
MODEL_LOAD_PATH = abs_path('models/' + MODEL_LOAD_NAME)

# VAE constants
latent_dim = 1024
hidden_dims = [32, 64, 128, 256, 512, 1024]
batch_size = 10


def resize(image_tensor, h=IMG_HEIGHT, w=IMG_WIDTH):
    new_image_tensor = torchvision.transforms.functional.resize(image_tensor, size=[h, w])
    return new_image_tensor


if __name__ == '__main__':
    # load test dataset
    test_dataset = FluvialDataset(TEST_DATASET_RELA_PATH, transform=resize, target_transform=resize,
                             channel_config=CHANNEL_CONFIG)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # init model
    model = VAE(in_channels=CHANNEL_CONFIG.value, latent_dim=latent_dim, hidden_dims=hidden_dims)
    model.eval()

    # load state dict
    model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=torch.device('cpu')))
    print(f'Model {MODEL_LOAD_NAME} is loaded!')

    # get reconstructed images
    source = next(iter(test_dataloader))
    reconstruction = model.generate(source)
    print(f'Reconstruction finished!')

    # visualize sample data
    plt.figure(figsize=(20, 4))
    for index in range(batch_size):
        # display original
        ax = plt.subplot(2, batch_size, index + 1)
        plt.imshow(source[index].permute((1, 2, 0)).detach().numpy(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, batch_size, index + 1 + batch_size)
        plt.imshow(reconstruction[index].permute((1, 2, 0)).detach().numpy(), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()




