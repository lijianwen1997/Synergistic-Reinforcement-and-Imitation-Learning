import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset import FluvialDataset, InputChannelConfig
from build_dataset import abs_path
from vae import VAE


"""
Define IO constants
"""
IMG_HEIGHT, IMG_WIDTH = 128, 128

# need to specify the csv file containing image paths to init FluvialDataset class
DATASET_PATH = ''

CHANNEL_CONFIG = InputChannelConfig.RGB_ONLY

MODEL_SAVE_NAME = 'vae-sim-rgb-all.pth'

MODEL_SAVE_PATH = abs_path('models/' + MODEL_SAVE_NAME)

"""
Seed constants
"""
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

"""
Training constants
"""
latent_dim = 1024
hidden_dims = [32, 64, 128, 256, 512, 1024]
batch_size = 128
epochs = 100
learning_rate = 1e-3


def resize(image_tensor, h=IMG_HEIGHT, w=IMG_WIDTH):
    new_image_tensor = torchvision.transforms.functional.resize(image_tensor, size=[h, w])
    return new_image_tensor


if __name__ == '__main__':
    # load dataset
    dataset = FluvialDataset(DATASET_PATH, transform=resize, target_transform=resize,
                             channel_config=CHANNEL_CONFIG)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # init model
    model = VAE(in_channels=CHANNEL_CONFIG.value, latent_dim=latent_dim, hidden_dims=hidden_dims)
    if torch.cuda.is_available():
        model.cuda()

    # init optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # start training
    print(f'Start training ...')
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(dataloader):
            img = data
            if torch.cuda.is_available():
                img = img.cuda()
            optimizer.zero_grad()
            recon_batch, _, mu, logvar = model(img)
            loss = model.loss_function(recon_batch, img, mu, logvar)['loss']
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print('Epoch: {}, Average loss: {:.4f}'.format(epoch, train_loss / len(dataset)))
        # if epoch % 10 == 0:
        #     save = to_img(recon_batch.cpu().data)
        #     save_image(save, './vae_img/image_{}.png'.format(epoch))

    print(f'Training finished!')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'Model saved!')
