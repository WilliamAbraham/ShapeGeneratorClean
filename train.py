import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from models import Model
from utils import loss_function, load_mat, show_image

if __name__ == '__main__':

    """
        A simple implementation of Gaussian MLP Encoder and Decoder
    """
    cuda = False
    device = torch.device("cuda" if cuda else "cpu")
    train_model = True
    im_x = 50
    im_y = 50 
    model_type = 'CNN'
    dataset_path = './datasets/Wang/ShapeSpace.mat'
    batch_size = 32
    x_dim  = 2500
    hidden_dim = 32
    latent_dim = 16
    lr = 1e-3
    epochs = 30
    mnist_transform = transforms.Compose([
            transforms.ToTensor(),
    ])

    kwargs = {'num_workers': 1, 'pin_memory': False} 

    ### load training and testing data from dataset, testing data account for 10%
    mat_data = load_mat(dataset_path)
    dataset = mat_data['ShapeSpace'].astype(np.float32)  # (50, 50, 248396)
    dataset = np.transpose(dataset, (2, 0, 1))  # --> (248396, 50, 50)
    dataset = dataset[:256]  # Use only first 256 samples
    dataset = dataset.reshape(-1, 1, im_x, im_y)  # --> (256, 1, 50, 50)
    print(dataset.shape)


    X_train, X_test = train_test_split(dataset, test_size=0.1, random_state=42)

    X_train = torch.tensor(X_train).to(torch.float32)
    X_test = torch.tensor(X_test).to(torch.float32)

    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(X_test, batch_size=batch_size, shuffle=False, **kwargs)

        
    model = Model(x_dim, hidden_dim, latent_dim,device,model_type,im_x,im_y).to(device)

    if train_model:
        optimizer = Adam(model.parameters(), lr=lr)
        print("Start training VAE...")
        model.train()
        for epoch in range(epochs):
            overall_loss = 0
            for batch_idx, x in enumerate(train_loader):
                #x = x.view(batch_size, x_dim)
                x = x.to(device)
                print(x.shape)

                optimizer.zero_grad()

                x_hat, mean, log_var = model(x)

                loss = loss_function(
                    x.view(x.size(0), x_dim),
                    x_hat.view(x.size(0), x_dim),
                    mean,
                    log_var
                )

                overall_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
            print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
            
        print("Finish!!")
        torch.save(model, './checkpoints/metalattice_model.pth')
    
    else:
        loaded_model = torch.load('./checkpoints/metalattice_model.pth')
        loaded_model.eval()
        with torch.no_grad():
            for batch_idx, x in enumerate(tqdm(test_loader)):
                print(x.shape)
                #x = x.view(batch_size, x_dim)
                x = x.to(device)
                
                x_hat, _, _ = loaded_model(x)
                #break
        
        show_image(x[0])
        show_image(x_hat[0].view(im_x,im_y))

        with torch.no_grad():
            noise = torch.randn(batch_size, latent_dim).to(device)
            generated_images = loaded_model.Decoder(noise)
        
        show_image(generated_images[0].view(im_x,im_y))