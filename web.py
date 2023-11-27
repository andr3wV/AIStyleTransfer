import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Setting up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Loading the VGG19 model
model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
for p in model.parameters():
    p.requires_grad = False
model.to(device)

# Function to get model activations
def model_activations(input, model):
    layers = {
        '0' : 'conv1_1',
        '5' : 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '24': 'conv4_3',
        '21': 'conv4_2',
        '28': 'conv5_1'
    }
    features = {}
    x = input
    x = x.unsqueeze(0)
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x 
    return features

# Function for image conversion
def imcnvt(image):
    x = image.to("cpu").clone().detach().numpy().squeeze()
    x = x.transpose(1, 2, 0)
    x = x * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    return np.clip(x, 0, 1)

# Function to calculate gram matrix
def gram_matrix(imgfeature):
    _, d, h, w = imgfeature.size()
    imgfeature = imgfeature.view(d, h * w)
    gram_mat = torch.mm(imgfeature, imgfeature.t())
    return gram_mat

# Function to perform style transfer
def perform_style_transfer(style_image_path, content_image_path, output_image_path):
    # Image transformation
    transform = transforms.Compose([transforms.Resize(300),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Loading content and style images
    content = Image.open(content_image_path).convert("RGB")
    content = transform(content).to(device)
    style = Image.open(style_image_path).convert("RGB")
    style = transform(style).to(device)

    # Style and content features
    style_features = model_activations(style, model)
    content_features = model_activations(content, model)

    # Weights for different layers
    style_wt_meas = {"conv1_1" : 1.0, 
                     "conv2_1" : 0.8,
                     "conv3_1" : 0.4,
                     "conv4_1" : 0.2,
                     "conv5_1" : 0.1}

    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    # Weighting factors for style and content loss
    content_wt = 100
    style_wt = 1e8

    # Target image
    target = content.clone().requires_grad_(True).to(device)

    # Optimizer
    optimizer = torch.optim.Adam([target], lr=0.007)

    # Style transfer iterations
    epochs = 4800
    for i in range(1, epochs + 1):
        target_features = model_activations(target, model)
        content_loss = torch.mean((content_features['conv4_2'] - target_features['conv4_2'])**2)

        style_loss = 0
        for layer in style_wt_meas:
            style_gram = style_grams[layer]
            target_gram = target_features[layer]
            _, d, w, h = target_gram.shape
            target_gram = gram_matrix(target_gram)

            style_loss += (style_wt_meas[layer] * torch.mean((target_gram - style_gram)**2)) / d * w * h

        total_loss = content_wt * content_loss + style_wt * style_loss

        if i%10==0:       
            print("epoch ",i," ", total_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 10 == 0:
            output_epoch_path = os.path.join('output', f'output_epoch_{i}.jpg')
            img = imcnvt(target)
            plt.imsave(output_epoch_path, img, format='jpg')


    # Save the final output image
    plt.imsave(output_image_path, imcnvt(target), format='jpg')