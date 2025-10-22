#import your nessessary libreries here


from PTLF.utils import Model

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

class EffiB3(Model):
    def __init__(self):
        super().__init__()
        self.args = { 'dropout',  'fine_tune_last', 'pre_trained_path' }

    def _setup(self, args):

        # Step 1: Load EfficientNet-B3 base model (without pre-trained weights)
        self.base_model = models.efficientnet_b3(weights=None)  # No pretrained weights at first
        self.base_model.classifier = nn.Identity()  # Remove final classification layer
        
        # Step 2: Load the pre-trained weights into the base model
        state_dict = torch.load(args['pre_trained_path'], weights_only=True)
        # Remove classifier weights from state_dict to avoid the mismatch
        state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
        
        # Load the filtered state_dict into the model
        self.base_model.load_state_dict(state_dict)
        # print("EfficientNet-B3 weights loaded successfully.")
        
        # Custom layers added after the base model
        self.batch_norm = nn.BatchNorm1d(1536)  # Output size of EfficientNet-B3 before classifier is 1536
        self.fc1 = nn.Linear(1536, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args['dropout'])
        self.fc2 = nn.Linear(256, 74)
        
        # Step 3: Create a list of all parameters
        params = list(self.base_model.named_parameters())

        # Step 4: Freeze all layers except the last 10 layers
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False  # Freeze all layers initially

        # Step 5: Unfreeze the last 10 layers
        for name, param in params[-args['fine_tune_last']:]:
            param.requires_grad = True  # Unfreeze the last 10 layers
        
    def forward(self, x):
        # Forward pass through EfficientNet-B3 base model
        x = self.base_model(x)  # Get features from EfficientNet-B3
        x = self.batch_norm(x)   # Apply batch normalization
        x = self.fc1(x)          # Apply fully connected layer 1
        x = self.relu(x)         # ReLU activation
        x = self.dropout(x)      # Apply dropout
        x = self.fc2(x)          # Output layer
        return x

class ResN50(Model):
    def __init__(self):
        super().__init__()
        self.args = { 'dropout',  'fine_tune_last', 'pre_trained_path' }

    def _setup(self, args):

        # Step 1: Load ResNet50 base model (without pre-trained weights)
        self.base_model = models.resnet50(weights=None)  # No pretrained weights at first
        self.base_model.fc = nn.Identity()  # Remove the final fully connected (fc) layer

        # Step 2: Load the pre-trained weights into the base model
        state_dict = torch.load(args['pre_trained_path'], weights_only=True)
        self.base_model.load_state_dict(state_dict, strict=False)  # Uncomment if you have custom pretrained weights

        # Custom layers after the base model
        self.fc1 = nn.Linear(2048, 256)  # Updated: Custom fully connected layer for 2048 input features
        self.batch_norm = nn.BatchNorm1d(256)  # Apply BatchNorm after fc1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args['dropout'])
        self.fc2 = nn.Linear(256, 74)  # Output layer (num_classes will be 74)

        # Step 3: Create a list of all parameters
        params = list(self.base_model.named_parameters())

        # Step 4: Freeze all layers except the last 10 layers
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False  # Freeze all layers initially

        # Step 5: Unfreeze the last 10 layers
        for name, param in params[-args['fine_tune_last']:]:
            param.requires_grad = True  # Unfreeze the last 10 layers
        
    def forward(self, x):
        # Forward pass through ResNet50 base model
        x = self.base_model(x)  # Get features from ResNet50
        x = self.fc1(x)          # Apply fully connected layer 1
        x = self.batch_norm(x)   # Apply batch normalization after fc1
        x = self.relu(x)         # ReLU activation
        x = self.dropout(x)      # Apply dropout
        x = self.fc2(x)          # Output layer
        return x


class Vgg19(Model):
    def __init__(self):
        super().__init__()
        self.args = { 'dropout',  'fine_tune_last', 'pre_trained_path' }

    def _setup(self, args):

        # Step 1: Load VGG19 base model (without pre-trained weights)
        self.base_model = models.vgg19(weights=None)  # Load VGG19 without pre-trained weights
        self.base_model.classifier = nn.Sequential(*list(self.base_model.classifier.children())[:-1])  # Remove the final fully connected (fc) layer

        # Step 2: Load the pre-trained weights into the base model
        state_dict = torch.load(args['pre_trained_path'], weights_only=True)
        self.base_model.load_state_dict(state_dict, strict=False)  # Load the state_dict, assuming you have custom weights

        # Custom layers after the base model
        self.fc1 = nn.Linear(4096, 256)  # Custom fully connected layer for 4096 input features (VGG19 FC layer before output)
        self.batch_norm = nn.BatchNorm1d(256)  # Apply BatchNorm after fc1
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args['dropout'])
        self.fc2 = nn.Linear(256, 74)  # Output layer (num_classes will be 74)

        # Step 3: Create a list of all parameters
        params = list(self.base_model.named_parameters())

        # Step 4: Freeze all layers except the last 10 layers
        for name, param in self.base_model.named_parameters():
            param.requires_grad = False  # Freeze all layers initially

        # Step 5: Unfreeze the last 10 layers
        for name, param in params[-args['fine_tune_last']:]:
            param.requires_grad = True  # Unfreeze the last 10 layers

    def forward(self, x):
        # Forward pass through VGG19 base model
        x = self.base_model(x)  # Get features from VGG19
        x = self.fc1(x)         # Apply fully connected layer 1
        x = self.batch_norm(x)  # Apply batch normalization after fc1
        x = self.relu(x)        # ReLU activation
        x = self.dropout(x)     # Apply dropout
        x = self.fc2(x)         # Output layer
        return x

  