import os
from PTLF.utils import DataSet
from torchvision import transforms
from PIL import Image
import torch



class DS1(DataSet):
    def __init__(self):
        super().__init__()
        self.args = {'data_src'}

    def _setup(self, args):
        self.root_dir = args['data_src']
        
        # Define default transformations (resize, to tensor, and normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to 224x224 pixels
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
        
        # Get all class names (subfolder names)
        self.class_names = sorted(os.listdir(self.root_dir))
        
        # Create a mapping from class name to class index
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        
        # Collect all image paths and their corresponding class labels
        self.image_paths = []
        self.labels = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.endswith(('.png', '.jpg', '.jpeg')):  # You can adjust extensions
                        self.image_paths.append(os.path.join(class_dir, filename))
                        self.labels.append(self.class_to_idx[class_name])
    @staticmethod
    def collate_fn(batch):
        inputs, labels = zip(*batch)  # Unzip the batch into inputs and labels
        inputs = torch.stack(inputs, dim=0)  # Stack inputs to create a tensor
        labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to tensor of type long (integer)
        
        return inputs, labels
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")  # Convert to RGB to ensure consistent color channels
        
        # Get the label for the image
        label = self.labels[idx]
        
        # Apply transformations if provided
        img = self.transform(img)
        
        return img, label