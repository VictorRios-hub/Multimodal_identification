import os
from PIL import Image
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, audio_transform=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.transform = transform
        self.audio_transform = audio_transform
        
    def __len__(self):
        return sum(len(os.listdir(os.path.join(self.root_dir, cls, 'audio'))) for cls in self.classes)
    
    def __getitem__(self, idx):
        class_idx = idx % len(self.classes)
        class_name = self.classes[class_idx]
        
        video_dir = os.listdir(os.path.join(self.root_dir, class_name, 'video'))
        video_dir = np.array(video_dir)
        video_files = os.listdir((os.path.join(video_dir[i])) for i in range(13) )
        audio_files = os.listdir(os.path.join(self.root_dir, class_name, 'audio'))
        
        video_file = video_files[idx // len(self.classes)]
        audio_file = audio_files[idx // len(self.classes)]
        
        video_path = os.path.join(self.root_dir, class_name, 'video', video_file)
        audio_path = os.path.join(self.root_dir, class_name, 'audio', audio_file)
        
        image = Image.open(video_path)
        audio, sample_rate = torchaudio.load(audio_path)
        
        if self.transform:
            image = self.transform(image)
        if self.audio_transform:
            audio = self.audio_transform(audio)
        
        return image, audio, class_idx

# Define transformations for image and audio data
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

audio_transform = torchaudio.transforms.MelSpectrogram()

# Create dataset and dataloader
root_dir = 'archive'
dataset = CustomDataset(root_dir, transform=image_transform, audio_transform=audio_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print(dataset.__getitem__(0))

# # Example usage in training loop
# for batch_idx, (images, audios, labels) in enumerate(dataloader):
#     # Your training code here
#     pass
