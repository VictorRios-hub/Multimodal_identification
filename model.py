import torch
import torch.nn as nn
import torchvision.models.video as models
import librosa

class I3D(nn.Module):
    def __init__(self, num_classes=400):
        super(I3D, self).__init__()
        self.i3d = models.video.r3d_18(pretrained=True)  # You can choose other I3D architectures too
        # Remove the fully connected layers
        self.i3d.fc = nn.Identity()
        self.i3d.avgpool = nn.AdaptiveAvgPool3d(1)
        
    def forward(self, x):
        return self.i3d(x)

def compute_mfcc(audio_file, n_mfcc=13, hop_length=512, n_fft=2048):
    # Chargement du fichier audio
    y, sr = librosa.load(audio_file)
    
    # Extraction des MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    
    return mfccs
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = compute_mfcc(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        return x

class FusionModel(nn.Module):
    def __init__(self, num_classes, cnn_feature_size, mfcc_feature_size):
        super(FusionModel, self).__init__()
        
        # I3D model
        self.i3d_model = I3D(num_classes=num_classes)
        
        # CNN model
        self.cnn_model = CNN(num_classes=num_classes)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(cnn_feature_size + mfcc_feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, video_input, audio_input):
        # Forward pass through I3D model
        i3d_features = self.i3d_model(video_input)
        
        # Forward pass through CNN model
        cnn_features = self.cnn_model(audio_input)
        
        # Concatenate features along height
        combined_features = torch.cat((i3d_features, cnn_features), dim=1)
        
        # Flatten the features before passing through fully connected layers
        combined_features = combined_features.view(combined_features.size(0), -1)
        
        # Forward pass through fully connected layers
        output = self.fc(combined_features)
        
        return output