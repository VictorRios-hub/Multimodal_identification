import cv2
import os

# Directory containing the images
image_directory = 'archive/class_35/video/sa2'

# Get a list of image files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith('.jpeg') or f.endswith('.png')]

# Sort the image files to maintain order
image_files.sort()

# Video settings
output_video = 'output_video_last.mp4'
frame_rate = 30  # Adjust as needed

# Get the first image to set the video dimensions
first_image = cv2.imread(os.path.join(image_directory, image_files[0]))
height, width, layers = first_image.shape

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

# Loop through the images and add them to the video
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    image = cv2.imread(image_path)
    video.write(image)

# Release the VideoWriter and close the video file
video.release()

print('Video created successfully!')
