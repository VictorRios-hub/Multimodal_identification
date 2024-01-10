from moviepy.editor import VideoFileClip

# Path to the input MP4 video
input_video_path = 'output_video_last.mp4'

# Path to the output GIF
output_gif_path = 'output_gif.gif'

# Load the video clip
video_clip = VideoFileClip(input_video_path)

# Convert the video clip to a GIF
video_clip.write_gif(output_gif_path)

print('GIF created successfully!')
