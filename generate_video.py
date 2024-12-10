import cv2
import os

def convert_images_to_video(input_folder, output_file, fps):
    # Get the list of image files in the input folder
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')])

    # Read the first image to get its dimensions
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, _ = first_image.shape

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec for the output video file
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Iterate over each image and write it to the video
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the video writer and close the video file
    video.release()
    cv2.destroyAllWindows()

# Provide the path to the input image folder, output video file, and desired FPS
input_folder = r'/home/ariele/cameras/capture03/video_02_c03'
output_file = r'/home/ariele/cameras/capture03/video_upper.mp4'
fps = 24  # Frames per second

# Call the function to convert the images to video
convert_images_to_video(input_folder, output_file, fps)