from pathlib import Path
from tqdm import tqdm
import cv2

def generate_video(num_frames, data_name, model_name,fps):
    image_files = []
    model_path = Path("./result") / data_name / model_name / "img"
    video_path = Path("./result") / data_name / model_name / "video"
    video_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    for i in range(1, num_frames + 1):
        image_files.append(f"{i}_fitting.png")

    # Define the output video file name
    filename = "video.mp4"

    # Get the size of the first image dynamically
    first_image_path = model_path / image_files[0]
    first_image = cv2.imread(str(first_image_path))
    height, width, _ = first_image.shape  # Extract the size of the first image

    # Create the video writer with the actual image dimensions
    output_size = (width, height)
    video = cv2.VideoWriter(str(video_path / filename), cv2.VideoWriter_fourcc(*'mp4v'), fps, output_size)

    # Add images to the video writer
    for image_file in tqdm(image_files, desc="Processing images", unit="image"):
        image_path = model_path / image_file
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Warning: Could not read {image_path}, skipping this image.")
            continue
               
        video.write(image)
    
    # Finalize and close the video writer
    video.release()
    print("MP4 video created successfully.")

if __name__ == "__main__":
    num_frames = 120
    data_name = "Beauty"
    model_name = "GaussianImage_Cholesky"
    fps=24
    generate_video(num_frames, data_name, model_name,fps)
