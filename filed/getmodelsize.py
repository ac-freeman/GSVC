import os
import pandas as pd

# Base path to the 'models' directory
base_dir = 'models'

# Directory where the results will be saved
output_dir = 'modelsize'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Subfolders to process
subfolders = ['Beauty', 'HoneyBee', 'Jockey']

# Process each subfolder
for subfolder in subfolders:
    folder_path = os.path.join(base_dir, subfolder)
    
    # Initialize a list to store results for the current subfolder
    results = []
    
    # Loop through the directories inside each subfolder
    for dir_name in os.listdir(folder_path):
        if dir_name.startswith('GaussianImage_Cholesky_50000_'):
            # Extract the Gaussian count from the folder name
            gaussian_count = int(dir_name.split('_')[-1])
            
            # Check if the Gaussian count is a multiple of 5000
            if gaussian_count % 5000 == 0:
                # Construct the path to the model file
                model_file_path = os.path.join(folder_path, dir_name, 'gmodels_state_dict.pth')
                
                # Ensure the file exists before checking the size
                if os.path.exists(model_file_path):
                    # Get the file size in bytes
                    file_size = os.path.getsize(model_file_path)
                    
                    # Append the result to the list
                    results.append([gaussian_count, file_size])

    # Convert the results to a pandas DataFrame
    df = pd.DataFrame(results, columns=['Gaussian_Count', 'File_Size_Bytes'])
    
    # Save the DataFrame to a CSV file in the 'modelsize' directory
    output_file = os.path.join(output_dir, f'{subfolder}_gaussian_model_sizes.csv')
    df.to_csv(output_file, index=False)

    print(f"Data extraction complete for {subfolder}. Results saved to '{output_file}'.")
