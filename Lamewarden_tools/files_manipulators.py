import os
import shutil

def create_directory(path):
    # Check if the directory exists
    if os.path.exists(path):
        # Remove the directory and its contents
        shutil.rmtree(path)
    # Create a new directory
    os.makedirs(path)
    print(f"Directory {path} created.")