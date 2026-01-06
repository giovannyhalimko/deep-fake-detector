import kagglehub
import shutil
import os
import random

# Create destination directories
base_dir = "/Users/sam/Documents/GitHub/deep-fake-detector/datasets"
real_dir = os.path.join(base_dir, "real")
fake_dir = os.path.join(base_dir, "fake")

os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

# Download dataset
print("Downloading dataset...")
path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")
print("Dataset downloaded to:", path)

# Robustly find the Real and Fake directories
def find_dir(base, name):
    for root, dirs, files in os.walk(base):
        for d in dirs:
            if d.lower() == name.lower():
                return os.path.join(root, d)
    return None

dataset_real = find_dir(path, "real")
dataset_fake = find_dir(path, "fake")

# Helper function to copy images
def copy_images(src_dir, dest_dir, count=5):
    if not os.path.exists(src_dir):
        print(f"Directory not found: {src_dir}")
        return
    
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print(f"No image files found in {src_dir}")
        return
        
    selected = random.sample(files, min(len(files), count))
    
    print(f"Copying {len(selected)} images from {src_dir} to {dest_dir}...")
    for f in selected:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dest_dir, f))

if not dataset_real or not dataset_fake:
    print("Could not find Real or Fake directories. Dumping structure:")
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        # for f in files[:5]:
        #     print('{}{}'.format(subindent, f))
else:
    copy_images(dataset_real, real_dir)
    copy_images(dataset_fake, fake_dir)

print("Done!")
