import os
import shutil
import random
import kagglehub

# Reuse the path we know or fetch it again (it serves as a cache lookup)
path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")

def find_dir(base, name):
    for root, dirs, files in os.walk(base):
        for d in dirs:
            if d.lower() == name.lower():
                return os.path.join(root, d)
    return None

# We want the Fake directory. 
# Based on previous output, it seemed to be deeper, but find_dir should work.
dataset_fake = find_dir(path, "fake")

dest_dir = "/Users/sam/Documents/GitHub/deep-fake-detector/datasets/fake"
existing_files = set(os.listdir(dest_dir))
target_count = 5
current_count = len([f for f in existing_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
needed = target_count - current_count

if needed > 0 and dataset_fake:
    all_files = [f for f in os.listdir(dataset_fake) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # Filter out files that are already in dest_dir (to avoid re-copying existing ones)
    # AND filter out the ones we specifically deleted if possible? 
    # The user deleted specific ones because they didn't work. 
    # Since I don't have the list of deleted ones in this script, just random sampling from the large dataset 
    # is the best bet. The dataset has thousands of images.
    
    available = [f for f in all_files if f not in existing_files]
    
    if available:
        selected = random.sample(available, min(len(available), needed))
        print(f"Copying {len(selected)} new fake images...")
        for f in selected:
            shutil.copy2(os.path.join(dataset_fake, f), os.path.join(dest_dir, f))
            print(f"Copied {f}")
    else:
        print("No new files available to copy.")
else:
    print(f"No more files needed (current: {current_count}) or source not found.")
