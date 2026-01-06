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

dataset_real = find_dir(path, "real")

dest_dir = "/Users/sam/Documents/GitHub/deep-fake-detector/datasets/real"
existing_files = set(os.listdir(dest_dir))
target_count = 5
current_count = len([f for f in existing_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
needed = target_count - current_count

if needed > 0 and dataset_real:
    all_files = [f for f in os.listdir(dataset_real) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    available = [f for f in all_files if f not in existing_files]
    
    if available:
        selected = random.sample(available, min(len(available), needed))
        print(f"Copying {len(selected)} new real images...")
        for f in selected:
            shutil.copy2(os.path.join(dataset_real, f), os.path.join(dest_dir, f))
            print(f"Copied {f}")
    else:
        print("No new files available to copy.")
else:
    print(f"No more files needed (current: {current_count}) or source not found.")
