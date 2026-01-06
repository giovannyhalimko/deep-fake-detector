import os
import urllib.request
import ssl

# Bypass SSL verification if needed (common on Mac dev environments)
ssl._create_default_https_context = ssl._create_unverified_context

TAG = "0.0.1"
BASE_URL = f"https://github.com/selimsef/dfdc_deepfake_challenge/releases/download/{TAG}"

WEIGHTS = [
    "final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36",
    "final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19",
    "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_29",
    "final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31",
    "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_37",
    "final_888_DeepFakeClassifier_tf_efficientnet_b7_ns_0_40",
    "final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23"
]

def download_weights():
    # Ensure weights directory exists
    if not os.path.exists("weights"):
        os.makedirs("weights")
        print("Created weights/ directory")

    print(f"Downloading {len(WEIGHTS)} model weights...")
    
    for weight_name in WEIGHTS:
        url = f"{BASE_URL}/{weight_name}"
        output_path = os.path.join("weights", weight_name)
        
        if os.path.exists(output_path):
            print(f"Skipping {weight_name} (already exists)")
            continue
            
        print(f"Downloading {weight_name}...")
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"✓ Downloaded {weight_name}")
        except Exception as e:
            print(f"✗ Failed to download {weight_name}: {e}")

    print("\nAll downloads complete!")

if __name__ == "__main__":
    download_weights()
