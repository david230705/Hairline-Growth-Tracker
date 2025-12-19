import urllib.request
import os

# Download a sample face image for testing
url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
save_path = "data/input/raw_images/lena_face.jpg"

print("📥 Downloading test face image...")
urllib.request.urlretrieve(url, save_path)
print(f"✅ Downloaded test image to: {save_path}")