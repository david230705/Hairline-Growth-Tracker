from data.data_manager import DataManager

# Test the data manager
dm = DataManager()

# Create sample images
dm.create_sample_images()

# Validate one of the sample images
is_valid, message = dm.validate_image("data/input/raw_images/photo.jpg")
print(f"Image valid: {is_valid}, Message: {message}")

# See what images are available
valid, invalid = dm.batch_process_images()
print(f"Valid images: {len(valid)}")