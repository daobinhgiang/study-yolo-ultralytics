import os
from PIL import Image, UnidentifiedImageError

def clean_images(directory, keep_ext='.jpg'):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(keep_ext):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        # If not JPEG, convert and save as JPEG (optional)
                        if img.format != 'JPEG':
                            # os.remove(file_path)
                            print(f"[REMOVE] {file_path}: Format is {img.format}, not JPEG")
                            os.remove(file_path)
                        else:
                            # Try re-saving to fix minor corruptions
                            img.save(file_path, "JPEG")
                except (UnidentifiedImageError, OSError) as e:
                    print(f"[CORRUPT] {file_path}: {e}")
                    os.remove(file_path)

if __name__ == "__main__":
    clean_images("split_images")
