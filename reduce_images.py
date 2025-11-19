import os
import shutil
from pathlib import Path

source = Path("Dog-Breeds-Dataset/Images")
dest = Path("Dog-Breeds-Dataset/Images-Sample")
dest.mkdir(exist_ok=True)

# Also copy the CSV and other files
for file in source.glob("*.*"):
    if file.is_file():
        shutil.copy(file, dest / file.name)

count = 0
breed_count = 0
for breed_folder in source.iterdir():
    if breed_folder.is_dir() and breed_folder.name.endswith(" dog"):
        # Create breed folder in sample
        (dest / breed_folder.name).mkdir(exist_ok=True)
        
        # Copy only first 2 images
        images = list(breed_folder.glob("*.jpg")) + list(breed_folder.glob("*.jpeg")) + list(breed_folder.glob("*.png"))
        for img in images[:2]:
            shutil.copy(img, dest / breed_folder.name / img.name)
            count += 1
        breed_count += 1

print(f"\n✅ Sample dataset created!")
print(f"📁 Copied {count} images from {breed_count} breeds")
print(f"💾 Estimated size: ~{(count * 50 / 1024):.1f} MB")