"""
Script to push images to GitHub in chunks to avoid size limits
Run with: python push_images_chunks.py
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and return success status"""
    print(f"\n{'='*60}")
    if description:
        print(f"📌 {description}")
    print(f"Running: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        return False

def get_image_files(folder_path):
    """Get all image files from a folder"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    image_files = []
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Folder not found: {folder_path}")
        return []
    
    for file in sorted(folder.rglob('*')):
        if file.is_file() and file.suffix.lower() in image_extensions:
            image_files.append(str(file))
    
    return image_files

def get_folder_size(files):
    """Calculate total size of files in MB"""
    total_size = sum(os.path.getsize(f) for f in files)
    return total_size / (1024 * 1024)  # Convert to MB

def chunk_files(files, chunk_size_mb=100):
    """Split files into chunks based on size"""
    chunks = []
    current_chunk = []
    current_size = 0
    
    for file in files:
        file_size = os.path.getsize(file) / (1024 * 1024)  # MB
        
        if current_size + file_size > chunk_size_mb and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [file]
            current_size = file_size
        else:
            current_chunk.append(file)
            current_size += file_size
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def push_images_in_chunks(image_folder, chunk_size_mb=100):
    """Push images to Git in chunks"""
    
    print("🐕 Dog Matchmaker - Push Images in Chunks")
    print("="*60)
    
    # Verify Git LFS is set up
    print("\n📋 Checking Git LFS setup...")
    if not run_command("git lfs install", "Installing Git LFS"):
        print("⚠️ Git LFS installation failed. Please install it manually.")
        return False
    
    # Track images with Git LFS
    print("\n📋 Setting up Git LFS tracking...")
    run_command(f'git lfs track "{image_folder}/**/*.jpg"', "Tracking JPG files")
    run_command(f'git lfs track "{image_folder}/**/*.jpeg"', "Tracking JPEG files")
    run_command(f'git lfs track "{image_folder}/**/*.png"', "Tracking PNG files")
    run_command(f'git lfs track "{image_folder}/**/*.gif"', "Tracking GIF files")
    
    # Commit .gitattributes
    run_command("git add .gitattributes", "Adding .gitattributes")
    run_command('git commit -m "Configure Git LFS for images"', "Committing LFS config")
    
    # Get all image files
    print(f"\n📁 Scanning for images in {image_folder}...")
    image_files = get_image_files(image_folder)
    
    if not image_files:
        print(f"❌ No images found in {image_folder}")
        return False
    
    total_size = get_folder_size(image_files)
    print(f"✅ Found {len(image_files)} images")
    print(f"📊 Total size: {total_size:.2f} MB")
    
    # Split into chunks
    chunks = chunk_files(image_files, chunk_size_mb)
    print(f"\n📦 Split into {len(chunks)} chunks ({chunk_size_mb}MB each)")
    
    # Push each chunk
    for i, chunk in enumerate(chunks, 1):
        chunk_size = get_folder_size(chunk)
        print(f"\n{'='*60}")
        print(f"📤 Pushing chunk {i}/{len(chunks)} ({len(chunk)} files, {chunk_size:.2f}MB)")
        print(f"{'='*60}")
        
        # Add files in this chunk
        for file in chunk:
            run_command(f'git add "{file}"', f"Adding {os.path.basename(file)}")
        
        # Commit this chunk
        commit_msg = f"Add images chunk {i}/{len(chunks)}"
        if not run_command(f'git commit -m "{commit_msg}"', f"Committing chunk {i}"):
            print(f"⚠️ Nothing to commit in chunk {i}, skipping...")
            continue
        
        # Push this chunk
        if not run_command("git push origin main", f"Pushing chunk {i} to GitHub"):
            print(f"\n❌ Failed to push chunk {i}")
            print("You may need to:")
            print("  1. Check your internet connection")
            print("  2. Verify GitHub authentication")
            print("  3. Check repository permissions")
            response = input("\nRetry this chunk? (y/n): ")
            if response.lower() == 'y':
                if not run_command("git push origin main", f"Retrying push for chunk {i}"):
                    print(f"❌ Retry failed. Stopping.")
                    return False
            else:
                return False
        
        print(f"✅ Chunk {i}/{len(chunks)} pushed successfully!")
    
    print("\n" + "="*60)
    print("🎉 All images pushed successfully!")
    print("="*60)
    return True

def main():
    # Configuration
    IMAGE_FOLDER = "Dog-Breeds-Dataset/Images"  # FULL IMAGE FOLDER (2.92GB)
    CHUNK_SIZE_MB = 80  # Push 80MB at a time to stay under limits
    
    print("🐕 Welcome to the Image Chunk Pusher!")
    print(f"📁 Target folder: {IMAGE_FOLDER}")
    print(f"📦 Chunk size: {CHUNK_SIZE_MB}MB")
    print(f"⚠️  NOTE: This will push the FULL image dataset (~2.92GB)")
    print(f"💰 GitHub LFS free tier is 1GB - you'll need a paid plan!")
    
    # Check if folder exists
    if not os.path.exists(IMAGE_FOLDER):
        print(f"\n❌ Error: Folder '{IMAGE_FOLDER}' not found!")
        print("Please make sure you're running this from the project root directory.")
        sys.exit(1)
    
    # Calculate and show cost
    print("\n💵 COST ESTIMATE:")
    print("   GitHub LFS Pricing:")
    print("   - Free: 1GB storage + 1GB bandwidth/month")
    print("   - Paid: $5/month for 50GB storage + 50GB bandwidth")
    print("   - Your dataset: ~2.92GB (requires paid plan)")
    print("\n   Alternative: Keep images in .gitignore and use cloud storage")
    
    # Confirm before starting
    print("\n⚠️  Are you sure you want to push 2.92GB to GitHub LFS?")
    print("   This will require a paid GitHub plan ($5/month).")
    response = input("Continue? (yes/no): ")
    
    if response.lower() != 'yes':
        print("\n❌ Cancelled.")
        print("\n💡 TIP: Consider these alternatives:")
        print("   1. Keep using Images-Sample (216MB) - stays in free tier")
        print("   2. Use Google Drive, Dropbox, or AWS S3 for full dataset")
        print("   3. Upload to Kaggle and download in your app setup")
        sys.exit(0)
    
    # Push images
    success = push_images_in_chunks(IMAGE_FOLDER, CHUNK_SIZE_MB)
    
    if success:
        print("\n✅ Done! Your images are now on GitHub.")
        print("🚀 Streamlit Cloud will automatically redeploy your app.")
        print("💰 Remember: You'll need to upgrade to GitHub LFS paid plan")
    else:
        print("\n❌ Push incomplete. Please resolve errors and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()