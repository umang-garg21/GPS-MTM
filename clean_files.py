import os
import shutil
import glob

def clean_training_folders(outputs_path):
    """
    Clean training folders by keeping only model files and removing everything else.
    
    Args:
        outputs_path (str): Path to the outputs directory
    """
    # Model file extensions and patterns to keep
    model_extensions = ['.pth', '.pt', '.ckpt', '.bin', '.safetensors']
    model_patterns = ['model', 'checkpoint', 'best', 'final', 'weights']
    
    # Folders to completely preserve (if they contain models)
    preserve_folders = ['checkpoints', 'models', 'weights']
    
    cleaned_count = 0
    total_size_freed = 0
    
    # Walk through all subdirectories in outputs
    for root, dirs, files in os.walk(outputs_path):
        # Check if current directory is a training folder
        folder_name = os.path.basename(root)
        parent_folder = os.path.basename(os.path.dirname(root))
        
        # Only process if we're in a folder that starts with 'train'
        if folder_name.startswith('train') or parent_folder.startswith('train'):
            print(f"\n🔍 Processing training folder: {root}")
            
            files_to_remove = []
            dirs_to_remove = []
            
            # Check each file
            for file in files:
                file_path = os.path.join(root, file)
                file_lower = file.lower()
                
                # Check if it's a model file
                is_model_file = False
                
                # Check by extension
                if any(file_lower.endswith(ext) for ext in model_extensions):
                    is_model_file = True
                
                # Check by pattern in filename
                if any(pattern in file_lower for pattern in model_patterns):
                    is_model_file = True
                
                # If not a model file, mark for removal
                if not is_model_file:
                    files_to_remove.append(file_path)
            
            # Check each directory
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                dir_lower = dir_name.lower()
                
                # Check if it's a preserved folder
                is_preserved = any(preserve in dir_lower for preserve in preserve_folders)
                
                # If not preserved, mark for removal
                if not is_preserved:
                    dirs_to_remove.append(dir_path)
            
            # Remove files
            for file_path in files_to_remove:
                try:
                    file_size = os.path.getsize(file_path)
                    os.remove(file_path)
                    total_size_freed += file_size
                    print(f"  ❌ Removed file: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"  ⚠️  Error removing {file_path}: {e}")
            
            # Remove directories
            for dir_path in dirs_to_remove:
                try:
                    dir_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                                 for dirpath, dirnames, filenames in os.walk(dir_path)
                                 for filename in filenames)
                    shutil.rmtree(dir_path)
                    total_size_freed += dir_size
                    print(f"  🗂️  Removed directory: {os.path.basename(dir_path)}")
                except Exception as e:
                    print(f"  ⚠️  Error removing {dir_path}: {e}")
            
            if files_to_remove or dirs_to_remove:
                cleaned_count += 1
                print(f"  ✅ Cleaned folder: {len(files_to_remove)} files, {len(dirs_to_remove)} directories removed")
            else:
                print(f"  ✨ No cleanup needed - only model files found")
    
    # Convert size to human readable format
    def format_size(size_bytes):
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    print(f"\n🎉 Cleanup Summary:")
    print(f"   📁 Training folders processed: {cleaned_count}")
    print(f"   💾 Total space freed: {format_size(total_size_freed)}")
    print(f"   ✅ Model files preserved in all training folders")

def main():
    # Path to outputs directory
    outputs_path = "/data/home/umang/Trajectory_project/GPS-MTM/outputs"
    
    # Verify the path exists
    if not os.path.exists(outputs_path):
        print(f"❌ Error: Outputs path does not exist: {outputs_path}")
        return
    
    print(f"🚀 Starting cleanup of training folders in: {outputs_path}")
    print(f"🎯 Will preserve model files (.pth, .pt, .ckpt, .pkl, .bin, .safetensors)")
    print(f"🎯 Will preserve folders: checkpoints, models, weights")
    
    # Ask for confirmation
    response = input("\n⚠️  This will permanently delete non-model files. Continue? (y/N): ")
    if response.lower() != 'y':
        print("❌ Operation cancelled")
        return
    
    try:
        clean_training_folders(outputs_path)
        print(f"\n✅ Cleanup completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during cleanup: {e}")

if __name__ == "__main__":
    main()