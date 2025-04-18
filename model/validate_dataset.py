#!/usr/bin/env python
"""
Dataset Validator for Deep Fake Detector

This script checks the integrity of the dataset and reports any issues.
Run this script before training to ensure your dataset is properly structured.
"""

import os
import sys
import cv2
import glob
import pandas as pd
import argparse
import time
from tqdm import tqdm
from mtcnn import MTCNN

def validate_dataset(dataset_dir, metadata_path, fix_issues=False, sample_limit=5):
    """
    Validate the dataset structure and report any issues found.
    
    Parameters:
    - dataset_dir: Path to the dataset directory
    - metadata_path: Path to the metadata CSV file
    - fix_issues: Whether to attempt fixing common issues
    - sample_limit: Number of samples to check for each test (for speed)
    
    Returns:
    - Boolean indicating whether the dataset is valid
    """
    print("\n" + "="*60)
    print("DATASET VALIDATION REPORT")
    print("="*60)
    
    all_checks_passed = True
    issues_found = []
    
    # Check 1: Does metadata file exist?
    print("\n📋 Checking metadata file...")
    if not os.path.exists(metadata_path):
        print(f"❌ ERROR: Metadata file not found at {metadata_path}")
        return False
    else:
        print(f"✅ Metadata file found at {metadata_path}")
    
    # Load metadata with progress bar
    try:
        print("📊 Loading metadata file...")
        # Get file size for progress bar
        file_size = os.path.getsize(metadata_path)
        chunk_size = 1024 * 1024  # 1MB chunks
        chunks = file_size // chunk_size + 1
        
        # Create progress bar for loading data
        with open(metadata_path, 'r') as f:
            pbar = tqdm(total=chunks, desc="Loading metadata", unit="MB")
            chunks_read = 0
            for _ in pd.read_csv(metadata_path, chunksize=100000):
                chunks_read += 1
                if chunks_read <= chunks:
                    pbar.update(1)
            pbar.close()
        
        metadata = pd.read_csv(metadata_path)
        print(f"✅ Metadata loaded successfully with {len(metadata)} entries")
    except Exception as e:
        print(f"❌ ERROR: Failed to load metadata: {str(e)}")
        return False
    
    # Check 2: Does dataset directory exist?
    print("\n📁 Checking dataset directory...")
    if not os.path.exists(dataset_dir):
        print(f"❌ ERROR: Dataset directory not found at {dataset_dir}")
        return False
    else:
        print(f"✅ Dataset directory found at {dataset_dir}")
    
    # Check 3: Label distribution
    print("\n🏷️ Checking label distribution...")
    if 'label' not in metadata.columns:
        print("❌ ERROR: Metadata missing 'label' column")
        all_checks_passed = False
        issues_found.append("Metadata missing 'label' column")
    else:
        real_count = len(metadata[metadata['label'] == 'REAL'])
        fake_count = len(metadata[metadata['label'] == 'FAKE'])
        other_count = len(metadata) - real_count - fake_count
        
        print(f"📊 Label distribution: {real_count} REAL videos, {fake_count} FAKE videos")
        
        if other_count > 0:
            print(f"⚠️ WARNING: Found {other_count} entries with labels other than REAL/FAKE")
            all_checks_passed = False
            issues_found.append(f"Found {other_count} entries with invalid labels")
        
        if real_count == 0 or fake_count == 0:
            print(f"❌ ERROR: Missing one or more classes (REAL: {real_count}, FAKE: {fake_count})")
            all_checks_passed = False
            issues_found.append("Missing one or more classes")
    
    # Check 4: Directory structure - do directories exist for each metadata entry?
    print("\n🔍 Checking directory structure...")
    # Get all directories in the dataset folder
    all_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"📁 Found {len(all_dirs)} directories in dataset folder")
    
    # Check metadata entries against directories
    print("🔄 Checking for matching directories for metadata entries...")
    
    # Get filenames from metadata and remove extensions to match directory names
    filenames_in_metadata = metadata['filename'].tolist()
    dir_names = [os.path.splitext(f)[0] for f in filenames_in_metadata]
    
    # Count missing directories
    missing_dirs = []
    for dirname in tqdm(dir_names[:sample_limit], desc="Checking directories"):
        dir_path = os.path.join(dataset_dir, dirname)
        if not os.path.exists(dir_path):
            missing_dirs.append(dirname)
    
    missing_count = len(missing_dirs)
    if missing_count > 0:
        print(f"⚠️ WARNING: {missing_count} directories are missing for metadata entries")
        print(f"   First few missing directories: {missing_dirs[:3]}")
        if fix_issues and missing_count > 0:
            print("🛠️ Creating missing directories...")
            for dirname in missing_dirs:
                os.makedirs(os.path.join(dataset_dir, dirname), exist_ok=True)
            print(f"✅ Created {len(missing_dirs)} missing directories")
        all_checks_passed = False
        issues_found.append(f"Missing {missing_count} directories")
    else:
        print(f"✅ All sampled metadata entries have matching directories")
    
    # Check 5: Do directories contain image files?
    print("\n📷 Checking for image files in directories...")
    dirs_without_images = []
    sample_dirs = all_dirs[:sample_limit]
    
    for dirname in tqdm(sample_dirs, desc="Checking for images"):
        dir_path = os.path.join(dataset_dir, dirname)
        jpg_files = glob.glob(os.path.join(dir_path, "*.jpg"))
        png_files = glob.glob(os.path.join(dir_path, "*.png"))
        
        if len(jpg_files) == 0 and len(png_files) == 0:
            dirs_without_images.append(dirname)
    
    if len(dirs_without_images) > 0:
        print(f"⚠️ WARNING: {len(dirs_without_images)} directories contain no image files")
        print(f"   Directories without images: {dirs_without_images[:3]}")
        all_checks_passed = False
        issues_found.append(f"{len(dirs_without_images)} directories without images")
    else:
        print(f"✅ All sampled directories contain image files")
    
    # Check 6: Can we load images and detect faces?
    print("\n👤 Testing face detection on sample images...")
    detector = MTCNN()
    face_detection_success = False
    
    for dirname in all_dirs[:5]:
        dir_path = os.path.join(dataset_dir, dirname)
        
        # Find image files
        jpg_files = glob.glob(os.path.join(dir_path, "*.jpg"))
        png_files = glob.glob(os.path.join(dir_path, "*.png"))
        image_files = jpg_files + png_files
        
        if not image_files:
            continue
        
        # Try to load and process an image
        test_img_path = image_files[0]
        try:
            img = cv2.imread(test_img_path)
            if img is None:
                print(f"⚠️ WARNING: Failed to load image {test_img_path}")
                continue
            
            print(f"✅ Successfully loaded image {os.path.basename(test_img_path)} with shape {img.shape}")
            
            # Test face detection
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(img_rgb)
            print(f"👤 Detected {len(faces)} faces in test image")
            
            if len(faces) > 0:
                face_detection_success = True
                break
        except Exception as e:
            print(f"❌ ERROR processing image {os.path.basename(test_img_path)}: {str(e)}")
    
    if not face_detection_success:
        print("⚠️ WARNING: No faces detected in sample images")
        all_checks_passed = False
        issues_found.append("No faces detected in sample images")
    else:
        print("✅ Face detection working properly")
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    if all_checks_passed:
        print("✅ All checks passed! Dataset structure is valid.")
        return True
    else:
        print(f"⚠️ Found {len(issues_found)} issues that need attention:")
        for i, issue in enumerate(issues_found):
            print(f"  {i+1}. {issue}")
        
        print("\nRecommended actions:")
        print("1. Make sure all video files have been properly processed to frames")
        print("2. Check that directory names match the basenames of files in metadata.csv")
        print("3. Verify that face detection is working properly with your images")
        print("4. Run this script with --fix flag to attempt automatic fixes")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Deep Fake Detector dataset")
    parser.add_argument("--dataset", default="dataset", help="Path to dataset directory")
    parser.add_argument("--metadata", default=os.path.join("dataset", "metadata.csv"), 
                        help="Path to metadata CSV file")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix common issues")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to check")
    
    args = parser.parse_args()
    
    is_valid = validate_dataset(
        dataset_dir=args.dataset,
        metadata_path=args.metadata,
        fix_issues=args.fix,
        sample_limit=args.samples
    )
    
    if is_valid:
        print("\n✨ Dataset is ready for training!")
        sys.exit(0)
    else:
        print("\n⚠️ Please fix the issues before training the model.")
        sys.exit(1)