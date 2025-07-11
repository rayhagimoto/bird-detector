"""
Preprocessing pipelines for bird watching data.

This module contains various preprocessing routines that can be called
independently or as part of a DVC pipeline.
"""

import argparse
import sys
from pathlib import Path
from typing import Union, Optional
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from .raw_image_utils import load_arw_image, arw_to_jpg
from .data_loaders import get_raw_data_config


def convert_to_jpg_and_resize(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_size: int = 256,
    jpg_quality: int = 95
) -> None:
    """
    Convert ARW image to JPG and resize to target_size px shortest side while preserving aspect ratio.
    
    Args:
        input_path: Path to ARW image file
        output_path: Path to save the processed JPG image
        target_size: Target size for shortest side (default: 256)
        jpg_quality: JPG compression quality (1-100, default: 95)
    """
    # Load the ARW image
    raw_img = load_arw_image(input_path)
    
    # Get image dimensions
    height, width = raw_img.shape[:2]
    
    # Calculate new dimensions preserving aspect ratio
    if height < width:
        # Portrait orientation
        new_height = target_size
        new_width = int(width * (target_size / height))
    else:
        # Landscape orientation
        new_width = target_size
        new_height = int(height * (target_size / width))
    
    # Resize image
    resized_img = cv2.resize(raw_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Convert to JPG
    jpg_bytes = arw_to_jpg(resized_img, quality=jpg_quality)
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(jpg_bytes)


def process_one_file(args):
    arw_file, output_file, target_size, jpg_quality, overwrite = args
    try:
        # Skip if file exists and overwrite is False
        if not overwrite and output_file.exists():
            return (str(arw_file), True, "skipped (already exists)")
        
        # Add memory cleanup before processing each file
        import gc
        gc.collect()
        
        convert_to_jpg_and_resize(
            input_path=arw_file,
            output_path=output_file,
            target_size=target_size,
            jpg_quality=jpg_quality
        )
        return (str(arw_file), True, None)
    except Exception as e:
        return (str(arw_file), False, str(e))


def process_set_to_jpg256(
    set_id: int,
    output_dir: Union[str, Path] = "data/interim/jpg256",
    jpg_quality: int = 95,
    target_size: int = 256,
    test_mode: bool = False,
    max_files: int = 10,
    overwrite: bool = False
) -> dict:
    """
    Process all ARW images in a data set to JPG format with shortest side = target_size px.
    Now parallelized using ProcessPoolExecutor.
    
    Args:
        set_id: Data set ID (e.g., 2 for set_2)
        output_dir: Directory to save processed images
        jpg_quality: JPG compression quality (1-100)
        target_size: Target size for shortest side (default: 256)
        test_mode: If True, only process first max_files files
        max_files: Maximum number of files to process in test mode
        overwrite: If False, skip files that already exist (default: False)
    """
    # Get data set configuration
    config = get_raw_data_config(set_id)
    if not config:
        raise ValueError(f"Configuration not found for set {set_id}")
    
    raw_data_path = Path(config['path'])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all ARW files (only uppercase, to avoid double-counting on case-insensitive filesystems)
    arw_files = list(raw_data_path.glob("*.ARW"))
    # arw_files = list(raw_data_path.glob("*.ARW")) + list(raw_data_path.glob("*.arw"))  # OLD: caused double-counting on Windows
    
    if not arw_files:
        print(f"No ARW files found in {raw_data_path}")
        return {"processed_files": 0, "errors": 0, "output_dir": str(output_dir)}
    
    # Limit files in test mode
    if test_mode:
        arw_files = arw_files[:max_files]
        print(f"TEST MODE: Processing only first {len(arw_files)} files from set {set_id}")
    else:
        print(f"Processing {len(arw_files)} ARW files from set {set_id} in parallel...")
    
    processed_count = 0
    error_count = 0
    results = []
    
    # Prepare arguments for each file
    file_args = [
        (arw_file, output_dir / (arw_file.stem + ".jpg"), target_size, jpg_quality, overwrite)
        for arw_file in arw_files
    ]
    
    # Calculate safe number of workers based on memory constraints
    # Each ARW file (~30MB) + processing overhead (~150MB) = ~180MB per worker
    # With ~5GB available RAM, we can safely use ~20 workers
    # But be conservative and use fewer to avoid memory issues
    available_ram_gb = 5  # Conservative estimate
    memory_per_worker_gb = 0.2  # 200MB per worker
    safe_workers = int(available_ram_gb / memory_per_worker_gb)
    max_workers = min(safe_workers, 8, multiprocessing.cpu_count())  # Cap at 8 for stability
    print(f"Using {max_workers} workers (memory-safe configuration)")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_one_file, args) for args in file_args]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Converting ARW to JPG"):
            arw_file, success, error = f.result()
            if success:
                processed_count += 1
            else:
                error_count += 1
                print(f"Error processing {arw_file}: {error}")
            results.append((arw_file, success, error))
    
    print(f"Processing complete. Output saved to {output_dir}")
    print(f"Processed: {processed_count}, Errors: {error_count}")
    
    return {
        "processed_files": processed_count,
        "errors": error_count,
        "output_dir": str(output_dir),
        "set_id": set_id,
        "jpg_quality": jpg_quality,
        "target_size": target_size,
        "input_path": str(raw_data_path),
        "results": results
    }


def main():
    """Command-line interface for preprocessing pipelines."""
    parser = argparse.ArgumentParser(description="Bird watching data preprocessing")
    parser.add_argument(
        "pipeline",
        choices=["convert_to_jpg_and_resize"],
        help="Preprocessing pipeline to run"
    )
    parser.add_argument(
        "--set-id",
        type=int,
        required=True,
        help="Data set ID to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/interim/jpg256",
        help="Output directory for processed images"
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPG compression quality (1-100)"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=256,
        help="Target size for shortest side (default: 256)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (default: False, skip existing files)"
    )
    
    args = parser.parse_args()
    
    if args.pipeline == "convert_to_jpg_and_resize":
        process_set_to_jpg256(
            set_id=args.set_id,
            output_dir=args.output_dir,
            jpg_quality=args.jpg_quality,
            target_size=args.target_size,
            overwrite=args.overwrite
        )
    else:
        print(f"Unknown pipeline: {args.pipeline}")
        sys.exit(1)


if __name__ == "__main__":
    main()
