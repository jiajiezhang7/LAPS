import os
import random
import argparse

def sample_and_prune(
    target_dir: str,
    num_to_keep: int,
    execute: bool,
):
    """
    Randomly samples a specified number of files from a directory and deletes the rest.

    Args:
        target_dir (str): The directory to process.
        num_to_keep (int): The number of files to keep.
        execute (bool): If False, runs in 'dry run' mode, only printing actions.
                        If True, performs the actual deletion.
    """
    print(f"Target Directory: {target_dir}")
    print(f"Number of files to keep: {num_to_keep}")
    print(f"EXECUTE MODE: {'ON (Deletions will occur)' if execute else 'OFF (Dry Run)'}")
    print("-" * 50)

    if not os.path.isdir(target_dir):
        print(f"Error: Directory not found at {target_dir}")
        return

    try:
        all_files = [
            f
            for f in os.listdir(target_dir)
            if os.path.isfile(os.path.join(target_dir, f))
        ]
    except OSError as e:
        print(f"Error reading directory {target_dir}: {e}")
        return

    num_total_files = len(all_files)
    print(f"Found {num_total_files} total files in the directory.")

    if num_total_files <= num_to_keep:
        print("Number of files is already less than or equal to the number to keep. No action needed.")
        return

    # Randomly sample files to keep
    files_to_keep = set(random.sample(all_files, num_to_keep))
    files_to_delete = [f for f in all_files if f not in files_to_keep]

    num_to_delete = len(files_to_delete)
    print(f"Selected {len(files_to_keep)} files to keep.")
    print(f"Will delete {num_to_delete} files.")

    if not execute:
        print("\n[DRY RUN] The following files would be deleted:")
        # Print first 10 for brevity
        for i, filename in enumerate(files_to_delete):
            if i < 10:
                print(f"  - {filename}")
        if num_to_delete > 10:
            print(f"  ... and {num_to_delete - 10} more.")
        print("\nTo execute deletion, run the script again with the --execute flag.")
    else:
        print("\nProceeding with deletion...")
        deleted_count = 0
        for i, filename in enumerate(files_to_delete):
            file_path = os.path.join(target_dir, filename)
            try:
                os.remove(file_path)
                deleted_count += 1
                # Print progress
                if (i + 1) % 100 == 0:
                    print(f"  Deleted {i + 1}/{num_to_delete} files...")
            except OSError as e:
                print(f"  Could not delete {file_path}: {e}")
        
        print(f"\nDeletion complete. Successfully deleted {deleted_count} out of {num_to_delete} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly sample a number of files to keep from a directory and delete the rest.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="/media/johnny/Data/data_motion_tokenizer/short_videos_inference/videos_ok",
        help="The target directory containing the files."
    )
    parser.add_argument(
        "-n",
        "--num-keep",
        type=int,
        default=2100,
        help="The number of files to randomly keep."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the actual file deletions. Without this flag, the script runs in 'dry run' mode."
    )

    args = parser.parse_args()

    sample_and_prune(
        target_dir=args.directory,
        num_to_keep=args.num_keep,
        execute=args.execute,
    )
