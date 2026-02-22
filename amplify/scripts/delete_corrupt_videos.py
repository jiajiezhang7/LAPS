import argparse
import os
import re
from pathlib import Path

def parse_report(report_path):
    """Parses the report file and returns a list of unique video paths."""
    if not os.path.exists(report_path):
        print(f"\033[91mError: Report file not found at '{report_path}'\033[0m")
        return []

    with open(report_path, 'r') as f:
        lines = f.readlines()

    video_paths = []
    # Regex to find lines that are absolute paths to video files (e.g., .mp4, .mov, etc.)
    path_regex = re.compile(r'^/.*\.(mp4|mov|avi|mkv)$', re.IGNORECASE)

    for line in lines:
        line = line.strip()
        if path_regex.match(line):
            video_paths.append(line)

    # The report may list the same file multiple times; get unique paths.
    unique_paths = sorted(list(set(video_paths)))
    return unique_paths

def delete_videos(video_paths, dry_run=True):
    """Deletes the videos specified in the list of paths."""
    if not video_paths:
        print("\033[93mNo video files to delete.\033[0m")
        return

    print(f"Found {len(video_paths)} unique corrupt video files to process.")

    deleted_count = 0
    not_found_count = 0
    error_count = 0

    for video_path in video_paths:
        path = Path(video_path)
        if path.exists():
            if dry_run:
                print(f"[DRY RUN] Would delete: {video_path}")
                deleted_count += 1
            else:
                try:
                    path.unlink()
                    print(f"\033[92mDeleted: {video_path}\033[0m")
                    deleted_count += 1
                except OSError as e:
                    print(f"\033[91mError deleting {video_path}: {e}\033[0m")
                    error_count += 1
        else:
            if dry_run:
                print(f"[DRY RUN] File not found, would skip: {video_path}")
            else:
                print(f"\033[93mFile not found, skipping: {video_path}\033[0m")
            not_found_count += 1

    print("\n--- Summary ---")
    if dry_run:
        print("\033[96mOperation was a dry run. No files were actually deleted.\033[0m")
        print(f"Files that would be deleted: {deleted_count}")
        print(f"Files that would be skipped (not found): {not_found_count}")
    else:
        print(f"\033[92mSuccessfully deleted {deleted_count} files.\033[0m")
        if not_found_count > 0:
            print(f"\033[93mSkipped {not_found_count} files because they were not found.\033[0m")
        if error_count > 0:
            print(f"\033[91mFailed to delete {error_count} files due to errors.\033[0m")
    print("-----------------")

def main():
    parser = argparse.ArgumentParser(
        description="Deletes corrupt videos listed in a report file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--report-path',
        type=str,
        default='./corrupt_videos_report.txt',
        help="Path to the corrupt videos report file.\n(default: ./corrupt_videos_report.txt)"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Perform a dry run without deleting any files. This will only list the files that would be deleted."
    )

    args = parser.parse_args()

    if args.dry_run:
        print("\033[96m--- Starting in DRY RUN mode. No files will be deleted. ---\033[0m")

    video_paths = parse_report(args.report_path)
    delete_videos(video_paths, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
