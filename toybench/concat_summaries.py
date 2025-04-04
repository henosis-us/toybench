import argparse
from pathlib import Path
import sys

def concatenate_summaries(main_directory: Path):
    """
    Finds 'summary_report.txt' in immediate subdirectories of main_directory
    and prints their contents sequentially.
    """
    if not main_directory.is_dir():
        print(f"Error: Provided path is not a valid directory: {main_directory}", file=sys.stderr)
        sys.exit(1)

    print(f"--- Searching for summary reports in subdirectories of: {main_directory} ---")

    subdirs_found = False
    report_files_found = 0

    # Iterate through items in the main directory, sorted alphabetically for consistency
    for item in sorted(main_directory.iterdir()):
        if item.is_dir():
            subdirs_found = True
            summary_file_path = item / "summary_report.txt" # Construct the path

            if summary_file_path.is_file():
                report_files_found += 1
                print(f"\n{'='*20} Contents from: {summary_file_path} {'='*20}\n")
                try:
                    with open(summary_file_path, 'r', encoding='utf-8') as f:
                        print(f.read().strip()) # Read and print content, stripping whitespace
                except Exception as e:
                    print(f"Error reading file {summary_file_path}: {e}", file=sys.stderr)
                # Optional: Add a separator after each file's content if needed
                # print("\n" + "-"*60 + "\n")
            else:
                # Optionally notify if a subdir doesn't have the report
                # print(f"Info: 'summary_report.txt' not found in {item}", file=sys.stderr)
                pass # Silently skip folders without the report file

    print(f"\n--- End of Summary Concatenation ({report_files_found} reports found) ---")

    if not subdirs_found:
        print(f"Warning: No subdirectories found in {main_directory}", file=sys.stderr)
    elif report_files_found == 0:
         print(f"Warning: No 'summary_report.txt' files found in any subdirectories of {main_directory}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Concatenates 'summary_report.txt' files found in immediate subdirectories of a given directory."
    )
    parser.add_argument(
        "results_directory",
        type=str,
        help="Path to the main directory containing the result subdirectories (e.g., 'results')."
    )

    args = parser.parse_args()
    main_dir_path = Path(args.results_directory)

    concatenate_summaries(main_dir_path)

if __name__ == "__main__":
    main()