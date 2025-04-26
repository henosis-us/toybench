import argparse
from pathlib import Path
import sys

def concatenate_summaries(main_directory: Path, output_file):
    """
    Finds 'summary_report.txt' in immediate subdirectories of main_directory
    and writes their contents sequentially to the provided output file.
    """
    if not main_directory.is_dir():
        print(f"Error: Provided path is not a valid directory: {main_directory}", file=sys.stderr)
        sys.exit(1)

    print(f"--- Searching for summary reports in subdirectories of: {main_directory} ---", file=output_file)

    subdirs_found = False
    report_files_found = 0

    # Iterate through items in the main directory, sorted alphabetically for consistency
    for item in sorted(main_directory.iterdir()):
        if item.is_dir():
            subdirs_found = True
            summary_file_path = item / "summary_report.txt"  # Construct the path

            if summary_file_path.is_file():
                report_files_found += 1
                print(f"\n{'='*20} Contents from: {summary_file_path} {'='*20}\n", file=output_file)
                try:
                    with open(summary_file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()  # Read content, stripping whitespace
                        print(content, file=output_file)  # Write content to output file
                except Exception as e:
                    print(f"Error reading file {summary_file_path}: {e}", file=sys.stderr)
                # Optional: Add a separator after each file's content if needed
                # print("\n" + "-"*60 + "\n", file=output_file)
            else:
                # Optionally notify if a subdir doesn't have the report
                # print(f"Info: 'summary_report.txt' not found in {item}", file=sys.stderr)
                pass  # Silently skip folders without the report file

    print(f"\n--- End of Summary Concatenation ({report_files_found} reports found) ---", file=output_file)

    if not subdirs_found:
        print(f"Warning: No subdirectories found in {main_directory}", file=sys.stderr)
    elif report_files_found == 0:
        print(f"Warning: No 'summary_report.txt' files found in any subdirectories of {main_directory}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="Concatenates 'summary_report.txt' files found in immediate subdirectories of a given directory and writes the output to a specified file."
    )
    parser.add_argument(
        "results_directory",
        type=str,
        help="Path to the main directory containing the result subdirectories (e.g., 'results')."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output file where the concatenated summary will be written (e.g., 'concatenated_summary.txt')."
    )

    args = parser.parse_args()
    main_dir_path = Path(args.results_directory)
    output_file_path = Path(args.output)

    # Open the output file and handle any errors
    try:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            concatenate_summaries(main_dir_path, output_file)
    except IOError as e:
        print(f"Error opening or writing to output file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()