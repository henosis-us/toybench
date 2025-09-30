import argparse
from pathlib import Path
import sys
import re
from datetime import datetime, date
from typing import Optional

DATE_PATTERN = re.compile(r"(\d{8})_(\d{6})")


def parse_datetime_from_name(name: str):
    """Extract datetime from a directory name like ..._YYYYMMDD_HHMMSS.
    Returns a datetime.date or None if not found/invalid.
    """
    m = DATE_PATTERN.search(name)
    if not m:
        return None
    yyyymmdd, hhmmss = m.groups()
    try:
        dt = datetime.strptime(f"{yyyymmdd}{hhmmss}", "%Y%m%d%H%M%S")
        return dt.date()
    except Exception:
        return None


def concatenate_summaries(main_directory: Path, output_file, start_date: Optional[date] = None, end_date: Optional[date] = None):
    """
    Finds 'summary_report.txt' in immediate subdirectories of main_directory
    and writes their contents sequentially to the provided output file.

    Optionally filters subdirectories whose names contain a timestamp
    suffix like *_YYYYMMDD_HHMMSS to within [start_date, end_date].
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

            # Date range filter (if requested)
            if start_date or end_date:
                d = parse_datetime_from_name(item.name)
                # If we cannot determine a date, skip it when filtering is enabled
                if d is None:
                    continue
                if start_date and d < start_date:
                    continue
                if end_date and d > end_date:
                    continue

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

def ask_input(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [default: {default}]" if default is not None else ""
    while True:
        value = input(f"{prompt}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    d = "y" if default else "n"
    while True:
        ans = input(f"{prompt} (y/n) [default: {d}]: ").strip().lower()
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")


def parse_date_str(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def interactive_main(results_directory: Optional[str], output: Optional[str], start: Optional[str], end: Optional[str]):
    # Choose a sensible default results directory
    cwd = Path.cwd()
    candidates = [
        cwd / "toybench" / "results",
        cwd / "results",
        cwd,
    ]
    default_results = next((str(p) for p in candidates if p.exists() and p.is_dir()), str(candidates[1]))

    if not results_directory:
        results_directory = ask_input("Enter path to results directory (contains subfolders with summary_report.txt)", default_results)

    if not output:
        output = ask_input("Enter path for concatenated output file", str(cwd / "concatenated_summary.txt"))

    # Date range filter
    start_date = None
    end_date = None
    if start or end:
        # CLI provided
        if start:
            start_date = parse_date_str(start)
        if end:
            end_date = parse_date_str(end)
    else:
        if ask_yes_no("Filter by a date range based on folder names like *_YYYYMMDD_HHMMSS?", False):
            while True:
                start_str = input("Start date (YYYY-MM-DD) or leave blank for no minimum: ").strip()
                end_str = input("End date (YYYY-MM-DD) or leave blank for no maximum: ").strip()
                try:
                    start_date = parse_date_str(start_str) if start_str else None
                    end_date = parse_date_str(end_str) if end_str else None
                    if start_date and end_date and end_date < start_date:
                        print("End date is before start date. Please try again.")
                        continue
                    break
                except ValueError:
                    print("Invalid date format. Please use YYYY-MM-DD.")

    main_dir_path = Path(results_directory)
    output_file_path = Path(output)

    print("\nSummary of choices:")
    print(f"- Results directory: {main_dir_path}")
    if start_date or end_date:
        print(f"- Date filter: from {start_date or '-inf'} to {end_date or '+inf'}")
    else:
        print("- Date filter: none")
    print(f"- Output file: {output_file_path}")

    # Open the output file and handle any errors
    try:
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            concatenate_summaries(main_dir_path, output_file, start_date=start_date, end_date=end_date)
    except IOError as e:
        print(f"Error opening or writing to output file: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Concatenates 'summary_report.txt' files found in immediate subdirectories of a given directory.\n"
            "By default this runs interactively to ask for the results directory, output file, and optional date range.\n"
            "You may also supply optional flags to skip prompts."
        )
    )
    parser.add_argument(
        "--results-directory",
        type=str,
        default=None,
        help="Path to the main directory containing the result subdirectories (e.g., 'results'). If omitted, you will be prompted."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output file for the concatenated summary. If omitted, you will be prompted."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date (YYYY-MM-DD) to filter subdirectories by encoded date in their names."
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date (YYYY-MM-DD) to filter subdirectories by encoded date in their names."
    )

    args = parser.parse_args()

    # Always run interactively unless all required info is supplied
    interactive_main(args.results_directory, args.output, args.start_date, args.end_date)

if __name__ == "__main__":
    main()
