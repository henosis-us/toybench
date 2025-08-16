# <file_system_env.py>
import logging
import posixpath # Use posixpath for consistent virtual path manipulation
import re
from .base_env import BaseEnvironment

logger = logging.getLogger(__name__)

class FileSystemEnv(BaseEnvironment):
    """
    Simulated file system environment - EXTREME COMPLEXITY VERSION.
    Agent interacts via commands (ls, cd, pwd, mkdir, cat, cp, rm, echo >, echo >>).
    Goal involves content parsing, conditional logic, file writing,
    copying/deleting, nested dirs, and precise state management.
    Evaluation is deterministic.
    """

    def __init__(self, goal_description: str):
        self._goal = goal_description
        self.fs = {} # {absolute_path: content_string_or_directory_dict}
        self.cwd = '/' # Current working directory

        # --- NEW EXTREMELY COMPLEX INITIAL STATE ---
        self.initial_state = {
            # Root and base structure
            '/': {},
            '/project': {},
            '/staging': {},
            # Config file to parse
            '/project/config.txt': "version=1.2\ntarget_dir=data_v1.2",
            # Source files to copy/delete
            '/project/src': {},
            '/project/src/main.py': "print('Main application logic')",
            '/project/src/utils.py': "# Utility functions\ndef helper(): pass",
            # Temporary files for conditional logic / deletion
            '/project/tmp': {},
            '/project/tmp/error.log': "Error: Disk space low.", # File exists for conditional check
            # Asset file to leave untouched (distractor)
            '/project/assets': {},
            '/project/assets/logo.png': "PNG_IMAGE_DATA_BYTES",
            # Staging file to copy/delete
            '/staging/ready.txt': "Signal file: Ready to process.",
        }
        # --- END NEW INITIAL STATE ---

        # Reset to apply the initial state
        self.reset()

    def reset(self) -> str:
        """Resets the environment and returns initial CWD description."""
        self.fs = {} # Clear existing state
        # Deep copy initial state structure
        for path, content in self.initial_state.items():
             if isinstance(content, dict):
                 self.fs[path] = {} # Ensure fresh dict for directories
             else:
                 self.fs[path] = content # Copy file content
        self.cwd = '/project/src' # Start in a nested directory
        logger.info(f"EXTREME Complex File System Environment Reset. Agent starts in: {self.cwd}")

        # Agent needs to figure out state via commands
        return f"You are currently in the '{self.cwd}' directory. Use commands like 'ls', 'pwd', 'cd' to navigate and explore."

    # --- Path and State Helpers (mostly unchanged) ---
    def _render_ls_output(self, path: str) -> str:
        normalized_path = self._normalize_path(path)

        # First, check if the path exists at all.
        if normalized_path not in self.fs:
            return f"Error: ls: cannot access '{path}': No such file or directory"

        # Next, check if it's a file (not a directory) to mimic real 'ls' behavior.
        if not isinstance(self.fs[normalized_path], dict):
            # Real-world `ls` on a file prints the path back.
            return path

        # If we get here, it's a directory. Proceed with listing contents.
        items = []
        prefix = normalized_path.rstrip('/') + '/'
        # Handle root edge case for prefix
        if normalized_path == '/': prefix = '/'
        for p, content in self.fs.items():
             # Check if path p is *directly* inside the directory 'prefix'
             if p.startswith(prefix) and p != normalized_path:
                 relative_path = p[len(prefix):]
                 if '/' not in relative_path: # Only list immediate children
                     item_name = relative_path
                     if isinstance(content, dict): items.append(f"{item_name}/")
                     else: items.append(item_name)
        return "\n".join(sorted(items)) if items else "(empty)"

    def _get_state_string(self) -> str:
        """Generates a state description including CWD and its contents."""
        ls_output = self._render_ls_output(self.cwd)
        status = f"CWD: {self.cwd}\nContents:\n{ls_output}"
        # Handle case where CWD listing might fail (e.g., if deleted externally)
        if ls_output.startswith("Error:"): status = f"CWD: {self.cwd}\n{ls_output}"
        return status

    def _normalize_path(self, path: str) -> str:
        """Normalizes a path relative to cwd into an absolute POSIX path."""
        if not path: return self.cwd
        # Handle echo command special case where path might be part of content
        if '>' in path and ('echo ' not in path.lower() or path.strip().lower().startswith('echo')):
             # Avoid normalizing the content part of an echo command if passed accidentally
             # This is a heuristic, assumes 'path' is only the target file for non-echo cases
             pass
        # Join with CWD if path is relative, otherwise use the absolute path
        abs_path = posixpath.join(self.cwd, path) if not posixpath.isabs(path) else path
        # Normalize the path (handles '..', '.', '//')
        normalized = posixpath.normpath(abs_path)
        # Ensure root path is represented correctly
        if normalized == '.' and self.cwd == '/': return '/'
        # Ensure result is absolute (should be after join/normpath unless input was weird)
        if not normalized.startswith('/'):
             # This might happen if path is like '.' relative to non-root cwd
             # Re-join based on cwd if normpath result is relative?
             if normalized == '.': return self.cwd
             logger.warning(f"Path normalization resulted in non-absolute path '{normalized}' from input '{path}' in cwd '{self.cwd}'. Defaulting to CWD.")
             return self.cwd # Fallback or error? Fallback for now.
        return normalized if normalized else '/'

    # --- Interface Methods Implementation (getters unchanged) ---
    def get_agent_player_mark(self) -> str | None: return None
    def get_state(self) -> str: return self._get_state_string()
    def get_goal(self) -> str: return self._goal
    def get_prompt_context(self) -> dict:
        """Returns the context dictionary needed for prompt formatting."""
        # Removed 'mv', added 'cat', 'cp', 'echo'
        available_commands = "ls, cd, pwd, mkdir, cat, cp, rm, echo > (overwrite), echo >> (append)"
        return {
            "goal": self.get_goal(),
            "current_state": self._get_state_string(), # Current state string
            "available_commands": available_commands
        }

    def validate_action(self, action: str) -> bool:
        """Basic syntax validation for known commands."""
        action = action.strip()
        if not action: logger.debug("Validation failed: Empty action."); return False
        parts = action.split(maxsplit=1)
        command = parts[0].lower()
        args_part = parts[1] if len(parts) > 1 else ""
        # Note: 'echo' validation is tricky due to content. We'll do basic checks here.
        known_commands = ["ls", "cd", "pwd", "mkdir", "cat", "cp", "rm", "echo"]
        if command not in known_commands:
            logger.debug(f"Validation failed: Unknown command '{command}'.")
            return False
        # Commands requiring >= 1 argument (path/target)
        if command in ["cd", "mkdir", "cat", "rm"] and not args_part:
             logger.debug(f"Validation failed: Cmd '{command}' needs argument(s)."); return False
        # Command requiring >= 2 arguments (source + destination)
        if command == "cp":
             if not args_part or len(args_part.split()) < 2:
                 logger.debug(f"Validation failed: Cmd 'cp' needs source and destination args."); return False
        # Basic 'echo' validation: needs content and redirection operator + target file
        if command == "echo":
             # Look for redirection operator (must be surrounded by spaces for simple split, or handle quotes)
             # Simplified check: look for > or >>. Robust parsing is hard.
             if '>' not in args_part:
                 logger.debug(f"Validation failed: Cmd 'echo' missing redirection operator '>' or '>>'."); return False
             # Check if there's *something* after the redirection
             redirect_match = re.search(r"(>>?)\s*(.*)", args_part)
             if not redirect_match or not redirect_match.group(2):
                 logger.debug(f"Validation failed: Cmd 'echo' missing target file after redirection."); return False
        # 'ls' and 'pwd' can optionally take args or no args.
        return True

    def step(self, action: str) -> str:
        """Executes the action command and returns a result string (output or error)."""
        action = action.strip()
        logger.debug(f"Executing FS action: CWD='{self.cwd}', Action='{action}'")
        try:
            # === Command Parsing === (More complex due to echo)
            parts = action.split(maxsplit=1)
            command = parts[0].lower() if parts else ""
            args_str = parts[1].strip() if len(parts) > 1 else ""

            # === Handle Commands ===
            if command == "pwd": return f"{self.cwd}"

            elif command == "ls":
                target_path_str = args_str if args_str else self.cwd
                target_path_norm = self._normalize_path(target_path_str)
                ls_output = self._render_ls_output(target_path_norm)
                return ls_output

            elif command == "cd":
                if not args_str: return "Error: cd: missing operand"
                target_path = self._normalize_path(args_str)
                if target_path not in self.fs: return f"Error: cd: no such file or directory: {args_str}"
                if not isinstance(self.fs[target_path], dict): return f"Error: cd: not a directory: {args_str}"
                self.cwd = target_path
                logger.info(f"Changed CWD to: {self.cwd}")
                return f"Current directory is now: {self.cwd}" # More informative

            elif command == "mkdir":
                if not args_str: return "Error: mkdir: missing operand"
                target_path = self._normalize_path(args_str)
                if target_path == '/': return f"Error: mkdir: cannot create directory '{args_str}': Root exists"
                parent_path = posixpath.dirname(target_path)
                if parent_path != '/' and (parent_path not in self.fs or not isinstance(self.fs[parent_path], dict)):
                    return f"Error: mkdir: cannot create directory '{args_str}': Parent directory '{parent_path}' does not exist"
                if target_path in self.fs: return f"Error: mkdir: cannot create directory '{args_str}': File or directory exists"
                self.fs[target_path] = {}
                logger.info(f"Created directory: {target_path}")
                return "Success."

            elif command == "cat":
                if not args_str: return "Error: cat: missing file operand"
                target_path = self._normalize_path(args_str)
                if target_path not in self.fs: return f"Error: cat: {args_str}: No such file or directory"
                if isinstance(self.fs[target_path], dict): return f"Error: cat: {args_str}: Is a directory"
                # Return file content
                content = self.fs.get(target_path, "") # Should exist based on check above
                logger.info(f"Read content from: {target_path}")
                return str(content) # Ensure it's a string

            elif command == "cp":
                cp_parts = args_str.split()
                if len(cp_parts) < 2: return "Error: cp: missing destination file operand after source"
                source_str = cp_parts[0]
                dest_str = cp_parts[-1] # Assume last is destination
                source_path = self._normalize_path(source_str)
                dest_path_raw = self._normalize_path(dest_str)

                # Validations
                if source_path == '/': return f"Error: cp: cannot copy '/' directory"
                if source_path not in self.fs: return f"Error: cp: cannot stat '{source_str}': No such file or directory"
                if isinstance(self.fs[source_path], dict): return f"Error: cp: omitting directory '{source_str}' (use -r recursive copy - not implemented)"

                # Determine final destination path
                final_dest_path = dest_path_raw
                parent_dest_path = posixpath.dirname(final_dest_path)

                if dest_path_raw in self.fs and isinstance(self.fs[dest_path_raw], dict):
                    # Destination is an existing directory, copy source *into* it
                    source_basename = posixpath.basename(source_path)
                    final_dest_path = posixpath.join(dest_path_raw, source_basename)
                    parent_dest_path = dest_path_raw
                elif dest_str.endswith('/'):
                    if dest_path_raw not in self.fs: return f"Error: cp: target directory '{dest_str}' does not exist"
                    if not isinstance(self.fs[dest_path_raw], dict): return f"Error: cp: target '{dest_str}' is not a directory"
                     # Should have been caught by previous check, but safety first

                # Check parent exists
                if parent_dest_path != '/' and (parent_dest_path not in self.fs or not isinstance(self.fs[parent_dest_path], dict)):
                     return f"Error: cp: cannot create regular file '{final_dest_path}': Parent directory '{parent_dest_path}' does not exist"

                # Check conflict (copying onto directory)
                if final_dest_path in self.fs and isinstance(self.fs[final_dest_path], dict):
                    return f"Error: cp: cannot overwrite directory '{final_dest_path}' with non-directory '{source_str}'"

                # Execute Copy (overwrite file if exists)
                if final_dest_path in self.fs: logger.warning(f"cp: Overwriting existing destination file '{final_dest_path}'")
                self.fs[final_dest_path] = self.fs[source_path] # Copy content (string)
                logger.info(f"Copied '{source_path}' to '{final_dest_path}'")
                return "Success."

            elif command == "rm":
                if not args_str: return "Error: rm: missing operand"
                target_path = self._normalize_path(args_str)
                if target_path == '/': return "Error: rm: cannot remove '/'"
                if target_path not in self.fs: return f"Error: rm: cannot remove '{args_str}': No such file or directory"
                if isinstance(self.fs[target_path], dict): return f"Error: rm: cannot remove '{args_str}': Is a directory"
                del self.fs[target_path]
                logger.info(f"Removed file: {target_path}")
                return "Success."

            elif command == "echo":
                # Complex parsing for echo "content" >/>> file
                # Regex to capture content (handling potential quotes) and redirection target
                # Pattern: optional quotes, content, optional quotes, operator, target
                match = re.match(r'''^\s*(?:"([^"]*)"|'([^']*)'|([^>]+?))\s*(>>?)\s*(.*)''', args_str, re.S)
                if not match:
                     # Try alternative pattern if first char is > (e.g., echo > file)
                     match_alt = re.match(r'''^\s*(>>?)\s*(.*)''', args_str)
                     if match_alt:
                         operator, target_str = match_alt.groups()
                         content_to_write = "" # Empty content
                     else:
                         return f"Error: echo: Invalid syntax. Use echo 'content' > file or echo 'content' >> file."
                else:
                    content_group1, content_group2, content_group3, operator, target_str = match.groups()
                    content_to_write = content_group1 or content_group2 or content_group3 # Get captured content
                    content_to_write = content_to_write.strip() if content_to_write else ""

                target_path_str = target_str.strip()
                if not target_path_str: return "Error: echo: Missing target file operand."
                target_path = self._normalize_path(target_path_str)
                parent_path = posixpath.dirname(target_path)

                # Check parent exists
                if parent_path != '/' and (parent_path not in self.fs or not isinstance(self.fs[parent_path], dict)):
                    return f"Error: echo: cannot create file '{target_path_str}': Parent directory '{parent_path}' does not exist"

                # Cannot echo to a directory
                if target_path in self.fs and isinstance(self.fs[target_path], dict):
                    return f"Error: echo: cannot write to '{target_path_str}': Is a directory"

                # Execute write/append
                if operator == '>': # Overwrite or create
                    self.fs[target_path] = content_to_write + "\n" # Add newline like shell echo
                    logger.info(f"Wrote to file (overwrite): {target_path}")
                elif operator == '>>': # Append or create
                    existing_content = self.fs.get(target_path, "")
                    # Ensure existing content is string (should be if not dir)
                    if not isinstance(existing_content, str):
                        return f"Error: echo: cannot append to '{target_path_str}': Not a regular file?"
                    # Append with newline separation only if existing content is not empty
                    separator = "" if not existing_content or existing_content.endswith('\n') else "\n"
                    self.fs[target_path] = existing_content + separator + content_to_write + "\n" # Add newline like shell echo
                    logger.info(f"Appended to file: {target_path}")
                return "Success."
            else: return f"Error: Unknown command: {command}"
        except Exception as e:
            logger.error(f"Unexpected error during FS step for action '{action}': {e}", exc_info=True)
            return f"Error: Internal environment error processing command '{command}'."

    # --- check_goal_achieved, assess_intermediate_status unchanged ---
    def check_goal_achieved(self) -> bool:
        """Internal check used by deterministic evaluation."""
        # Consider also checking CWD here if required by goal?
        eval_score = self.evaluate_final_state()
        # Check final CWD if necessary
        # final_cwd_ok = self.cwd == '/final' # Example check
        # return eval_score == 3 and final_cwd_ok
        return eval_score == 3

    def assess_intermediate_status(self) -> any:
        """Intermediate progress assessment (disabled for FS)."""
        return None

    # --- get_final_eval_input unchanged ---
    def get_final_eval_input(self) -> str:
        """Returns a comprehensive final state description for logging/evaluation."""
        final_state_lines = ["Final File System State:", "------------------------", f"Final CWD: {self.cwd}"]
        paths_to_log = sorted(self.fs.keys())
        if not paths_to_log:
             final_state_lines.append("(File system is empty)")
        else:
            for path in paths_to_log:
                content = self.fs[path]
                if isinstance(content, dict):
                    final_state_lines.append(f"DIR : {path}/")
                else:
                    content_preview = ""
                    if isinstance(content, str):
                         content_preview = (content[:70] + '...' if len(content) > 70 else content).replace('\n', '\\n')
                    final_state_lines.append(f"FILE: {path} (content: '{content_preview}')")
        return "\n".join(final_state_lines)

    # --- NEW ULTRA COMPLEX Deterministic Evaluation Method ---
    def evaluate_final_state(self) -> int:
        """
        Checks the final file system state against the ultra-complex goal.
        Returns score: 3 (Success), 2 (Partial), 1 (Fail).
        """
        logger.info(f"Performing ULTRA complex deterministic evaluation against goal: '{self._goal}'")
        score = 1 # Default to Fail
        issues = [] # Collect reasons for failure/partial success

        # --- Define Target State & Requirements derived from Goal ---
        # Required Dirs (dependent on parsed config)
        # Required Files & Content (dependent on parsed config)
        # Conditional File/Absence (based on initial error.log)
        # File to be Deleted
        # Distractor file state
        # Original files state (should be removed)
        # Final CWD state

        # 1. Parse expected values from initial config (simulate what agent should have done)
        initial_config_content = self.initial_state.get('/project/config.txt', '')
        version_match = re.search(r"version=(\S+)", initial_config_content)
        target_dir_match = re.search(r"target_dir=(\S+)", initial_config_content)
        expected_version = version_match.group(1) if version_match else "UNKNOWN_VERSION"
        expected_target_dir_name = target_dir_match.group(1) if target_dir_match else "UNKNOWN_TARGET_DIR"

        # Construct expected paths based on parsed values
        expected_archive_base = f'/archive/{expected_version}'
        expected_final_data_dir = f'/final/{expected_target_dir_name}'
        expected_dirs = ['/archive', expected_archive_base, '/final', '/final/logs', expected_final_data_dir]

        # Expected files after copy/rename
        expected_copied_files = {
            f'{expected_archive_base}/main.py': self.initial_state.get('/project/src/main.py'),
            f'{expected_archive_base}/utils.py': self.initial_state.get('/project/src/utils.py'),
            f'{expected_final_data_dir}/ready.txt': self.initial_state.get('/staging/ready.txt')
        }

        # Conditional file logic (based on initial state)
        initial_error_log_existed = '/project/tmp/error.log' in self.initial_state
        expected_conditional_files = {}
        if initial_error_log_existed:
            expected_conditional_files['/final/logs/error.log'] = self.initial_state.get('/project/tmp/error.log')
        else:
            # If error log did NOT exist initially, status_ok.txt should be created (empty)
            expected_conditional_files['/final/logs/status_ok.txt'] = "\n" # echo "" > file adds newline

        # File to be written/appended
        expected_summary_path = '/final/summary.txt'
        expected_summary_line1 = f"Archived version {expected_version} to /archive/{expected_version}/\n"
        expected_summary_line2 = f"Processed data to /final/{expected_target_dir_name}/\n"
        expected_summary_content = expected_summary_line1 + expected_summary_line2

        # File that must be deleted
        file_to_delete_tmp = '/project/tmp/error.log' if initial_error_log_existed else None # Only check deletion if it existed initially
        file_to_delete_staging = '/staging/ready.txt'
        files_to_delete_src = ['/project/src/main.py', '/project/src/utils.py']

        # Distractor file that must remain
        distractor_file = '/project/assets/logo.png'
        initial_distractor_content = self.initial_state.get(distractor_file)

        # Expected final CWD
        expected_final_cwd = '/final'

        # --- Start Checking ---
        all_ok = True
        # Check Dirs
        for d in expected_dirs:
            if not (d in self.fs and isinstance(self.fs[d], dict)):
                issues.append(f"Missing or not directory: {d}")
                all_ok = False

        # Check Copied Files (existence and content match)
        for f_path, expected_content in expected_copied_files.items():
             if not (f_path in self.fs and not isinstance(self.fs[f_path], dict)):
                 issues.append(f"Missing copied file: {f_path}")
                 all_ok = False
             elif self.fs[f_path] != expected_content:
                 issues.append(f"Incorrect content for copied file: {f_path}")
                 all_ok = False

        # Check Conditional Files
        for f_path, expected_content in expected_conditional_files.items():
             if not (f_path in self.fs and not isinstance(self.fs[f_path], dict)):
                 # Check if the *other* conditional file exists incorrectly
                 other_file = '/final/logs/status_ok.txt' if initial_error_log_existed else '/final/logs/error.log'
                 if other_file in self.fs:
                      issues.append(f"Incorrect conditional file: {other_file} exists instead of {f_path}")
                 else:
                      issues.append(f"Missing conditional file: {f_path}")
                 all_ok = False
             # Allow empty file match for status_ok.txt created via echo "" >
             elif f_path.endswith('status_ok.txt') and self.fs[f_path] == "\n" and expected_content == "\n":
                  pass # Correct empty file with newline
             elif self.fs[f_path] != expected_content:
                 issues.append(f"Incorrect content for conditional file: {f_path}")
                 all_ok = False

        # Check Written/Appended File Content
        if expected_summary_path not in self.fs or isinstance(self.fs[expected_summary_path], dict):
            issues.append(f"Missing summary file: {expected_summary_path}")
            all_ok = False
        elif self.fs[expected_summary_path] != expected_summary_content:
             # **FIX Syntax Error Here**
             actual_content = self.fs[expected_summary_path]
             actual_content_escaped = actual_content.replace('\n', '\\n')
             expected_content_escaped = expected_summary_content.replace('\n', '\\n')
             issues.append(f"Incorrect content in summary file: {expected_summary_path}. Got: '{actual_content_escaped}' Expected: '{expected_content_escaped}'")
             all_ok = False

        # Check Deletions (Originals should be gone)
        if file_to_delete_tmp and file_to_delete_tmp in self.fs:
             issues.append(f"File not deleted: {file_to_delete_tmp}")
             all_ok = False
        if file_to_delete_staging in self.fs:
             issues.append(f"File not deleted: {file_to_delete_staging}")
             all_ok = False
        for f_path in files_to_delete_src:
             if f_path in self.fs:
                 issues.append(f"Original source file not deleted: {f_path}")
                 all_ok = False

        # Check Distractor
        if distractor_file not in self.fs or isinstance(self.fs[distractor_file], dict) or self.fs[distractor_file] != initial_distractor_content:
            issues.append(f"Distractor file modified or missing: {distractor_file}")
            all_ok = False

        # Check Final CWD
        if self.cwd != expected_final_cwd:
            issues.append(f"Incorrect final CWD: expected '{expected_final_cwd}', got '{self.cwd}'")
            all_ok = False

        # --- Determine Score ---
        if all_ok:
            logger.info("Evaluation: Success (All complex conditions met).")
            score = 3
        else:
            # Basic Partial vs Fail: Did core directories get created?
            core_dirs_ok = all(d in self.fs and isinstance(self.fs[d], dict) for d in ['/archive', '/final', '/final/logs'])
            if core_dirs_ok and (expected_archive_base in self.fs) and (expected_final_data_dir in self.fs) :
                 logger.info(f"Evaluation: Partial (Core directories OK, but other issues exist: {'; '.join(issues)})")
                 score = 2
            else:
                 logger.info(f"Evaluation: Fail (Fundamental structure incorrect or critical errors: {'; '.join(issues)})")
                 score = 1
        return score
# --- END NEW ULTRA COMPLEX Deterministic Evaluation Method ---