# utils/browser_utils.py
"""
Utilities for browser automation (Selenium) and related file operations,
primarily used by the SolarSystemEnv and Evaluator for multimodal tasks.
"""

import base64
import mimetypes
import os
import time
import logging
from datetime import datetime

# Import Selenium components
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.remote.webdriver import WebDriver
    from selenium.common.exceptions import WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    # Define dummy classes/types if selenium is not installed
    class WebDriver: pass
    class ChromeOptions: pass
    class WebDriverException(Exception): pass
    webdriver = None

logger = logging.getLogger(__name__)

if not SELENIUM_AVAILABLE:
    logger.warning("Selenium library not found. Browser automation features (e.g., for 'solar_gen' task) will not be available.")


# --- Browser Setup ---
def setup_browser(browser_type="chrome") -> WebDriver | None:
    """Sets up a headless Selenium WebDriver with logging enabled."""
    if not SELENIUM_AVAILABLE:
        logger.error("Cannot setup browser: Selenium library is not installed.")
        return None

    options = None
    driver = None
    browser_type = browser_type.lower()

    try:
        if browser_type == "chrome":
            options = ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1800,1000")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--enable-logging")
            options.add_argument("--v=1")
            # Use goog:loggingPrefs for Chrome
            options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})
            logger.info("Setting up headless Chrome driver...")
            # Assumes chromedriver is in PATH. Consider using Service object.
            driver = webdriver.Chrome(options=options)
        # Add other browser types here if needed
        # elif browser_type == "firefox":
        #    ...
        else:
            logger.error(f"Unsupported browser type: {browser_type}")
            return None # Return None for unsupported type

        logger.info(f"WebDriver for {browser_type} initialized successfully.")
        return driver

    except WebDriverException as e:
        logger.error(f"Failed WebDriver init for {browser_type}: {e.msg}\nEnsure WebDriver is installed, matches browser version, and is in PATH.")
        # No driver object likely exists here, so no cleanup needed within this exception
        return None # Return None on WebDriver specific errors
    except Exception as e:
        logger.error(f"Unexpected error initializing WebDriver for {browser_type}: {e}", exc_info=True)
        # Clean up partial driver if initialization failed mid-way during unexpected errors
        if driver:
            try:
                driver.quit()
            except Exception:
                pass # Ignore errors during cleanup quit
        return None # Return None on general errors

# --- Browser Interaction ---
def capture_browser_logs(driver: WebDriver) -> str:
    """Captures and formats browser console logs."""
    # Check if Selenium is available and if driver is a valid WebDriver instance
    if not SELENIUM_AVAILABLE or not isinstance(driver, WebDriver):
        logger.error("WebDriver unavailable or invalid for log capture.")
        return "Error: WebDriver unavailable for log capture."

    logs = []
    try:
        # Attempt to retrieve logs
        browser_logs = driver.get_log('browser')
        if browser_logs:
            for entry in browser_logs:
                try:
                    # Safely get values with defaults
                    timestamp_ms = entry.get('timestamp', 0)
                    level = entry.get('level', 'UNKNOWN')
                    message = entry.get('message', '')
                    source = entry.get('source', '')

                    # Format timestamp
                    timestamp_dt = datetime.fromtimestamp(timestamp_ms / 1000)
                    timestamp_str = timestamp_dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

                    # Clean up console messages safely
                    try:
                        if 'console-api' in source:
                            if message.startswith('"') and message.endswith('"'): message = message[1:-1]
                            message = message.replace('\\\\n', '\\n').replace('\\n', '\n').replace('\\"', '"')
                            # Check prefixes added by interceptor script
                            prefixes_to_remove = ["console-api ", "SELENIUM_LOG_MSG:", "SELENIUM_ERROR_MSG:"]
                            for prefix in prefixes_to_remove:
                                if message.startswith(prefix):
                                    message = message[len(prefix):]
                                    break # Remove only one prefix
                            if message.startswith("SELENIUM_LOG_OBJ:"):
                                message = f"JS Object: {message.split('SELENIUM_LOG_OBJ:', 1)[-1]}"
                    except Exception as cleanup_e:
                         logger.debug(f"Minor error during log message cleanup: {cleanup_e}") # Log cleanup errors at debug

                    logs.append(f"[{timestamp_str}] [{level}] [{source}] {message}")
                except Exception as parse_e:
                    # Log error parsing a specific entry but continue with others
                    logger.warning(f"Could not parse individual log entry: {entry}. Error: {parse_e}")
                    logs.append(f"[PARSE_ERROR] Entry: {entry}. Error: {parse_e}")
        else:
             logs.append("[INFO] No browser console logs captured.")

    except WebDriverException as e:
         logger.error(f"WebDriverException capturing browser logs: {e.msg}")
         logs.append(f"[CAPTURE_ERROR] WebDriverException retrieving logs: {e.msg}")
    except Exception as e:
        logger.error(f"Unexpected error capturing browser logs: {e}", exc_info=True)
        logs.append(f"[CAPTURE_ERROR] Unexpected error retrieving logs: {e}")

    return "\n".join(logs)


# --- Render & Capture ---
def render_and_capture(html_file_path: str, screenshot_path: str, browser_log_path: str, browser_type="chrome") -> tuple[bool, str]:
    """
    Renders an HTML file in a headless browser, captures a screenshot, and saves console logs.
    Uses actual Selenium logic with increased wait time.
    """
    if not SELENIUM_AVAILABLE:
        error_msg = "Cannot render/capture: Selenium library not installed."
        logger.error(error_msg)
        # Try creating empty files safely
        try: os.makedirs(os.path.dirname(screenshot_path), exist_ok=True); open(screenshot_path, 'w').close()
        except OSError as e: logger.error(f"Failed to create empty screenshot file on Selenium import error: {e}")
        try: os.makedirs(os.path.dirname(browser_log_path), exist_ok=True); open(browser_log_path, 'w').close()
        except OSError as e: logger.error(f"Failed to create empty log file on Selenium import error: {e}")
        return False, error_msg

    driver = None
    captured_logs = "Logs not captured initially." # Default value
    try:
        # Ensure output directories exist
        try:
             os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
             os.makedirs(os.path.dirname(browser_log_path), exist_ok=True)
        except OSError as e:
             logger.error(f"Failed to create output directories: {e}")
             return False, f"Failed to create output directories: {e}"

        driver = setup_browser(browser_type)
        if not driver:
            # Error already logged by setup_browser
            return False, "Failed to initialize WebDriver."

        absolute_html_path = os.path.abspath(html_file_path)
        if not os.path.exists(absolute_html_path):
             raise FileNotFoundError(f"HTML file not found: {absolute_html_path}")
        file_url = f"file:///{absolute_html_path.replace(os.sep, '/')}"

        # Inject console script BEFORE navigating
        script = """
        try {
            let originalLog = console.log; console.log = function() { originalLog.apply(console, arguments); try { if (arguments.length && typeof arguments[0] === 'object') { originalLog.call(console, 'SELENIUM_LOG_OBJ:' + JSON.stringify(arguments[0])); } else if (arguments.length) { originalLog.call(console, 'SELENIUM_LOG_MSG:' + String(arguments[0])); } } catch (e) { originalLog.call(console, 'SELENIUM_LOG_ERROR: Could not stringify.'); } };
            let originalError = console.error; console.error = function() { originalError.apply(console, arguments); try { if (arguments.length) { originalError.call(console, 'SELENIUM_ERROR_MSG:' + String(arguments[0])); } } catch(e){} };
         } catch(e) {}
        """
        try:
            driver.execute_script(script)
            logger.debug("Injected console log interceptor script.")
        except Exception as script_e:
            # Log warning but continue, log capture might still work partially
            logger.warning(f"Could not execute console interceptor script: {script_e}")

        logger.info(f"Navigating browser to: {file_url}")
        driver.get(file_url)

        # INCREASED WAIT TIME
        wait_seconds = 5
        logger.debug(f"Waiting {wait_seconds} seconds for page load and rendering...")
        time.sleep(wait_seconds)
        logger.debug("Wait finished. Attempting screenshot.")

        logger.info(f"Attempting to save screenshot to: {screenshot_path}")
        screenshot_success = driver.save_screenshot(screenshot_path)

        # Verify screenshot file existence and size
        screenshot_valid = False
        if os.path.exists(screenshot_path) and os.path.getsize(screenshot_path) > 100: # Check size > 100 bytes
             screenshot_valid = True
             logger.info(f"Screenshot saved successfully to {screenshot_path} (Size: {os.path.getsize(screenshot_path)} bytes)")
        else:
             # Log specific error based on why it failed
             if not os.path.exists(screenshot_path):
                 logger.error(f"Failed to save screenshot to {screenshot_path} (File does not exist). Reported success: {screenshot_success}")
             elif os.path.getsize(screenshot_path) <= 100:
                 logger.error(f"Failed to save valid screenshot to {screenshot_path} (File size too small: {os.path.getsize(screenshot_path)} bytes). Reported success: {screenshot_success}")
             # Don't raise RuntimeError here, allow log capture attempt, return False later

        # Attempt to capture logs regardless of screenshot success
        logger.info("Attempting to capture browser logs...")
        captured_logs = capture_browser_logs(driver) # Get logs
        try:
            with open(browser_log_path, 'w', encoding='utf-8') as f:
                 f.write(captured_logs)
            logger.info(f"Browser logs saved to: {browser_log_path}")
        except Exception as log_write_e:
             logger.error(f"Failed to write browser logs to {browser_log_path}: {log_write_e}")
             # Store the error within the captured_logs string if saving failed
             captured_logs += f"\n[ERROR] Failed to write logs to file: {log_write_e}"

        # Return status based on screenshot validity
        if screenshot_valid:
             return True, captured_logs
        else:
             # If screenshot failed, return False, but still provide captured logs
             error_msg = f"Failed to save valid screenshot to {screenshot_path}"
             return False, f"{error_msg}\n\n--- CAPTURED LOGS ---\n{captured_logs}"


    except FileNotFoundError as e:
         error_msg = f"File error during render/capture: {e}"
         logger.error(error_msg)
         # Try to write error to log file
         try: open(browser_log_path, 'w').write(f"Error before log capture:\n{error_msg}")
         except Exception: pass
         return False, error_msg
    except WebDriverException as e:
         error_msg = f"WebDriver error during render/capture: {e.msg}"
         logger.error(error_msg, exc_info=True) # Log full traceback
         try: open(browser_log_path, 'w').write(f"Error before log capture:\n{error_msg}")
         except Exception: pass
         return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error during render/capture: {e}"
        logger.error(error_msg, exc_info=True)
        # Try to create empty screenshot file and write error to log file
        try: open(screenshot_path, 'a').close()
        except Exception as file_e: logger.error(f"Failed creating empty screenshot file on error: {file_e}")
        try: open(browser_log_path, 'w').write(f"Unexpected Error during capture:\n{error_msg}\n\n{captured_logs}")
        except Exception as file_e: logger.error(f"Failed writing error to log file on error: {file_e}")
        return False, error_msg # Return error message on failure

    finally:
        # Ensure driver quit happens reliably
        if driver:
            try:
                driver.quit()
                logger.info("WebDriver quit successfully.")
            except Exception as quit_e:
                logger.warning(f"Error quitting WebDriver: {quit_e}")


# --- File Encoding ---
# (encode_file_base64 and encode_file_inline_data_gemini remain unchanged)
def encode_file_base64(file_path: str) -> str | None:
    try:
        with open(file_path, "rb") as f: return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError: logger.error(f"File not found for encoding: {file_path}"); return None
    except Exception as e: logger.error(f"Error encoding file {file_path}: {e}"); return None

def encode_file_inline_data_gemini(file_path: str) -> dict | None:
    try:
        mime_type, _ = mimetypes.guess_type(file_path); mime_type = mime_type or 'image/png'
        base64_data = encode_file_base64(file_path)
        if base64_data is None: return None
        return {"type": "image", "source": {"inline_data": {"mime_type": mime_type, "data": base64_data}}}
    except Exception as e: logger.error(f"Failed to prep inline data for {file_path}: {e}", exc_info=True); return None


# --- Formatting ---
# (format_feedback_message remains unchanged)
def format_feedback_message(eval_response: str, browser_logs: str) -> str:
    instruction = "Based on the goal and the feedback below, generate the *complete* and *corrected* HTML/JS code, wrapped in <solar.html>...</solar.html> tags."
    return f"""{instruction}

EVALUATION FEEDBACK:
{eval_response}

BROWSER CONSOLE LOGS:
{browser_logs}
--- END FEEDBACK ---
""".strip()