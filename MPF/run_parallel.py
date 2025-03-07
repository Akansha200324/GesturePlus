import multiprocessing
import subprocess
import os

def run_python_file(file_path):
    """Runs a Python file using subprocess."""
    try:
        # Construct the command to execute the Python file
        command = ["python", file_path]  # Or sys.executable for current Python

        # Execute the command and capture output/errors (optional)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()  # Wait for the process to finish

        if process.returncode != 0:
            print(f"Error running {file_path}:\n{stderr.decode()}") # Decode if bytes
        else:
            print(f"{file_path} completed successfully:\n{stdout.decode()}") # Decode if bytes
        return process.returncode # Return the exit code

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return 1  # Indicate an error
    except Exception as e:
        print(f"An unexpected error occurred while running {file_path}:\n{e}")
        return 1

def main():
    """Runs multiple Python files in parallel using multiprocessing."""

    # Get the current directory (where the script is located)
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # List of Python files to run (relative to the current directory)
    python_files = ["Gesture_Controller.py", "proton.py"] # Add your file names here

    full_paths = [os.path.join(current_directory, file) for file in python_files]

    processes = []
    for file_path in full_paths:
        process = multiprocessing.Process(target=run_python_file, args=(file_path,))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()  # Important to prevent zombie processes

    # Check for any errors
    any_errors = False
    for process in processes:
        if process.exitcode != 0:
            any_errors = True
            break
    
    if any_errors:
        print("Some Python files encountered errors.")
    else:
        print("All Python files executed successfully.")


if __name__ == "__main__":
    main()
