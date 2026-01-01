import os

def ensure_dir(path: str) -> None:
    """
    Ensures that a directory exists at the specified path.
    
    If the directory structure does not exist, it is created automatically 
    (equivalent to the shell command `mkdir -p`). This prevents I/O errors 
    when the application attempts to save logs, graphs, or experiment results 
    to a nested folder that hasn't been initialized yet.

    Args:
        path: The directory path to verify or create. If an empty string 
              is provided, the function does nothing.
    """
    if path and not os.path.exists(path):
        # exist_ok=True prevents a race condition error if the 
        # directory is created by another process between the check and the call.
        os.makedirs(path, exist_ok=True)