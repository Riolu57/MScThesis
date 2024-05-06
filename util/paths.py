import os


def get_subdirs(path: str) -> list:
    """Filters out non-subdirectories from a given path.

    @param path: Path to be analyzed.
    @return: List of subdirectories.
    """
    dir_names = os.listdir(path)
    dir_names = [
        dir_name
        for dir_name in dir_names
        if os.path.isdir(os.path.join(path, dir_name))
    ]
    return dir_names


def get_subfiles(path: str) -> list:
    """Filters out non-files from a given path.

    @param path: Path to be analyzed.
    @return: List of files in path.
    """
    file_names = os.listdir(path)
    file_names = [
        file_name
        for file_name in file_names
        if not os.path.isdir(os.path.join(path, file_name))
    ]
    return file_names
