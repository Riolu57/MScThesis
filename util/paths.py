import os


def get_subdirs(path):
    dir_names = os.listdir(path)
    dir_names = [
        dir_name
        for dir_name in dir_names
        if os.path.isdir(os.path.join(path, dir_name))
    ]
    return dir_names


def get_subfiles(path):
    file_names = os.listdir(path)
    file_names = [
        file_name
        for file_name in file_names
        if not os.path.isdir(os.path.join(path, file_name))
    ]
    return file_names
