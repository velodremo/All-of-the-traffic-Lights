import shutil
import os


def filename_from_path(path, without_postfix=True):
    name = path.split("/")[-1]
    if without_postfix:
        name = ".".join(name.split(".")[:-1])
    return name


def safe_delete_dir(path):
    if not os.path.exists(path):
        return
    answer = input("are you sure you want to delete dir {}? (y/n)".format(path))
    while answer not in ["y", "n"]:
        answer = input("invalid input! \n", "are you sure you want to delete dir {}? (y/n)".format(path))
    if answer == "y":
        shutil.rmtree(path)
    else:
        exit()


