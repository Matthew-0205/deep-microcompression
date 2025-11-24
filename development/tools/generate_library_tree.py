import sys
import os
import shutil 

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("generate_dmc_library_tree", 
                            description="This program generates the deep microcompression library in your specified lib director.")
    parser.add_argument("lib_dir", help="This is the base of the lib directory where the library tree would be created.")

    args = parser.parse_args()
    lib_dir = args.lib_dir

    deploment_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../deployment/")) 
    destination_dir = os.path.join(lib_dir, "deep_microcompression")

    if os.path.exists(lib_dir) and os.path.exists(deploment_dir):
        if os.path.exists(destination_dir): 
            print(f"deep_micropression lib aleady exists in {lib_dir}")
        else:
            shutil.copytree(deploment_dir, destination_dir)
            print(f"Create deep_microprocompression lib in {lib_dir}")
    else:
        raise NotADirectoryError(f"{lib_dir} doesnot exist")