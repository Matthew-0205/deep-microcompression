"""
@file generate_library_tree.py
@brief Deployment Utility: Exports the C Runtime Library.

This script implements the final hand-off from the Model Development Pipeline 
(Python/PyTorch) to the Bare-Metal Environment (C/C++).

This script copies the static C++ source files (the inference engine core) 
to the target microcontroller project directory.
"""

import os
import shutil 

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("generate_dmc_library_tree", 
                            description="This program generates the deep microcompression library in your specified lib director.")
    parser.add_argument("lib_dir", help="This is the base of the lib directory where the library tree would be created.")

    args = parser.parse_args()
    lib_dir = args.lib_dir

    # Locate the static C++ deployment source files relative to this script
    # This folder contains the headers (layer.h, sequential.h) and implementations
    # that form the runtime engine.
    current_script_path = os.path.dirname(__file__)
    deploment_dir = os.path.abspath(os.path.join(current_script_path, "../../deployment/")) 
    destination_dir = os.path.join(lib_dir, "deep_microcompression")

    if os.path.exists(lib_dir) and os.path.exists(deploment_dir):
        if os.path.exists(destination_dir): 
            print(f"Skipping: DMC library already exists at {destination_dir}")
            print("Delete the existing folder if you wish to update the runtime engine.")
        else:
            shutil.copytree(deploment_dir, destination_dir)
            print(f"Success: Deep Microcompression bare-metal library created at: {destination_dir}")
    else:
        raise NotADirectoryError(f"{lib_dir} doesnot exist")