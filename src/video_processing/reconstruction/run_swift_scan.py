"""
./src/video_processing/reconstruction/run_swift_scan.py

Uses native Swift commands to create the 3D model from the video using Photogrammetry.
"""

import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input-folder",
    default="data/env_imgs/albert_room",
    help="The relative path to your folder of images",
)
parser.add_argument(
    "--output-file",
    default="data/reconstructions/albert_room.usdz",
    help="Output file for the .usdz file",
)

args = parser.parse_args()


def native_scan(input_dir: str, output_file: str) -> None:
    """
    Runs a scan on the folder of images to create the 3D environment using native SwiftUI.

    input_dir (str): The input directory of images.
    output_file (str): The output file.
    """
    abs_input = os.path.abspath(input_dir)
    abs_output = os.path.abspath(output_file)

    swift_script = f"""
import RealityKit
import Foundation

let inputFolderUrl = URL(fileURLWithPath: "{abs_input}", isDirectory: true)
let outputFileUrl = URL(fileURLWithPath: "{abs_output}")

Task {{
    do {{
        let session = try PhotogrammetrySession(input: inputFolderUrl)
        let request = PhotogrammetrySession.Request.modelFile(url: outputFileUrl, detail: .full)
        try session.process(requests: [request])

        // Added 'try' here to fix the error you saw
        for try await output in session.outputs {{
            switch output {{
            case .requestProgress(_, let fraction):
                print("Progress: \\(Int(fraction * 100))%")
            case .requestComplete(_, _):
                print("3D Model File Created!")
                exit(0)
            case .requestError(_, let error):
                print("Error: \\(error)")
                exit(1)
            case .processingComplete:
                print("All tasks finished.")
                exit(0)
            default:
                // This 'default' fixes the 'must be exhaustive' warning
                break
            }}
        }}
    }} catch {{
        print("Fatal Error: \\(error)")
        exit(1)
    }}
}}

print("Reconstructing room... (Check Activity Monitor for 'RealityAssets' CPU usage)")
RunLoop.main.run()
"""
    with open("process.swift", "w") as f:
        f.write(swift_script)

    print("Starting Apple's goofy 3D reconstruction")
    try:
        subprocess.run(["swift", "process.swift"])
    except Exception as e:
        print("So whoever wrote this code is lowkey dum: ", e)
    print("Done.")

    os.remove("process.swift") if os.path.exists("process.swift") else print(
        "Something went wrong..."
    )


native_scan(args.input_folder, args.output_file)
