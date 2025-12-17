import os
import subprocess
import adress_folder


# creare a file adress_folder.py with the input_folder and output_folder variables

# step 1 : detect all the path to video files
input_folder = adress_folder.input_folder
output_folder = adress_folder.output_folder
run_deface = False
run_ffmpeg = True
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# check in all subfolders for video files
video_files = []
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            full_path = os.path.join(root, file)
            video_files.append(full_path)

print(f"Found {len(video_files)} video files to process.")
# Extract the folder path  before the file 
crf_to_do = [10,15,23,30,40]
black_and_white = True

for black_and_white in [True, False]:
    for crf in crf_to_do:
        output_folder_deface_crf = os.path.join(output_folder,"deface" ,
                                              f"crf_{crf}", "bw" if black_and_white else "color")
        output_folder_nodeface_crf= os.path.join(output_folder,"no_deface" ,
                                              f"crf_{crf}", "bw" if black_and_white else "color")
        if not os.path.exists(output_folder_deface_crf):
            os.makedirs(output_folder_deface_crf)
        if not os.path.exists(output_folder_nodeface_crf):
            os.makedirs(output_folder_nodeface_crf)
            
        for video_file in video_files:
            relative_path = os.path.relpath(video_file, input_folder)
            # Change extension to .mp4 for HEVC compatibility
            base_name = os.path.splitext(relative_path)[0]
            output_path = os.path.join(output_folder_nodeface_crf, base_name + ".mp4")
            output_path_deface = os.path.join(output_folder_deface_crf, base_name + ".mp4")
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            if black_and_white:
                command_compression = [
                        "ffmpeg",
                        "-y",
                        "-i", video_file,
                        "-vf", "format=gray",
                        "-c:v", "libx265",
                        "-crf", str(crf),
                        "-preset", "fast",
                        output_path
                    ]
            else:
                command_compression = [
                        "ffmpeg",
                        "-y",
                        "-i", video_file,
                        "-c:v", "libx265",
                        "-crf", str(crf),
                        "-preset", "fast",
                        output_path
                    ]
            if run_ffmpeg:
                try:
                    print(f"Processing: {relative_path}")
                    result = subprocess.run(command_compression, capture_output=True, text=True, check=True)
                    print(f"Processed: {relative_path}")
                except subprocess.CalledProcessError as e:
                    print(f"  Return code: {e.returncode}")
                    print(f"  STDERR: {e.stderr}")
            
            command_deface = ["deface",
                        output_path, 
                        "--output", output_path_deface,
                        "--replacewith", "solid",
                        "--ffmpeg-config","{\"codec\": \"libx265\"}" 
                        "-t", "0.1", "--scale", "640x360"
                ]
            
            f'"codec":"libx265","crf":26,"preset":"fast","pix_fmt":"yuv420p"'
            # check that the output full path exists
            output_dir_deface = os.path.dirname(output_path_deface)
            if not os.path.exists(output_dir_deface):
                os.makedirs(output_dir_deface)
            if run_deface:
                try:
                    print(f"Defacing: {relative_path}")
                    result = subprocess.run(command_deface, capture_output=True, text=True, check=True)
                    print(f"✓ Defaced: {relative_path}")
                except subprocess.CalledProcessError as e:
                    print(f"  Return code: {e.returncode}")
                    print(f"  STDERR: {e.stderr}")


