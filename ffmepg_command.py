

def command_compression(original_data_path, new_data_path):
    """
    This function generates the ffmpeg command for video formatting.
    """
    # command for black and white ffmpeg -i input -vf format=gray output
    # command for different compression ffmpeg -i input -c:v libx265 -crf 26 -preset fast 


    command = f'ffmpeg -i {original_data_path}  -c:v libx265 -crf 26 -preset fast {new_data_path}'
    
    return command

