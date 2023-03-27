from ID_matching import *
from RFID import *
from DLC import *

import argparse

parser = argparse.ArgumentParser(
                    prog = 'MPIT',
                    description = 'Mouse Pose estimation, Identification and Tracking',
                    epilog = 'Text at the bottom of help')

parser.add_argument('-i',
                    '--input_files_directory', 
                    type=str,
                    metavar='',
                    required=True,
                    help='The path to the directory that holds all required input data. In this directory there must exist three files:\n' \
                         '- `mp4` or `H264` raw video file containing animal(s). # todo test multiple videos in the same folder.\n' \
                         '- `h5` file that is analyzed by DLC. The file is in a hierarchical data format and should contain the coords of all individuals across video frames.\n' \
                         '- `CSV` file holding all RFID detections. The file must contain the animal tag name, position, timestamp of reading and the duration of each event.\n' \
                         '- `txt` file that contains timestamp of the first video frame.')

parser.add_argument('-o',
                    '--output_directory', 
                    type=str,
                    metavar='',
                    required=True,
                    help='The path to directory where all results should be stored.')

parser.add_argument('-t',
                    '--length_time_interval',
                    type=float,
                    metavar='',
                    default= 2.0,
                    required=False,
                    help='Length of the time interval in seconds.' \
                    'The RFID detections are divided into intervals that are necessary for the DLC-RFID '\
                    'procedure. An interval could contain several occurrences (RFID events). For each '\
                    'time interval, only one position for each RFID tag will be estimated (or no position '\
                    'if no readings occur within the interval). This will allow to have a unique and decent '\
                    'estimated tag position.')

parser.add_argument('-m',
                    '--correction_method', 
                    type=str,
                    metavar='',
                    default= 'all',
                    choices = ['bp_perFrame', 'centroid_perFrame', 'bp_frameDistDiff', 'centroid_frameDistDiff', 'all'],
                    required=False,
                    help='Method used for fixing the identity switching. The methods function once the ID-matching procedure is '\
                         'accomplished and each mouse is assigned an RFID tag. There exists four methods (todo find bp of centroid '\
                         'and compare with other method):\n'\
                         '- `bp_perFrame`: proceed frame by frame, mouse by mouse based on body parts.\n'\
                         '- `centroid_perFrame`: proceed frame by frame, mouse by mouse based on the centroid.\n'\
                         '- `bp_frameDistDiff`: Predict the identity swapping based on a euclidean distance threshold value between a mouse DeepLabCut-detections in each two consecutive frames (based on body parts).\n'\
                         '- `centroid_frameDistDiff`: Predict the identity swapping based on a euclidean distance threshold value between a mouse DeepLabCut-detections in each two consecutive frames (based on centroid).\n'\
                         '- `all` : (default) Performs all method and store the results in different folders. # todo: change the default to centroid!')

parser.add_argument('-d',
                    '--minDistDiff_threshold', 
                    type=int,
                    metavar='',
                    default= 50,
                    required=False,
                    help='Minimum value used as a threshold for the correction methods `bp_frameDistDiff` and `centroid_frameDistDiff`.')

parser.add_argument('-p',
                    '--plot_type', 
                    type=str,
                    metavar='',
                    default= 'all',
                    choices = ['RFID', 'DLC_bp', 'DLC_centroid', 'all'],
                    required=False,
                    help='Determines which labels should be plotted on the video. Use `RFID` to only plot RFID annotation (big circles), '\
                         '`DLC_bp` for plotting only the body parts of the animals predicted by DLC (small circles) and `DLC_centroid` '\
                         'for plotting the centroid of the DLC keypoints predictions (rings). `all` will plot all mentioned annotations (set as default)')

parser.add_argument('-l',
                    '--likelihood_threshold', 
                    type=float,
                    metavar='',
                    default= 0.9,
                    required=False,
                    help='A threshold value used for filtering the DLC detections when calculating the centroid. Labels with likelihood lower than this value will be excluded.')

args = parser.parse_args()

    
def main():
    files_directory = args.input_files_directory
    output_directory = args.output_directory
    length_time_interval = args.length_time_interval
    correction_method = args.correction_method
    minDistDiff_threshold = args.minDistDiff_threshold
    plot_type = args.plot_type
    likelihood_threshold = args.likelihood_threshold
    
    # files_directory = r"E:\Tracking\MOTA_01.32.31--01.34.31__2s"
    # output_directory = r"E:\Tracking\MOTA_01.32.31--01.34.31__2s\identification_output"
    # length_time_interval = 2.0 # float number only!
    # minDistDiff_threshold = 50
    # correction_method = "all"
    # plot_type = 'all'
    # likelihood_threshold = 0.9
    # Name of the RFID readers. Note that the order is very important. In this order the coords position will by annotated using the left mouse click
    RFID_reader_names = ['R1.1', 'R1.2', 'R1.3', 'R1.4', 
                         'R2.1', 'R2.2', 'R2.3', 'R2.4']

    # get some informations about the data
    input_files_directory, input_video_path, file_start_ts, dlc_file_path, dlc_file_name, RFID_file_path = get_files_paths(files_directory)
    video_name, output_directory, ts_start, frame, width, height, fps, frame_count, duration, minutes, seconds = get_video_info(input_video_path, input_files_directory, file_start_ts, output_directory)

    # Process RFID and DLC
    RFID_df_coords, RFID_df_mean = process_RFID(input_video_path, output_directory, video_name, RFID_file_path, ts_start, duration, fps, height, width, length_time_interval, RFID_reader_names, method='first_video_frame')
    DLC_hdf_centroid, DLC_hdf, dlc_hdf_info =  process_DLC(dlc_file_path, likelihood_threshold)


    # match and assign IDs
    DLC_hdf, DLC_hdf_centroid = match_id(RFID_df_coords, RFID_df_mean, dlc_hdf_info, DLC_hdf, DLC_hdf_centroid, length_time_interval, fps, output_directory)
    save_DLC_hdf(DLC_hdf, output_directory, dlc_file_name, 'without_correction')
    plot_annotatins_on_video(plot_type,
                            input_video_path, 
                            output_directory,
                            video_name + "_" + correction_method,
                            length_time_interval,
                            RFID_df_coords,
                            RFID_df_mean,
                            DLC_hdf,
                            DLC_hdf_centroid,
                            dlc_hdf_info)
    
    # Fix identity switching
    DLC_hdf_corrected, nr_total_corrections = fix_indetity_switching(correction_method, RFID_df_coords, RFID_df_mean, DLC_hdf, dlc_hdf_info, length_time_interval, fps, output_directory, DLC_hdf_centroid, minDistDiff_threshold)
    

    # plot and save annotations
    if type(DLC_hdf_corrected) == list:
        for i, method in zip(range(4), ['bp_perFrame', 'centroid_perFrame', 'bp_frameDistDiff', 'centroid_frameDistDiff']):
            
            save_DLC_hdf(DLC_hdf, output_directory, dlc_file_name, 'matched_with_ID_'+ method)
            
            plot_annotatins_on_video(plot_type,
                        input_video_path, 
                        output_directory,
                        video_name + "_" + method,
                        length_time_interval,
                        RFID_df_coords,
                        RFID_df_mean,
                        DLC_hdf_corrected[i],
                        DLC_hdf_centroid,
                        dlc_hdf_info)

            print('Number of total corrections:', nr_total_corrections[i])
    
    else:
        save_DLC_hdf(DLC_hdf, output_directory, dlc_file_name, 'matched_with_ID_' + correction_method)
        
        plot_annotatins_on_video(plot_type,
                                input_video_path, 
                                output_directory,
                                video_name + "_" + correction_method,
                                length_time_interval,
                                RFID_df_coords,
                                RFID_df_mean,
                                DLC_hdf_corrected,
                                DLC_hdf_centroid,
                                dlc_hdf_info)

        print('Number of total corrections:', nr_total_corrections)



if __name__ == "__main__":
    main()