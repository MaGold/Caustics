#!/bin/bash
set -e
# This script will run through each directory in the "data/real_frames" directory.
# For each such subdirectory, the images will be run by the python script
# 'process.py', which feeds the images through the trained network.
# Each subdirectory yields a new directory in "/output_videos/" containing:
#    - the output images from the network
#    - original.mp4: A video of the original input frames
#    - cropped.mp4: A cropped version of original.mp4
#    - caustics.mp4: A video of the output images
#    - side-by-side.mp4: A video showing both cropped.mp4 and caustics.mp4
base=$PWD
for d in data/real_frames/*/ ; do
    echo "------------------------------------------------------------------"
    echo "Processing images in $d ..."
    # rm caustic_frames/*
    echo "$d"
    rm -f "${d}"/*.mp4
    python3 process.py "$d"
    echo "Done python processing."
    cd caustic_frames
    echo "Creating caustics video..."
    ffmpeg -framerate 50 -i %d.png -c:v libx264 -r 30 -pix_fmt yuv420p caustics.mp4
    cd "$base"
    path=$d
    cd $path
    echo "Creating original video..."
    ffmpeg -framerate 50 -i "%04d.png" -c:v libx264 -r 30 -pix_fmt yuv420p original.mp4

    echo "Creating cropped version of original video..."
    ffmpeg -i original.mp4 -filter:v "crop=480:480:0:0" cropped.mp4

    mv original.mp4 "$base/caustic_frames/original.mp4"
    mv cropped.mp4 "$base/caustic_frames/cropped.mp4"

    cd "$base/caustic_frames"
    tomake=${d#*/*/}
    vid=${tomake::-1} # remove last /
    echo "Creating side-by-side video of cropped video with caustics video..."
    ffmpeg -i cropped.mp4 -vf "[in] pad=2*iw:ih [left]; movie=caustics.mp4 [right]; [left][right] overlay=main_w/2:0 [out]" "$vid"-side-by-side.mp4
    cd "$base"
    
    echo "$tomake"
    echo "output_videos/$tomake"
    mkdir -p "output_videos/$tomake"
    mv caustic_frames/* "output_videos/$tomake"
done
