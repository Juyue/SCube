#!/bin/bash
# Author: Jianfei Guo
# https://github.com/PJLab-ADG/neuralsim/blob/main/dataio/autonomous_driving/waymo/download_waymo.sh

# NOTE: Before proceeding, you need to fill out the Waymo terms of use and complete `gcloud auth login`.

# NOTE: For some reason, could not download file to s3 nfs directly. Therefore, we download to a temporary directory (ec2 home directory) first, then move to the final dataset directory.

jfile=$1 # json file containing the list of Waymo clip ids to download
dest=$2 # destination directory to save the downloaded tfrecords
finaldset=$3 # final dataset name

mkdir -p $dest

# read json file. the json file is a list of clip ids
lst=$(jq -r '.[]' $jfile)
total_files=$(jq '. | length' $jfile)

counter=0

# loop through the clip ids in lst as filename
for filename in $lst; do
    # filename_full is 'segment-' + $filename + '_with_camera_labels'
    filename_full="segment-${filename}_with_camera_labels.tfrecord"

    counter=$((counter + 1))
    echo "[${counter}/${total_files}] Dowloading $filename_full ... (can be OK if No URLs matched)"

    # can be in training
    source=gs://waymo_open_dataset_v_1_4_2/individual_files/training
    gsutil cp -n ${source}/${filename_full} ${dest}

    # or can be in validation
    source=gs://waymo_open_dataset_v_1_4_2/individual_files/validation
    gsutil cp -n ${source}/${filename_full} ${dest}

    sleep 1
    # move the file to the final dataset directory
    cp ${dest}/${filename_full} ${finaldset}/${filename_full} 

    sleep 1
    rm ${dest}/${filename_full}
done