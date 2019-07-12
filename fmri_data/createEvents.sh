#!/bin/bash

niidir=$1
metadir=$2
subj=$3
sess=$4
nruns=10


##Create  event files per subject-session-run triple
for run in $(seq 1 $nruns); do
  # echo $run

  #Generate event tsv if it doesn't exist
  if [ -e ${niidir}/sub-${subj}/ses-${sess}/func/sub-${subj}_ses-${sess}_task-QuiltLanguage_run-${run}_events.tsv ]; then
    echo "Events file already exists."
  else
    #Create events file with headers
    echo -e onset'\t'duration'\t'trial_type'\t'stim_file > ${niidir}/sub-${subj}/ses-${sess}/func/sub-${subj}_ses-${sess}_task-QuiltLanguage_run-${run}_events.tsv
  fi
  #This file will be placed at the level where dataset_description file and subject folders are.
  #The reason for this file location is because the event design is consistent across subjects.
  #If the event design is consistent across subjects, we can put it at this level. This is because of the Inheritance principle.
  # subj=1
  # sess=1
  # run=1
  # metadir=/data0/Dropbox/quilts/fmri_exp/stim_order_run_data/s${subj}/run_data
  if [ $sess == 2 ]; then
    timingrun=$(($run + 10))
  else
    timingrun=$run
  fi
  timingfile=$(ls $metadir/s${subj}_run${timingrun}_stimuli_timing*.txt)
  # timinginfo=( $(cat $timingfile | sed 's/^[ \t]*//;s/[ \t]*$//') | )
  # | sed -e "s/[[:space:]]\+/ /g" # turn all whitespace into single spaces
  onsets=( $(cat $timingfile | sed 's/^[ \t]*//;s/[ \t]*$//'  | cut -d ' ' -f 1) )
  # cat $timingfile | sed 's/^[ \t]*//;s/[ \t]*$//' | sed -e "s/[[:space:]]\+/ /g" > test.txt
  stim_names=( $(cat $timingfile | sed 's/^[ \t]*//;s/[ \t]*$//'  | cut -d ' ' -f 3) )
  lang=( $(cut -d '/' -f 2 -s $timingfile) )
  #
  # #Create onset column
  # echo ${onsets[*]}
  printf "%s\n" "${onsets[@]:1}" > ${niidir}/temponset.txt
  #
  #Create duration column
  echo -e 60'\n'60'\n'60'\n'60'\n'60'\n'60'\n'60'\n'60'\n'60 > ${niidir}/tempdur.txt
  #
  #Create trial_type column
  printf "%s\n" "${lang[@]}"  > ${niidir}/temptrial.txt
  #
  # # Create stim_file column
  stim_names=("${stim_names[@]/stimuli/audio}")
  # echo ${stim_names[*]}
  printf "%s\n" "${stim_names[@]:1}"  > ${niidir}/tempnames.txt

  #Paste onset and duration into events file
  paste -d '\t' ${niidir}/temponset.txt ${niidir}/tempdur.txt ${niidir}/temptrial.txt ${niidir}/tempnames.txt >> ${niidir}/sub-${subj}/ses-${sess}/func/sub-${subj}_ses-${sess}_task-QuiltLanguage_run-${run}_events.tsv

  datalad add --to-git ${niidir}/sub-${subj}/ses-${sess}/func/sub-${subj}_ses-${sess}_task-QuiltLanguage_run-${run}_events.tsv

  # # remove temp files
  rm ${niidir}/tempdur.txt ${niidir}/temponset.txt ${niidir}/temptrial.txt ${niidir}/tempnames.txt

done
