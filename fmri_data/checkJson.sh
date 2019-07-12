#!/bin/bash

set -e

niidir=$1
subj=$2
sess=$3

###Check func json for required fields
#Required fields for func: 'RepetitionTime','VolumeTiming' or 'SliceTiming', and 'TaskName'
#capture all jsons to test
cd ${niidir}/sub-${subj}/ses-${sess}/func #Go into the func folder
for funcjson in $(ls *.json); do
  #Repeition Time exist?
  repeatexist=$(cat ${funcjson} | jq '.RepetitionTime')
  if [[ ${repeatexist} == "null" ]]; then
    echo "${funcjson} doesn't have RepetitionTime defined"
  else
  echo "${funcjson} has RepetitionTime defined"
  fi

  #VolumeTiming or SliceTiming exist?
  #Constraint SliceTiming can't be great than TR
  volexist=$(cat ${funcjson} | jq '.VolumeTiming')
  sliceexist=$(cat ${funcjson} | jq '.SliceTiming')
  if [[ ${volexist} == "null" && ${sliceexist} == "null" ]]; then
  echo "${funcjson} doesn't have VolumeTiming or SliceTiming defined"
  else
  if [[ ${volexist} == "null" ]]; then
  echo "${funcjson} has SliceTiming defined"
  #Check SliceTiming is less than TR
  sliceTR=$(cat ${funcjson} | jq '.SliceTiming[] | select(.>="$repeatexist")')
  if [ -z ${sliceTR} ]; then
  echo "All SliceTiming is less than TR" #The slice timing was corrected in the newer dcm2niix version called through command line
  else
  echo "SliceTiming error"
  fi
  else
  echo "${funcjson} has VolumeTiming defined"
  fi
  fi

  #Does TaskName exist?
  taskexist=$(cat ${funcjson} | jq '.TaskName')
  if [ "$taskexist" == "null" ]; then
  jsonname="${funcjson%.*}"
  taskfield=$(echo $jsonname | cut -d '_' -f2 | cut -d '-' -f2)
  jq '. |= . + {"TaskName":"'${taskfield}'"}' ${funcjson} > tasknameadd.json
  rm ${funcjson}
  mv tasknameadd.json ${funcjson}
  echo "TaskName was added to ${jsonname} and matches the tasklabel in the filename"
  else
  Taskquotevalue=$(jq '.TaskName' ${funcjson})
  Taskvalue=$(echo $Taskquotevalue | cut -d '"' -f2)
  jsonname="${funcjson%.*}"
  taskfield=$(echo $jsonname | cut -d '_' -f2 | cut -d '-' -f2)
  if [ $Taskvalue == $taskfield ]; then
  echo "TaskName is present and matches the tasklabel in the filename"
  else
  echo "TaskName and tasklabel do not match"
  fi
  fi

done
