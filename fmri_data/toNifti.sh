#!/bin/bash

niidir=$1
dcmdir=$2
subj=$3
sess=$4

###Create dataset_description.json
jo -p "Name"="Maastricht 7T Speech Quilts Dataset" "BIDSVersion"="1.0.2" > ${niidir}/dataset_description.json
datalad add --to-git ${niidir}/dataset_description.json

####Anatomical Organization####
###Create structure
mkdir -p ${niidir}/sub-${subj}/ses-${sess}/anat

###Convert dcm to nii
#Only convert the Dicom folder anat
for direc in T1 PD; do
  dcm2niix -o ${niidir}/sub-${subj} -f ${subj}_%f_%p -z y ${dcmdir}/s0${subj}_sess${sess}/${direc}
done

#Changing directory into the subject folder
cd ${niidir}/sub-${subj}
echo $pwd

###Change filenames
##Rename anat files
#Example filename: 2475376_anat_MPRAGE
#BIDS filename: sub-2475376_ses-1_T1w
#Capture the number of anat files to change
for anattype in T1 PD; do
  anatfiles=$(ls -1 *${anattype}* | wc -l)
  for ((i=1;i<=${anatfiles};i++)); do
    Anat=$(ls *${anattype}*) #This is to refresh the Anat variable, if this is not in the loop, each iteration a new "No such file or directory error", this is because the filename was changed.
    tempanat=$(ls -1 $Anat | sed '1q;d') #Capture new file to change
    tempanatext="${tempanat##*.}"
    tempanatfile="${tempanat%.*}"
    if [ $anattype == 'T1' ] ; then # add w to modality_label for T1 only
      mv ${tempanatfile}.${tempanatext} sub-${subj}_ses-${sess}_${anattype}w.${tempanatext}
      echo "${tempanat} changed to sub-${subj}_ses-${sess}_${anattype}w.${tempanatext}"
    else
      mv ${tempanatfile}.${tempanatext} sub-${subj}_ses-${sess}_${anattype}.${tempanatext}
      echo "${tempanat} changed to sub-${subj}_ses-${sess}_${anattype}.${tempanatext}"
    fi
  done
done

###Organize files into folders
for files in $(ls sub*); do
  Orgfile="${files%.*}"
  Orgext="${files##*.}"
  Modality=$(echo $Orgfile | rev | cut -d '_' -f1 | rev)
  if [ $Modality == "T1w" ] || [ $Modality == "PD" ] ; then
    mv ${Orgfile}.${Orgext} ses-${sess}/anat
  else
  :
  fi
done

# add anat to datalad dataset
datalad add "ses-${sess}/anat"


####Functional Organization####
#Create subject folder
mkdir -p ${niidir}/sub-${subj}/ses-${sess}/func
mkdir -p ${niidir}/sub-${subj}/ses-${sess}/fmap

###Convert dcm to nii
for direcs in AP1 PA1 run1 run2 run3 run4 run5 run6 run7 run8 run9 run10; do
  dcm2niix -o ${niidir}/sub-${subj} -f ${subj}_${direcs}_%p ${dcmdir}/s0${subj}_sess${sess}/${direcs}
done

#Changing directory into the subject folder
cd ${niidir}/sub-${subj}

##Rename func files
#Example filename: 2475376_TfMRI_visualCheckerboard_645_CHECKERBOARD_645_RR
#BIDS filename: sub-2475376_ses-1_task-Checkerboard_acq-TR645_bold
#Capture the number of files to change
runfiles=$(ls -1 *run*_cmrr* | wc -l)
for ((i=1;i<=${runfiles};i++)); do
  run=$(ls *run*_cmrr*) #This is to refresh the Checker variable, same as the Anat case
  temprun=$(ls -1 $run | sed '1q;d') #Capture new file to change
  runstring=$(echo $temprun | cut -d '_' -f2) # e.g. 'run10'
  r=${runstring:3} # e.g. '10'
  temprunext="${temprun##*.}"
  temprunfile="${temprun%.*}"
  # TR=$(echo $temprun | cut -d '_' -f4) #f4 is the third field delineated by _ to capture the acquisition TR from the filename
  mv ${temprunfile}.${temprunext} sub-${subj}_ses-${sess}_task-QuiltLanguage_run-${r}_bold.${temprunext}
  echo "${temprunfile}.${temprunext} changed to sub-${subj}_ses-${sess}_task-Identify_Language_run-${r}_bold.${temprunext}"
done

## Rename reference epis in opposite phase encoding directions
apfiles=$(ls -1 *AP1_cmrr* | wc -l)
for ((i=1;i<=${apfiles};i++)); do
  ap=$(ls *AP1_cmrr*) #This is to refresh the Checker variable, same as the Anat case
  tempap=$(ls -1 $ap | sed '1q;d') #Capture new file to change
  tempapext="${tempap##*.}"
  tempapfile="${tempap%.*}"
  # TR=$(echo $temprun | cut -d '_' -f4) #f4 is the third field delineated by _ to capture the acquisition TR from the filename
  mv ${tempapfile}.${tempapext} sub-${subj}_ses-${sess}_dir-AP_epi.${tempapext}
  echo "${tempapfile}.${tempapext} changed to sub-${subj}_ses-${sess}_dir-AP_epi.${tempapext}"
done
pafiles=$(ls -1 *PA1_cmrr* | wc -l)
for ((i=1;i<=${pafiles};i++)); do
  pa=$(ls *PA1_cmrr*) #This is to refresh the Checker variable, same as the Anat case
  temppa=$(ls -1 $pa | sed '1q;d') #Capture new file to change
  temppaext="${temppa##*.}"
  temppafile="${temppa%.*}"
  # TR=$(echo $temprun | cut -d '_' -f4) #f4 is the third field delineated by _ to capture the acquisition TR from the filename
  mv ${temppafile}.${temppaext} sub-${subj}_ses-${sess}_dir-PA_epi.${temppaext}
  echo "${temppafile}.${temppaext} changed to sub-${subj}_ses-${sess}_dir-PA_epi.${temppaext}"
done
jo -p "Intended For"="["ses-${sess}/func/sub-2_ses-${sess}_task-QuiltLanguage_run-1_bold.nii",
                  "ses-${sess}/func/sub-2_ses-${sess}_task-QuiltLanguage_run-2_bold.nii",
		  "ses-${sess}/func/sub-2_ses-${sess}_task-QuiltLanguage_run-3_bold.nii",
		  "ses-${sess}/func/sub-2_ses-${sess}_task-QuiltLanguage_run-4_bold.nii",
		  "ses-${sess}/func/sub-2_ses-${sess}_task-QuiltLanguage_run-5_bold.nii",
		  "ses-${sess}/func/sub-2_ses-${sess}_task-QuiltLanguage_run-6_bold.nii",
		  "ses-${sess}/func/sub-2_ses-${sess}_task-QuiltLanguage_run-7_bold.nii",
		  "ses-${sess}/func/sub-2_ses-${sess}_task-QuiltLanguage_run-8_bold.nii",
	  	  "ses-${sess}/func/sub-2_ses-${sess}_task-QuiltLanguage_run-9_bold.nii",
		  "ses-${sess}/func/sub-2_ses-${sess}_task-QuiltLanguage_run-10_bold.nii"]" >> sub-${subj}_ses-${sess}_dir-PA_epi.json


###Organize files into folders
for files in $(ls sub*); do
  Orgfile="${files%.*}"
  Orgext="${files##*.}"
  Modality=$(echo $Orgfile | rev | cut -d '_' -f1 | rev)

  if [ $Modality == "bold" ]; then
    mv ${Orgfile}.${Orgext} ses-${ses}/func
  else
  if [ $Modality == "epi" ]; then
    mv ${Orgfile}.${Orgext} ses-${ses}/fmap
  fi
  fi
done

# Add functional files to datalad dataset
datalad add "ses-${sess}/func"
datalad add "ses-${sess}/fmap"
