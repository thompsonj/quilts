#!/bin/bash

set -e

#### Defining pathways
toplvl=/data1/quilts/speechQuiltsfMRIdata
datalad create $toplvl

dcmdir=/data1/quilts/dicoms
metadir=/data0/Dropbox/quilts/fmri_exp/stim_order_run_data
# Create nifti directory
# mkdir -p ${toplvl}/Nifti
niidir=${toplvl}

for subj in 1 2 3 4 5 6 ; do
  datalad create -d $toplvl ${toplvl}/sub-${subj}
# subj=1
  for sess in 1 2; do
    datalad create -d $toplvl ${toplvl}/sub-${subj}/ses-${sess}
  # sess=1
    echo "Processing subject $subj session $sess"

    ### Convert to Nifti format
    ### rename and reorganize according to BIDS specification
    ./toNifti.sh $niidir $dcmdir $subj $sess
    wait
    ### Create events tsv files
    ./createEvents.sh $niidir $metadir/s${subj}/run_data $subj $sess
    wait
    ### Check func json for required fields
    ./checkJson.sh $niidir $subj $sess

  done
  echo "${subj} complete!"
done
