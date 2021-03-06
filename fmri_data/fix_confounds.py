# sudo chown jessica sub-5/ses-2/func/sub-5_ses-2_task-QuiltLanguage_run-9_desc-confounds_regressors.tsv
"""Replace NaN's in confound regressors file with means."""
import pandas as pd
import numpy as np

BIDS_dataset = '/data1/quilts/speechQuiltsfMRIdata'
confound_file = BIDS_dataset + '/derivatives/fmriprep/sub-{0}/ses-{1}/func/sub-{0}_ses-{1}_task-QuiltLanguage_run-{2}_desc-confounds_regressors.tsv'
confound_file = BIDS_dataset + '/derivatives/fmriprep/sub-{0}/ses-{1}/func/sub-{0}_ses-{1}_task-QuiltLanguage_run-{2}_desc-confounds_regressors.tsv'
for sub in np.arange(5, 6):
    for ses in np.arange(1, 3):
        for run in np.arange(1, 11):
            file_to_fix = confound_file.format(sub, ses, run)
            # Load confounds TSV file
            df = pd.read_csv(file_to_fix, sep='\t')
            for col in df.columns:
                if np.isnan(df[col][0]):
                    print('Found NaN in ', col)
                    # Replace NaN with mean of the rest of the column
                    df[col][0] = df[col][1:].mean()
            # Save confounds TSV file
            df.to_csv(file_to_fix, sep='\t')


# Update event duration to be 59 seconds instead of 60 (which we did to shorten the total duration of the experiment slightly)
events_file = BIDS_dataset + '/sub-{0}/ses-{1}/func/sub-{0}_ses-{1}_task-QuiltLanguage_run-{2}_events.tsv'
for sub in np.arange(1, 7):
    for ses in np.arange(1, 3):
        for run in np.arange(1, 11):
            file_to_fix = events_file.format(sub, ses, run)
            # Load confounds TSV file
            df = pd.read_csv(file_to_fix, sep='\t')
            df.duration = 59
            # Save confounds TSV file
            df.to_csv(file_to_fix, sep='\t')
