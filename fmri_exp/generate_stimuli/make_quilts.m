% make_quilts.m

segdir = '/Users/jthompson/data/enu/inh/seg/';
wavdir = '/Users/jthompson/data/enu/inh/wav/';
lang = 'enu';
% lang = 'deu';
segdir = ['/Users/jthompson/data/' lang '/seg/'];
wavdir = ['/Users/jthompson/data/' lang '/wav/'];
stimdir = ['/Users/jthompson/Dropbox/quilts/stimuli/' lang '/']; 
out_length_in_sec = 60;

% seg_dir and wav_dir have the same organization. They both contain one
% directory per basename. This basename is the name of the original nwv file
% from which the individual utterances were extracted. Each .mat file or .wav in
% these directories represents one utterance. The filenames of corresponding
% audio and segmentation info are identical except for the extension. 

% Set quilt parameters
%values from NN paper:
win_ms = 30;

P.filt_density = 1; %1 for regular filterbank, 2 for 4x overcomplete
P.N_audio_channel = 30;
P.audio_low_lim_Hz = 20;
P.audio_high_lim_Hz = 8000;
P.audio_sr = 16000;

% Which quilting method to use when reordering audio segments
method = 3;

% matlist = dir([segdir '*.mat']);
% d = dir(segdir);
% isub = [d(:).isdir]; %# returns logical vector
% nameFolds = {d(isub).name}';
% nameFolds(ismember(nameFolds,{'.','..'})) = [];

% spklistf = dir([stimdir 'selected_utts/female/*.txt']);
% spklistm = dir([stimdir 'selected_utts/male/*.txt']);
% 
% spklist = cat(1, spklistf, spklistm);

spkr_list_m = fopen([stimdir 'selected_male_spkrs.txt']);
M = textscan(spkr_list_m,'%s');
mspkrs = M{1};
fclose(spkr_list_m);

spkr_list_f = fopen([stimdir 'selected_female_spkrs.txt']);
F = textscan(spkr_list_f,'%s');
fspkrs = F{1};
fclose(spkr_list_f);

allspkrs = [mspkrs fspkrs];
% allspkrs = fspkrs;
allgenders = {'male', 'female'};

for g = 1:2
    spkrs = allspkrs(:,g);
    gender = allgenders{g}
    for i = 1:length(spkrs)
%         if g==2 && i <= 1 % first 1 female speakers done already.
%             continue
%         end
%         if g==1 && i <= 9 % first 6 female speakers done already.
%             continue
%         end
    %     name = 'hub4_0029_f1';
    %     if i<5
    %         continue
    %     end
    %     segbasename = nameFolds{i};
    %     name = segbasename(5:end);
    %     name = segbasename;
        fileID = fopen([stimdir 'proposed_utts/' spkrs{i}]);
        C = textscan(fileID,'%f');
        fclose(fileID);
        utts = C{1};
        name = spkrs{i}(1:end-4); % remove .txt extension
        [source_s, sr, phn_start, phn_end, phns] = get_audio_and_seg(name, utts, segdir, wavdir);
        if phn_end(end) > length(source_s)
            display(['*** phn_end(end): ' num2str(phn_end(end)) '> length(source_s) ' num2str(length(source_s))])
            phn_end(end) = length(source_s);
        end
        
        start_smp = round(phn_start * (P.audio_sr/1000))+1; % add 1 because MATLAB?
%         stop_smp = round((phn_end+10) * (P.audio_sr/1000));% add 10 ms because one frame = 10 ms
        stop_smp = round((phn_end) * (P.audio_sr/1000));% I added the 10ms where phn_end is saved
        
        len_all_segs = sum(stop_smp - start_smp)/sr

    %     matlist = dir([segdir filesep segbasename filesep '*.mat']);
    %     for j = 1:length(matlist) % for each utterance
    %         [pathstr,base,ext] = fileparts(matlist(j).name);
    %         baseparts = strsplit(base, '_');
    %         utt = baseparts{end};
    %         % load segmentation info
    %         display('Loading segmentation...')
    %         segfname = [segdir filesep segbasename filesep segbasename '_' utt]
    %         load(segfname)
    %         
    %         % load audio
    %         display('loading audio...')
    %         wavname = [wavdir, name, filesep, name, '_', utt,'.wav']
    %         [source_s,sr] = audioread(wavname);

            % If wav was stereo, couldn't use length(source_s)
        stim_length_in_sec=length(source_s)/sr;
        out_length_in_sec = 60; % Two minutes = 120 seconds, too long for deu

        if sr~=P.audio_sr
            source_s = resample(source_s,P.audio_sr,sr);
        end
    %     bound_method = 2;
    %     segment_bounds = get_bounds(phn_start, phn_end, bound_method, P.audio_sr, length(source_s));

        phonedir = [stimdir filesep 'phonemes' filesep 'border_mid' filesep name];
    %     mkdir(phonedir)
        [quilted_s, final_seg_order, source_seg_changes, quilt_seg_changes, kl] = ...
                generate_quilt(source_s, win_ms, start_smp, stop_smp, ...
                stim_length_in_sec, out_length_in_sec, P, method, phonedir);

        display('Saving quilt...')
        fname = sprintf('%squilts/%s/%s_%ds.wav', stimdir, gender, name, out_length_in_sec)
        audiowrite(fname, quilted_s, P.audio_sr)

        % Save seg order with the quilts
        fname = sprintf('%squilts/%s/%s_%ds_order.mat', stimdir, gender, name, out_length_in_sec)
        save(fname, 'final_seg_order')
    end
end
 
    
    