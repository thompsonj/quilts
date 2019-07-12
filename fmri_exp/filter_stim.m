clear
lang = 'nld';
stimdir = ['/Users/jthompson/Dropbox/quilts/stimuli/' lang '/']
out_length_in_sec = 60;

% read in stimuli paths
spkr_list_m = fopen([stimdir 'selected_male_spkrs.txt']);
M = textscan(spkr_list_m,'%s');
mspkrs = M{1};
fclose(spkr_list_m);

spkr_list_f = fopen([stimdir 'selected_female_spkrs.txt']);
F = textscan(spkr_list_f,'%s');
fs_quiltpkrs = F{1};
fclose(spkr_list_f);

allspkrs = [mspkrs fs_quiltpkrs];
chLabel={'L','R'};

allgenders = {'male', 'female'};
fs_final = 16000;

for g = 1:2
    spkrs = allspkrs(:,g);
    gender = allgenders{g}
    for i = 1:length(spkrs)
        % read wav file
        name = spkrs{i}(1:end-4);
        wavname = sprintf('%squilts/%s/%s_%ds.wav', stimdir, gender, name, out_length_in_sec)
%         [s,fs_quilt,bits]=wavread(wavname);
        [s, fs_quilt] = audioread(wavname);
        
        for indCh=1:size(chLabel,2) % loop for channels
            [h,fs_filt]=load_filter(['EQF_219',chLabel{indCh},'.bin']);
            if fs_filt~=fs_quilt
                disp('Need to resample signal!'); 
                % resample to 44100 for filtering
                % Annoying that the sensimetrics function doesn't allow for variable
                % sample rates...
                s = resample(s,fs_filt,fs_quilt);
                fs_quilt = fs_filt;
            end;
            s_filt(:,indCh)=conv(h,s);
            s_filt(:,indCh) = s_filt(:,indCh)/(max(abs(s_filt(:,indCh))));
    %         sound(s_filt)
        end % loop for channels

        % Resample to 16kHz because that was the original sample rate and will
        % be easier for psychopy
        s_filt = resample(s_filt,fs_final,fs_filt);
        
        % Take exactly 59.5 seconds
        s_filt = s_filt(1:59.5*fs_final);
        
        % save s_filt to fmri_exp/stimuli directory
        [pathstr,base,ext] = fileparts(wavname);
        dirtosave = ['stimuli/' lang filesep gender filesep];
        if ~isdir(dirtosave)
            mkdir(dirtosave)
        end
%         wavwrite(s_filt, fs_final, [dirtosave base '_filtered.wav'])
        audiowrite([dirtosave base '_filtered.wav'], s_filt, fs_final)
        clear s_filt
    end
end