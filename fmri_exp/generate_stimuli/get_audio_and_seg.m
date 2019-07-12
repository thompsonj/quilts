function [source_s, sr, all_phn_start, all_phn_end, all_phns] = get_audio_and_seg(name, utts, segdir, wavdir)

source_s = [];
all_phn_start = [];
all_phn_end = [];
all_phns = [];
offset = 0;
for i=1:length(utts)
    utt = num2str(utts(i));
    
    % load segmentation info
%     display('Loading segmentation...')
    segfname = [segdir filesep name filesep name '_' utt];
    load(segfname)
    if ~isempty(phn_end)
        % load audio
%         display('loading audio...')
        wavname = [wavdir, name, filesep, name, '_', utt,'.wav'];
        [s,sr] = audioread(wavname);

        source_s = cat(1,source_s, s);
        all_phn_start = [all_phn_start (phn_start + offset)];
        all_phn_end = [all_phn_end (phn_end + offset)];
        if size(phns, 2) == 2
            phns(:,3) = ' ';
        end
        all_phns = cat(1, all_phns, phns);
        offset = offset + phn_end(end); % Before I was adding 1 here but I think that was wrong and led to index out of bounds errors
    else
        display([name '_' utt ' phn bounds array is empty'])
    end
    
end
