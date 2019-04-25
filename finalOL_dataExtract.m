% Written by Eric Sung
% Optimization and Learning
% Final project
% finalOL_dataExtract.m
% Custom written code to process MIT
%
% Hypothesis: There are inherent features within a single ECG beat that can
% distinguish between normal beats and PVC beats

%% (1) Downloading the Database
% We first need to download the data, and adjust the format
% Read in our data, find the beats, and find the corresponding nnotations

% Total number of beats happens to be 112646
% For our purposes, there are about 8000 premature ventricular
% contractions, we'll extract them and also a set of normal beats to go
% to match the two class sizes

ALL_files = [100:109, 111:119, 121:124, 200:203,...      % MIT database names
             205, 207:210, 212:215, 217 219, 220:223 228 230:234];

ALL_beats = {};         % Each entry will be the time series for a given beat
label_beats = {};       % Provide the name of each beat (ex: PtID_Beat##)
ann_beats = {};         % Provides the given annotation provided by a physician
count = 1;              % Counter variable
S = rng();              % Seed the random number negerator for reproducibility

% Now, go through each patient, extract the ECG beats, and store them
for i = ALL_files
    [sig, Fs, tm] = rdsamp(['mitdb/' num2str(i)], 1);           % Read in the time series
    [RR,tms]=ann2rr(['mitdb/' num2str(i)],'atr');               % Read in the R-R intervals
    [ann,anntype,subtype,chan,num,comments]=...                 % Read in the annotations
               rdann(['mitdb/' num2str(i)],'atr');
    shift = min([ann(1)-1,100]);                % Annotations occur in the middle of a beat
                                                % Thus incorporate the
                                                % previous 100 points as
                                                % well
                                                                                                     
    TMS = [ann; length(sig)+1]-shift;           % Shift frame to include the whole beat 
    disp('----------------------------------------------')
    disp(['Processing dataset ' num2str(i) ])
    % Go through each beat
    for j = 1:(size(ann,1))
       disp(['Dataset ' num2str(i) ' Trace ' num2str(j) ' of ' num2str(size(ann,1)) '...'])
       traces_i{j} = sig(TMS(j):TMS(j+1)-1);
       ALL_beats{count} = traces_i{j};
       label_beats{count} = ['Beat_' num2str(j) '_ECG' num2str(i) ];
       ann_beats{count} = anntype(j);
       count = count+1;
    end
    disp(['Finished dataset ' num2str(i) ])
    disp('----------------------------------------------')
end

%% (2) Extracting labels
% Now that we have all of our beats, figure out which beats are normal, and
% which beats correspond to premature ventricular contractions (PVCs), the
% label of interest

% Because there are a lot of normal beats compared to PVCs, we will
% downsample our data to match that of the PVCs
N_idx = find(strcmp(ann_beats,'N')); % Find location of normal beats ('N' is the annotation for normal)
rand_samp = randperm(length(N_idx)); % Randomly select normal beats
N_samp = N_idx(rand_samp(1:8000));   % Downsample to 8000 because thats roughly how many PVCs there are
N_beats = ALL_beats(N_samp);         % Extract the normal beat time series
N_ann = ann_beats(N_samp);           % Extract the normal beat annotations
N_labels = label_beats(N_samp);      % Extract the IDs of each normal beat

% Do the same for the PVCs
V_idx = find(strcmp(ann_beats,'V')...
            +strcmp(ann_beats,'F')); % Both 'V' and 'F' refer to PVCs
V_beats = ALL_beats(V_idx);          % Extract the PVC beat time series
V_ann = ann_beats(V_idx);            % Extract the PVC annotations
V_labels = label_beats(V_idx);       % Extract the IDs of each PVC

%% (3) Computing the features
% The HCTSA package requires a specific format to perform the computations.
% Specifically, it asks for 3 fields: timeSeriesData, labels, and keywords
% to be put into a .mat file that will be processed.

timeSeriesData = [N_beats V_beats]';
labels = [N_labels V_labels]';
keywords = [N_ann V_ann]';

% Save these variables out to INP_test.mat:
save('INP_test.mat','timeSeriesData','labels','keywords');

% Initialize a new hctsa analysis using these data and the default feature library:
TS_init('INP_test.mat');

% Lastly, we run the very time-consuming command: TS_compute to compute our
% features. For computational purposes, I chose to just use the first 800
% features out of 7642 possible features.
operations = 1:800;
TS_compute(false,[],operations)

%% (4) Post-Processing into mat-file

load('HCTSA.mat')             % load in the .mat file, variable of interest is TS_DataMat
X = TS_DataMat(:,operations); % Create a features matrix X
Y = strcmp(keywords,'N');     % For now, our label space will be 1 for normal, 0 for PVCs
save('PVCdata.mat','X','Y')   % Save Data
