% make_planewave_sample.m
% Builds the bundled plane-wave smoke-test sample (data/sample/planewave)
% from the full CIRS040GSE download: one wire-target frame (low_attenuation,
% frame 6) cut down to the 3 transmit angles around broadside, with the
% header reduced to the same 3 angles. Sample count and element count are
% unchanged, so the file layout matches the full dataset.
%
% Input:  PLANEWAVE_DATA_PATH (folder with low_attenuation/USHEADER_*.mat and USDATA_*.mat)
% Output: RESULTS_DIR/planewave_sample/low_attenuation/ (copy to data/sample/planewave/)

dataRoot = getenv('PLANEWAVE_DATA_PATH');
if isempty(dataRoot)
    error('PLANEWAVE_DATA_PATH environment variable not set.');
end
resultsDir = getenv('RESULTS_DIR');
if isempty(resultsDir)
    error('RESULTS_DIR environment variable not set.');
end
frameIdx = 6;   % wire target
inDir  = fullfile(dataRoot, 'low_attenuation');
outDir = fullfile(resultsDir, 'planewave_sample', 'low_attenuation');
if ~exist(outDir, 'dir'), mkdir(outDir); end

hf = dir(fullfile(inDir, 'USHEADER_*.mat'));
H  = load(fullfile(hf(1).folder, hf(1).name));
USHEADER = H.USHEADER;
[~, b] = min(abs(USHEADER.xmitAngles));
sel = b-1:b+1;
USHEADER.xmitAngles       = USHEADER.xmitAngles(sel);
USHEADER.xmitDelay        = USHEADER.xmitDelay(sel, :);
USHEADER.xmitFocus        = USHEADER.xmitFocus(sel);
USHEADER.xmitApodFunction = USHEADER.xmitApodFunction(sel, :);
save(fullfile(outDir, hf(1).name), 'USHEADER', '-v7');

df = dir(fullfile(inDir, 'USDATA_*.mat'));
D  = load(fullfile(df(frameIdx).folder, df(frameIdx).name));
USDATA = D.USDATA(:, :, :, sel);   % int16 [samples elements 1 angles]
save(fullfile(outDir, df(frameIdx).name), 'USDATA', '-v7.3');   % HDF5, as read by h5py in the pipeline
fprintf('Sample written to %s: %s, angles %s deg, USDATA %s\n', outDir, df(frameIdx).name, ...
    mat2str(USHEADER.xmitAngles', 3), mat2str(size(USDATA)));
