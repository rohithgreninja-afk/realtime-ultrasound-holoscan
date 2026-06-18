% Phase4_OASBUD_Export.m
% Export 20 malignant + 20 benign OASBUD patients as PNG training images
% Uses A-line reconstruction (hilbert + power law) -- same as pipeline
% Applies heavy augmentation to each patient to generate multiple images
% Output: C:\Users\rohit\Downloads\OASBUD_PNG\malignant\ and \benign\

clearvars; clc;

%% Settings
dataPath   = 'C:\Users\rohit\Downloads\Real Time Image Processing Project\OASBUD.mat';
outRoot    = 'C:\Users\rohit\Downloads\OASBUD_PNG';
fs         = 40e6;
c          = 1540;
probeWidth = 38e-3;
gamma      = 0.3;
imgSize    = [224 224];
augPerPat  = 20;   % generate 20 augmented versions per patient

% Training patient indices (from class distribution check)
train_mal = [1 2 3 5 7 8 9 12 15 16 17 19 21 23 24 25 26 27 30 32];
train_ben = [4 6 10 11 13 14 18 20 22 28 29 31 35 60 61 62 63 64 65 66];

%% Create output folders
mkdir(fullfile(outRoot, 'malignant'));
mkdir(fullfile(outRoot, 'benign'));
fprintf('Output folders created at %s\n', outRoot);

%% Load dataset
data = load(dataPath);
fprintf('OASBUD loaded: %d patients\n', numel(data.data));

%% Export function
function bmode = reconstruct(rf, fs, c, gamma)
    env   = abs(hilbert(rf));
    norm  = env / (max(env(:)) + 1e-12);
    bmode = norm .^ gamma;
end

%% Process training patients
allIdx    = {train_mal, train_ben};
allLabels = {'malignant', 'benign'};

for cls = 1:2
    patList = allIdx{cls};
    label   = allLabels{cls};
    outDir  = fullfile(outRoot, label);
    count   = 0;

    for p = 1:numel(patList)
        patIdx = patList(p);
        rf     = double(data.data(patIdx).rf1);

        % Base reconstruction
        bmode = reconstruct(rf, fs, c, gamma);

        % Resize to 224x224
        bmode_img = imresize(bmode, imgSize);

        % Save base image
        count = count + 1;
        fname = sprintf('oasbud_%s_p%03d_base.png', label, patIdx);
        imwrite(uint8(bmode_img * 255), fullfile(outDir, fname));

        % Also save rf2 view as second base image
        rf2       = double(data.data(patIdx).rf2);
        bmode2    = reconstruct(rf2, fs, c, gamma);
        bmode2_img = imresize(bmode2, imgSize);
        count = count + 1;
        fname2 = sprintf('oasbud_%s_p%03d_view2.png', label, patIdx);
        imwrite(uint8(bmode2_img * 255), fullfile(outDir, fname2));

        % Heavy augmentation -- generate augPerPat versions
        for aug = 1:augPerPat
            img = bmode_img;

            % 1: Random horizontal flip
            if rand > 0.5
                img = fliplr(img);
            end

            % 2: Random rotation -20 to +20 degrees
            angle = (rand - 0.5) * 40;
            img   = imrotate(img, angle, 'bilinear', 'crop');

            % 3: Random brightness shift -0.15 to +0.15
            img = img + (rand - 0.5) * 0.3;
            img = max(0, min(1, img));

            % 4: Random contrast scaling 0.7 to 1.3
            mu  = mean(img(:));
            img = (img - mu) * (0.7 + rand * 0.6) + mu;
            img = max(0, min(1, img));

            % 5: Random Gaussian noise sigma 0 to 0.04
            img = img + randn(size(img)) * rand * 0.04;
            img = max(0, min(1, img));

            % 6: Random zoom crop 85-100% then resize back
            zoomFactor = 0.85 + rand * 0.15;
            cropSize   = round(imgSize * zoomFactor);
            startR = randi(imgSize(1) - cropSize(1) + 1);
            startC = randi(imgSize(2) - cropSize(2) + 1);
            img    = img(startR:startR+cropSize(1)-1, startC:startC+cropSize(2)-1);
            img    = imresize(img, imgSize);

            % 7: Random elastic-like deformation via small grid warp
            [X, Y] = meshgrid(1:imgSize(2), 1:imgSize(1));
            strength = 4;
            dx = strength * (rand(imgSize) - 0.5);
            dy = strength * (rand(imgSize) - 0.5);
            dx = imgaussfilt(dx, 8);
            dy = imgaussfilt(dy, 8);
            Xw = max(1, min(imgSize(2), X + dx));
            Yw = max(1, min(imgSize(1), Y + dy));
            img = interp2(X, Y, img, Xw, Yw, 'linear', 0);

            count = count + 1;
            fname = sprintf('oasbud_%s_p%03d_aug%03d.png', label, patIdx, aug);
            imwrite(uint8(img * 255), fullfile(outDir, fname));
        end

        fprintf('Patient %d (%s): %d images generated\n', patIdx, label, augPerPat + 2);
    end

    fprintf('\n%s total images: %d\n\n', label, count);
end

fprintf('Export complete.\n');
fprintf('Malignant training images: %d\n', numel(dir(fullfile(outRoot,'malignant','*.png'))));
fprintf('Benign training images: %d\n',    numel(dir(fullfile(outRoot,'benign','*.png'))));