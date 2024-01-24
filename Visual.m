% Example code for hyperspectral reconstruction
clc;close all;
clear;

disp('Loading ground truth image');

% Load ground truth HS image
load('bgu_0403-1511.mat'); % Provides 'rad' and 'bands'
rad = permute(rad, [2, 1 , 3]);
% Reconstrucy HS information from the camera response image
disp('Loading Reconstruction HS image');
% load('bgu_0403-1511_result.mat'); % Provides 'rad' and 'bands'
load('bgu_0403-1511_result.mat'); % Provides 'rad' and 'bands'

rec_hs = permute(result,[2,3,1]);
% rec_hs = permute(result,[3,2,1]);
fprintf('Done\n');


% Partial visualization of results:
figure(1);
visualized_bands=[1,2,3];

% visualized_bands=[15,23,30];
for i=1:3
    subplot(3,3,i);
    imagesc(rad(:,:,visualized_bands(i)));
    title([num2str(visualized_bands(i)) 'th channal Ground Truth']);
%     colormap bone;
    axis image;axis off;
    
    subplot(3,3,3+i);
    imagesc(rec_hs(:,:,visualized_bands(i)));
    title([num2str(visualized_bands(i)) 'th channal Reconstructed']);
%     colormap bone;
    axis image; axis off;
    
    subplot(3,3,6+i);
    imagesc( ((rad(:,:,visualized_bands(i))-rec_hs(:,:,visualized_bands(i)))./max(rad(:)))*255, [-20 20]  );
    title('Error map');
    axis image; axis off;
end