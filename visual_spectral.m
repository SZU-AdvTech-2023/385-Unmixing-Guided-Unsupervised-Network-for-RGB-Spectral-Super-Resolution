% Example code for hyperspectral reconstruction
clc;close all;
clear;

disp('Loading ground truth image');

% Load ground truth HS image
load('HoustonU_200.mat'); % Provides 'rad' and 'bands'
rad = permute(rad, [2, 3 , 1]);
% Reconstrucy HS information from the camera response image
disp('Loading Reconstruction HS image');
% load('bgu_0403-1511_result.mat'); % Provides 'rad' and 'bands'
% load('bgu_0403-1511_result.mat'); % Provides 'rad' and 'bands'
% rec_hs1 = permute(result,[2,3,1]);
% rec_hs1 = result;
% load('bgu_0403-1511_result2.mat'); % Provides 'rad' and 'bands'
% rec_hs2 = permute(result,[2,3,1]);
% rec_hs = permute(result,[3,2,1]);

fprintf('Done\n');

func_hyperImshow(rad,[15,30,39]);

% [rad] = scale_new(rad);
% [rec_hs1] = scale_new(rec_hs1);
% [rec_hs2] = scale_new(rec_hs2);
% 
% point = rad(300:300, 300:300,:);
% [no_lines, no_rows, no_bands] = size(point);
% point = reshape(point, (no_lines * no_rows), no_bands)';
% 
% 
% point1 = rec_hs1(300:300, 300:300,:);
% [no_lines, no_rows, no_bands] = size(point1);
% point1 = reshape(point1, (no_lines * no_rows), no_bands)';
% 
% point2 = rec_hs2(300:300, 300:300,:);
% [no_lines, no_rows, no_bands] = size(point2);
% point2 = reshape(point2, (no_lines * no_rows), no_bands)';
% 
% figure
% plot(point,'r', 'LineWidth',1.5)
% hold on
% 
% plot(point1,'b', 'LineWidth',1.5)
% 
% plot(point2,'g', 'LineWidth',1.5)
% 
% xlabel('Band Number','fontname','Cambria')
% % xlabel('Wavelength \it(nm)','fontname','Cambria')
% ylabel('Normalized Reflectance','fontname','Cambria')
% set(gca,'FontName','Cambria','FontSize',18)
% set(gca,'linewidth',1.2)
% set(gca,'XTick',[1:5:31]) %x轴范围1-6，间隔1
% legend('Groud Truth', 'UnGUN(New)','UnGUN');   %右上角标注





