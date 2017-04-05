close all;
clear all;

%% set parameters
testfolder = 'Set5\';
resultfolder=['result\' testfolder];
up_scale = 3;
model = 'model\DRLSR_inception_v2_iter_2000000.mat';
modelFSRCNNs = 'model\FSRCNN-s\x3.mat';
modelFSRCNN = 'model\FSRCNN\x3.mat';
modelSRCNN = 'model\SRCNN\x3.mat';

filepaths = dir(fullfile(testfolder,'*.bmp'));
tt_bic = zeros(length(filepaths),1);
tt_drlsr = zeros(length(filepaths),1);
% tt_fsrcnn = zeros(length(filepaths),1);
% tt_fsrcnns = zeros(length(filepaths),1);
% tt_srcnn = zeros(length(filepaths),1);

psnr_bic = zeros(length(filepaths),1);
psnr_drlsr = zeros(length(filepaths),1);
% psnr_fsrcnn = zeros(length(filepaths),1);
% psnr_fsrcnns = zeros(length(filepaths),1);
% psnr_srcnn = zeros(length(filepaths),1);

ssim_bic = zeros(length(filepaths),1);
ssim_drlsr = zeros(length(filepaths),1);
% ssim_fsrcnn = zeros(length(filepaths),1);
% ssim_fsrcnns = zeros(length(filepaths),1);
% ssim_srcnn = zeros(length(filepaths),1);

fsim_bic = zeros(length(filepaths),1);
fsim_drlsr = zeros(length(filepaths),1);
% fsim_fsrcnn = zeros(length(filepaths),1);
% fsim_fsrcnns = zeros(length(filepaths),1);
% fsim_srcnn = zeros(length(filepaths),1);

for i = 1 : length(filepaths)

    %% read ground truth image
    [add,imname,type] = fileparts(filepaths(i).name);
    im = imread([testfolder imname type]);

    %% work on illuminance only
    if size(im,3) > 1
        im_ycbcr = rgb2ycbcr(im);
        im = im_ycbcr(:, :, 1);
    end
    im_gnd = modcrop(im, up_scale);
    im_gnd = single(im_gnd)/255;
    im_l = imresize(im_gnd, 1/up_scale, 'bicubic');

    %% bicubic interpolation
    tic
    im_b = imresize(im_l, up_scale, 'bicubic');
    t_bic=toc;

    %% DRLSR
    tic
    %[im_drl,layerout] = DRLSR3x_inception_v3(model, im_b);
    [im_drl,layerout] = DRLSR3x_inception(model, im_b);
    %[im_drl,layerout] = DRLSR3x(model, im_b);
    t_drlsr=toc;

%      %% FSRCNNs
%      tic
%      im_fsrs = FSRCNN(modelFSRCNNs, im_l, up_scale);
%      t_fsrs=toc;
%
%       %% FSRCNN
%      tic
%      im_fsr = FSRCNN(modelFSRCNN, im_l, up_scale);
%      t_fsr=toc;
%
%       %% SRCNN
%      tic
%      im_srcnn = SRCNN(modelSRCNN, im_b);
%      t_srcnn=toc;

    %% remove border
    im_drl = shave(uint8(im_drl * 255), [up_scale, up_scale]);
%     im_fsr = shave_x3(uint8(im_fsr * 255), [up_scale, up_scale]);
%     im_fsrs = shave_x3(uint8(im_fsrs * 255), [up_scale, up_scale]);
%     im_srcnn = shave(uint8(im_srcnn * 255), [up_scale, up_scale]);
    im_gnd = shave(uint8(im_gnd * 255), [up_scale, up_scale]);
    im_b = shave(uint8(im_b * 255), [up_scale, up_scale]);


    %% compute time
    tt_bic(i) = t_bic;
    tt_drlsr(i) = t_drlsr+t_bic;
%     tt_fsrcnn(i) = t_fsr;
%     tt_fsrcnns(i) = t_fsrs;
%     tt_srcnn(i) = t_srcnn+t_bic;

    %% compute PSNR
    psnr_bic(i) = compute_psnr(im_gnd,im_b);
    psnr_drlsr(i) = compute_psnr(im_gnd,im_drl);
%     psnr_fsrcnn(i) = compute_psnr(im_gnd,im_fsr);
%     psnr_fsrcnns(i) = compute_psnr(im_gnd,im_fsrs);
%     psnr_srcnn(i) = compute_psnr(im_gnd,im_srcnn);

    %% compute FSIM
    fsim_bic(i) = FeatureSIM(im_gnd,im_b);
    fsim_drlsr(i) = FeatureSIM(im_gnd,im_drl);
%     fsim_fsrcnn(i) = FeatureSIM(im_gnd,im_fsr);
%     fsim_fsrcnns(i) = FeatureSIM(im_gnd,im_fsrs);
%     fsim_srcnn(i) = FeatureSIM(im_gnd,im_srcnn);

     %% compute SSIM
     ssim_bic(i) = ssim_index(im_gnd,im_b);
    ssim_drlsr(i) = ssim_index(im_gnd,im_drl);
%     ssim_fsrcnn(i) = ssim_index(im_gnd,im_fsr);
%     ssim_fsrcnns(i) = ssim_index(im_gnd,im_fsrs);
%     ssim_srcnn(i) = ssim_index(im_gnd,im_srcnn);

     [aa,bb,cc]=size(im_drl);

    im_h_drl=imresize(im_ycbcr,[aa bb],'bicubic');
    im_h_drl(:,:,1)=im_drl;
    im_h_drl=ycbcr2rgb(im_h_drl);

%     im_h_fsr=imresize(im_ycbcr,[aa bb],'bicubic');
%     im_h_fsr(:,:,1)=im_fsr;
%     im_h_fsr=ycbcr2rgb(im_h_fsr);
%
%     im_h_fsrs=imresize(im_ycbcr,[aa bb],'bicubic');
%     im_h_fsrs(:,:,1)=im_fsrs;
%     im_h_fsrs=ycbcr2rgb(im_h_fsrs);
%
%     im_h_srcnn=imresize(im_ycbcr,[aa bb],'bicubic');
%     im_h_srcnn(:,:,1)=im_srcnn;
%     im_h_srcnn=ycbcr2rgb(im_h_srcnn);
%
    im_b_final=imresize(im_ycbcr,[aa bb],'bicubic');
    im_b_final(:,:,1)=im_b;
    im_b_final=ycbcr2rgb(im_b_final);

    %% save results
    imwrite(im_b_final, [resultfolder imname '_bic.bmp']);
    imwrite(im_h_drl, [resultfolder imname '_DRLSR.bmp']);
%     imwrite(im_h_fsr, [resultfolder imname '_FSRCNN.bmp']);
%     imwrite(im_h_fsrs, [resultfolder imname '_FSRCNNs.bmp']);
%     imwrite(im_h_srcnn, [resultfolder imname '_SRCNN.bmp']);

end

fprintf('Bicubic: %f , %f , %f , %f \n', mean(psnr_bic), mean(ssim_bic), mean(fsim_bic), mean(tt_bic));
fprintf('DRLSR: %f , %f , %f , %f \n', mean(psnr_drlsr), mean(ssim_drlsr), mean(fsim_drlsr), mean(tt_drlsr));
% fprintf('FSRCNN: %f , %f , %f , %f \n', mean(psnr_fsrcnn), mean(ssim_fsrcnn), mean(fsim_fsrcnn), mean(tt_fsrcnn));
% fprintf('FSRCNNs: %f , %f , %f , %f \n', mean(psnr_fsrcnns), mean(ssim_fsrcnns), mean(fsim_fsrcnns), mean(tt_fsrcnns));
% fprintf('SRCNN: %f , %f , %f , %f \n', mean(psnr_srcnn), mean(ssim_srcnn), mean(fsim_srcnn), mean(tt_srcnn));
