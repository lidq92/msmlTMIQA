%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%    Extracted features  from multiple layers   %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;clc

caffe_path = '/home/ldq/caffe/matlab/'; % point to the caffe path
addpath(genpath(caffe_path)); 

load('data/ESPLLIVEHDRinfo'); % im_dir, im_ids, im_lists, im_names, index, ref_ids, subjective_scores, subjective_scoresSTD
layer_names = {'res2a','res2b','res2c','res3a','res3b','res3c','res3d',...
    'res4a','res4b','res4c','res4d','res4e','res4f'};

features = SFAfeature(im_lists, layer_names);

save('data/mSmLfeatures', '-v7.3', 'features');


