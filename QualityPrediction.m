%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%    Quality Prediction with Extracted Features    %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;clc

%%
%==========================================================================
%                        Data Preparetion
%==========================================================================
load('data/ESPLLIVEHDRinfo'); % im_dir, im_ids, im_lists, im_names, index, ref_ids, subjective_scores, subjective_scoresSTD
%index = index(:,1:ceil(size(index,2)*0.8));

N = length(im_lists);% 1811: 
numIdx = size(index, 1);
train_ratio = 0.8;

resuTMO = zeros(numIdx,1);
resuMEF = zeros(numIdx,1);
resuEff = zeros(numIdx,1); 

resu = zeros(numIdx,1);
resuPea = zeros(numIdx,1);
resuKen = zeros(numIdx,1);
resuRMSE = zeros(numIdx,1);
resuOR = zeros(numIdx,1);

load('data/mSmLfeatures');
layer_names = {'res2a','res2b','res2c','res3a','res3b','res3c','res3d',...
    'res4a','res4b','res4c','res4d','res4e','res4f'};

chosen_layer = [1 9 13]; % [13 9 1]; % 
nlc = 15; % 

%%
%==========================================================================
%         Calculate Aggregated Features
%==========================================================================
feature = cell2mat(arrayfun(@(i)features{chosen_layer(i)}, ...
    1:length(chosen_layer),'UniformOutput',false));

%%    
%==========================================================================
%          Calculate 'beta' of PLSR Used for Regression
%==========================================================================
predict_statistics = cell(size(index,1),1);

for t = 1 : numIdx

fprintf('the %d-th iteration\n',t); 

%split the data of three types seperately
train_im_index = index(t,1:ceil(train_ratio*size(index,2)));    
train_im_index = cell2mat(arrayfun(@(i)find(ref_ids==train_im_index(i))',...
    1:length(train_im_index),'UniformOutput',false));
test_im_index = index(t,1+ceil(train_ratio*size(index,2)):size(index,2));
test_im_index = cell2mat(arrayfun(@(i)find(ref_ids==test_im_index(i))',...
    1:length(test_im_index),'UniformOutput',false));
train_labels = subjective_scores(train_im_index);
test_labels = subjective_scores(test_im_index);

tmo_index = cell2mat(arrayfun(@(i)find(ref_types==1)',1:length(ref_types),'UniformOutput',false));
mef_index = cell2mat(arrayfun(@(i)find(ref_types==2)',1:length(ref_types),'UniformOutput',false));
eff_index = cell2mat(arrayfun(@(i)find(ref_types==3)',1:length(ref_types),'UniformOutput',false));

test_tmo_index = intersect(test_im_index, tmo_index);
test_mef_index = intersect(test_im_index, mef_index);
test_eff_index = intersect(test_im_index, eff_index);

tmo_labels = subjective_scores(test_tmo_index);
mef_labels = subjective_scores(test_mef_index);
eff_labels = subjective_scores(test_eff_index);

feature_size = size(feature,2);

%PLSR
p = nlc; % number of components;
[~,~,~,~,betaR] = plsregress(feature(train_im_index(:),:),repmat(train_labels,1,1),p);


%%    
%==========================================================================
%                 Prediciton and Results
%==========================================================================

predict_statistics_tmo = zeros(numel(tmo_labels),1);
predict_statistics_mef = zeros(numel(mef_labels),1);
predict_statistics_eff = zeros(numel(eff_labels),1);
%=============================TEST=========================================

predict_statistics{t} = [ones(length(test_im_index(:)),1) feature(test_im_index, :)]*betaR;

predict_statistics_tmo = [ones(length(test_tmo_index(:)),1) feature(test_tmo_index, :)]*betaR;
predict_statistics_mef = [ones(length(test_mef_index(:)),1) feature(test_mef_index, :)]*betaR;
predict_statistics_eff = [ones(length(test_eff_index(:)),1) feature(test_eff_index, :)]*betaR;

%============================RESULTS=======================================
objective_scores = mean(predict_statistics{t},2);
objective_scoresTMO = mean(predict_statistics_tmo,2);
objective_scoresMEF = mean(predict_statistics_mef,2);
objective_scoresEff = mean(predict_statistics_eff,2);

resu(t) = corr(objective_scores, test_labels, 'type', 'Spearman');
resuPea(t) = corr(objective_scores, test_labels);
resuKen(t) = corr(objective_scores, test_labels, 'type', 'Kendall');
resuRMSE(t) = sqrt(mean((objective_scores-test_labels).^2));
resuOR(t) = mean(abs(objective_scores-test_labels)>2*subjective_scoresSTD(test_im_index));
med_SROCC = median(resu(1:t));
med_PLCC = median(resuPea(1:t));

resuTMO(t) = corr(objective_scoresTMO, tmo_labels, 'type', 'Spearman');
resuMEF(t) = corr(objective_scoresMEF, mef_labels, 'type', 'Spearman');
resuEff(t) = corr(objective_scoresEff, eff_labels, 'type', 'Spearman');
tmo_sr = median(resuTMO(1:t));
mef_sr = median(resuMEF(1:t));
eff_sr = median(resuEff(1:t));

SRresult = [med_SROCC tmo_sr mef_sr eff_sr]
end
fid=fopen('srocc.txt','a');
fprintf(fid, '%d  ',  nlc);
fprintf(fid, '%f %f %f %f\n',  SRresult(:));
fclose(fid);
