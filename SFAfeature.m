function features = SFAfeature(im_lists, layer_names)

% Initialize the DCNN
model = [fileparts(which(mfilename)) '/models/ResNet-50-deploy-without-fc.prototxt']; %
weights = [fileparts(which(mfilename)) '/models/ResNet-50-model.caffemodel']; %
caffe.set_mode_gpu();
caffe.set_device(0); % set gpu id
net = caffe.Net(model, weights, 'test');

%============================================================================
%                           PREPARE OUTPUTS
%============================================================================
nscale = 2; % 
features = cell(length(layer_names), 1);
for i = 1:length(layer_names)
    l = net.blobs(layer_names{i}).shape; 
    l = l(3);
    eval(['feature{' char(string(i)) '} = zeros(length(im_lists),2*nscale*l);']);
end

%============================================================================
%                           EXTRACT FEATURES
%============================================================================
for k = 1:length(im_lists)
    fprintf(['Extracting the feature of the ' num2str(k) 'th image ...\n']);

    % Multi-Scale representation
    im_data = cell(nscale,1);
    im_data{1} = imread(im_lists{k});
    for sub = 2:nscale
        im_data{sub,1} = downsampling(im_data{sub-1,1});
    end

    % Extract features from all the layers we need
    imfeatures = DCNN(im_data, net, layer_names, nscale);
    for i = 1:length(layer_names)
        features{i}(k,:) = imfeatures{i};
    end

end
caffe.reset_all(); 

function features = DCNN(im_data, net, layer_names, nscale)
% Extract multi-layer and multi-scale features from ResNet-50

% Preprocessing
for sub = 1:nscale
    im_data{sub} = im_data{sub}(:,:,[3 2 1]); % RGB2BGR
    im_data{sub} = permute(im_data{sub},[2 1 3]); % HWC2WHC
    im_data{sub} = single(im_data{sub});

    im_data{sub}(:,:,1) = im_data{sub}(:,:,1) - 104;
    im_data{sub}(:,:,2) = im_data{sub}(:,:,2) - 117;
    im_data{sub}(:,:,3) = im_data{sub}(:,:,3) - 123;
end

% Extract features
features = cell(length(layer_names), 1);
for i = 1:length(layer_names)
    l = net.blobs(layer_names{i}).shape;
    l = l(3);
    eval(['features{' char(string(i)) '} = zeros(1, 2*l*nscale);']);
end

for sub = 1:length(im_data)
    curIm = im_data(sub);
    sz = size(curIm{1});
    sz = [sz 1];
    net.blobs('data').reshape(sz);
    net.forward(curIm);
    for i = 1:length(layer_names)
        layer_name = layer_names{i};
        l = net.blobs(layer_name).shape;
        l = l(3);

        mt = mean(reshape(net.blobs(layer_name).get_data(),[],l,1));
        st = std(reshape(net.blobs(layer_name).get_data(),[],l,1));        
        eval(['features{' char(string(i)) '}((sub-1)*2*l+1:sub*2*l) = [mt,st];']);
    end
end

function submat = downsampling(src)

w = size(src,1);
l = size(src,2);
c = size(src,3);

submat = src(1:2:end,1:2:end,:);
