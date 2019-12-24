addpath(genpath('/home/Pedestrian/Documents/PycharmProjects/Incremental_Learning_for_Detection/edges'));
addpath(genpath('/home/Pedestrian/Documents/PycharmProjects/Incremental_Learning_for_Detection/piotr_toolbox/toolbox/'));

voc_dir = '/home/Pedestrian/Documents/PycharmProjects/Incremental_Learning_for_Detection/datasets/';
proposals_dir = [voc_dir 'EdgeBoxesProposals/'];
image_dir = [voc_dir 'Images/'];

content = dir([image_dir '*.jpg']);

model = load('/home/Pedestrian/Documents/PycharmProjects/Incremental_Learning_for_Detection/edges/models/forest/modelBsds');
model = model.model;
model.opts.multiscale = 1;
model.opts.sharpen = 2;
model.opts.nThreads = 4;

opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 2000;  % max number of boxes to detect
opts.minBoxArea = 50;  % min box area 
% opts.minBoxArea = 1000;  % min box area 

for i = 1 : length(content)
  if rem(i, 10) == 0
      disp(i)
  end
  img_name = content(i).name;
  I = imread([image_dir img_name]);
  if length(size(I)) == 2
      I = cat(3, I, I, I);
  end
  % disp(size(I))
  bbs = edgeBoxes(I, model, opts);
  save([proposals_dir img_name(1:end-3) 'mat'], 'bbs');
end