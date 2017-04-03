caffe.reset_all();
clear; close all;

folder = '';
model = [folder '../DRLSR_clip_mat_inception_v2.prototxt'];
weights = [folder '../DRLSR_inception_v2_iter_2000000.caffemodel'];
savepath = ['model\' 'DRLSR_inception_v2_iter_2000000.mat'];

net = caffe.Net(model,weights,'test');

[in_weights_conv1,in_biases_conv1]= getWeight(net,'inception_conv1_3');
[in_weights_conv2,in_biases_conv2]= getWeight(net,'inception_conv1_5');
[in_weights_conv3,in_biases_conv3]= getWeight(net,'inception_conv1_9');
[weights_conv2,biases_conv2]= getWeight(net,'conv2');
[weights_conv22,biases_conv22]= getWeight(net,'conv22');
[weights_conv23,biases_conv23]= getWeight(net,'conv23');
[in3_weights_conv1,in3_biases_conv1]= getWeight(net,'inception_conv3_3');
[in3_weights_conv2,in3_biases_conv2]= getWeight(net,'inception_conv3_5');
[in3_weights_conv3,in3_biases_conv3]= getWeight(net,'inception_conv3_9');
[weights_conv3,biases_conv3]= getWeight(net,'conv3');
    
save(savepath,'in_weights_conv1','in_biases_conv1'...
    ,'in_weights_conv2','in_biases_conv2','in_weights_conv3','in_biases_conv3'...
    ,'weights_conv2','biases_conv2','weights_conv22','biases_conv22','weights_conv23',...
    'biases_conv23','in3_weights_conv1','in3_biases_conv1'...
    ,'in3_weights_conv2','in3_biases_conv2','in3_weights_conv3','in3_biases_conv3','weights_conv3','biases_conv3');
