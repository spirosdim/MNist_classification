% load training set and testing set
train_set = loadMNISTImages('train-images.idx3-ubyte')';
train_label = loadMNISTLabels('train-labels.idx1-ubyte');
test_set = loadMNISTImages('t10k-images.idx3-ubyte')';
test_label = loadMNISTLabels('t10k-labels.idx1-ubyte');

tic; %start  timer 
%Train C-SVM model (multi-class classification)
%kernel RBF : exp(-gamma*|u-v|^2)
% -g gamma 
model = svmtrain(train_label ,train_set ,'-s 0 -t 2 -g 0.5 ' );

  %Test model on the train set  (not required)
% fprintf('**Training prediction:\n');
% predict_train = svmpredict(train_label ,train_set , model);


%Test model on the test set
fprintf('** Testing prediction:\n');
predict_test = svmpredict(test_label ,test_set , model);
toc; %end  timer 