% There are three main steps to the training process:
%   1. Prototype selection through k-means clustering.
%   2. Calculation of beta coefficient (which controls the width of the 
%      RBF neuron activation function) for each RBF neuron.
%   3. Training of output weights for each category using gradient descent.

% Add the subdirectories to the path.
addpath('kMeans');
addpath('RBFN');


% Load the data set. 
 X = loadMNISTImages('train-images.idx3-ubyte')';
 y = loadMNISTLabels('train-labels.idx1-ubyte');
 X_test = loadMNISTImages('t10k-images.idx3-ubyte')';
 y_test = loadMNISTLabels('t10k-labels.idx1-ubyte');



% Set 'm' to the number of data points.
m = size(X, 1);

% Set 'm_test' to the number of test-data points.
m_test = size(X_test, 1);

% Change the 0s with 10s for the code
for  (i = 1 : m)
    if y(i)==0
        y(i)=10;
    end
end

% Change the 0s with 10s for the code
for  (i = 1 : m_test)
    if y_test(i)==0
        y_test(i)=10;
    end
end


tic; %start timer
% ===================================
%     Train RBF Network
% ===================================

disp('Training the RBFN...');

% Train the RBFN using 10 centers per category.
[Centers, betas, Theta] = trainRBFN(X, y, 10, true);     % 10 is the number of centers per category, | category=10.
 
toc; %stop timer

% ========================================
%       Measure Training Accuracy
% ========================================

disp('Measuring training accuracy...');

numRight = 0;

wrong = [];

% For each training sample...
for (i = 1 : m)
    % Compute the scores for all categories.
    scores = evaluateRBFN(Centers, betas, Theta, X(i, :));
    
	[maxScore, category] = max(scores);
	
    % Validate the result.
    if (category == y(i))
        numRight = numRight + 1;
    else
        wrong = [wrong; X(i, :)];
    end
    
end

accuracy = numRight / m * 100;
fprintf('Training accuracy: %d / %d, %.1f%%\n', numRight, m, accuracy);


% ========================================
%       Measure Test Accuracy
% ========================================

disp('Measuring test accuracy...');

numRight = 0;

wrong = [];

% For each test sample...
for (i = 1 : m_test)
    % Compute the scores for all categories.
    scores = evaluateRBFN(Centers, betas, Theta, X_test(i, :));
    
	[maxScore, category] = max(scores);
	
    % Validate the result.
    if (category == y_test(i))
        numRight = numRight + 1;
    else
        wrong = [wrong; X_test(i, :)];
    end
    
end

accuracy = numRight / m_test * 100;
fprintf('Test accuracy: %d / %d, %.1f%%\n', numRight, m_test, accuracy);
if exist('OCTAVE_VERSION') fflush(stdout); end;
