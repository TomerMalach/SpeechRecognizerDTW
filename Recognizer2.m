clc;
clear all;
close all;

load data_training-test.mat

Fs = 16384; % 16KHz

MKM = 1;

Training = 1;
TrainAccuracy = 1;
ValidationAccuracy = 1;
TestAccuracy = 1;

% Cross varidation (train: 70%, val: 30%)
cv = cvpartition(size(training_data, 2),'HoldOut',0.3);
idx = cv.test;

% Separate to training and test data
%dataTrain = training_data(:, 1:73);
%dataVal  = training_data(:, 74:end);

dataTrain = training_data(:, ~idx);
% dataTrain = training_data;
dataVal  = training_data(:, idx);

Overlap = 0.5;
MFCCs = 13;
WindowsLength = 10*10^-3; % msec

Numbers = size(dataTrain, 1);
Speakers = size(dataTrain, 2);

NumberOfSamplesAtEachWindow = round(Fs * WindowsLength); 
StepSizeBetweenFrames = round(Overlap * NumberOfSamplesAtEachWindow);


%% Training

if Training

    LBG = 1;
    
    dataTrainMFCC = cell(Numbers, Speakers); % allocate tarin dataset MFCC
    dtwMFCC = cell(Numbers, 1); % allocate fixed length train dataset MFCC 
    NumsCodeBook = cell(Numbers, 1); % Codebook for the numbers
    CodeBookSize = 16;
    
    for num = 1:Numbers
                      
        % Extract MFCC for all training dataset
        HammingWindow = hamming(NumberOfSamplesAtEachWindow); % how much windows to create
        for speaker = 1:Speakers % par

            % Edge Detector
%             subplot(2,1,1);
%             plot(dataTrain{num,speaker});
            [StartPoint, EndPoint] = edge_point_detect(dataTrain{num,speaker}, Fs, 0);
            dataTrain{num,speaker} = dataTrain{num,speaker}(StartPoint:EndPoint);
%             subplot(2,1,2);
%             plot(dataTrain{num,speaker});
                
            % Framing
            FramesSig = enframe(dataTrain{num,speaker}, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);

            % Hamming Window
            NumberOfFrame = size(FramesSig, 1);
            FramesSig = (FramesSig .* repmat(HammingWindow', NumberOfFrame, 1))';           
            
            dataTrainMFCC{num, speaker} = squeeze(mfcc(FramesSig ,Fs, 'WindowLength', round(Fs*WindowsLength), ...
                                         'OverlapLength', round(Fs*WindowsLength*Overlap), 'NumCoeffs', MFCCs));                
        end

        
        % Compute dynamic dist between all of the records
        dynamic_distances = zeros(Speakers);
        for speaker1 = 1:Speakers
            parfor speaker2 = 1:Speakers
                if speaker1 ~= speaker2
                    dynamic_distances(speaker1, speaker2) = dynamic_dist(dataTrainMFCC{num,speaker1}, dataTrainMFCC{num,speaker2}, 0);
                end
            end
        end
        
        dynamic_distances(dynamic_distances == inf) = -1;
        dynamic_distances(dynamic_distances == -1) = max(dynamic_distances, [], 'all');
        [~, typical_word_index] = min(sum(dynamic_distances));
         
        %[~, typical_word_index] = min(max(dynamic_distances)); % MinMax dynamic distance
        typical_word = dataTrainMFCC{num,typical_word_index};  
        TypicalWordFrames = size(typical_word, 2);
  
        
        % Adjusting words length to the typical word length (by changing
        % the overlap)
        for speaker = 1:Speakers % par
            if speaker == typical_word_index
                continue;
            end
            
            % Compute the new overlapping
            RecordLength = size(dataTrain{num, speaker}, 1);
            StepSizeBetweenFrames = round(RecordLength/(TypicalWordFrames + 1)); 
               
            % Framing
            FramesSig = enframe(dataTrain{num, speaker}, NumberOfSamplesAtEachWindow, StepSizeBetweenFrames);

            % Hamming Window
            NumberOfFrame = size(FramesSig, 1);
            FramesSig = (FramesSig .* repmat(HammingWindow', NumberOfFrame, 1))';           
            
            dataTrainMFCC{num, speaker} = squeeze(mfcc(FramesSig ,Fs, 'WindowLength', round(Fs*WindowsLength), ...
                                          'OverlapLength', round(Fs*WindowsLength*0.8), 'NumCoeffs', MFCCs));                   
        end

        
        % Linear Time Wrapping (LTW) to the typical word
        dtwMFCC{num} = zeros(MFCCs + 1, TypicalWordFrames, Speakers);                
        
        for speaker = 1:Speakers
            
            % DTW - Add relexation!!!!!!!!!!!!!!
            [~, mapping] = dynamic_dist(dataTrainMFCC{num,typical_word_index}, dataTrainMFCC{num, speaker}, 0);
             
            % Track the path
            i = size(mapping, 2);
            j = size(mapping, 1);
            dynamic_map = [];
            
            while i > 1 && j > 1
                dynamic_map = [mapping(j, i) dynamic_map];
                if dynamic_map(1) == 1  % D12
                    i = i - 2;
                    j = j - 1;
                elseif dynamic_map(1) == 2  % D11
                     i = i - 1;
                     j = j - 1;
                elseif dynamic_map(1) == 3  % D21
                    i = i - 1;
                    j = j - 2;
                else
                    break;
                end
            end
            
            % LTW
            i = 1;
            j = 1;
            for k = 1:size(dynamic_map, 2)
                if dynamic_map(k) == 1  % Duplicate
                    dtwMFCC{num}(:, i, speaker) = dataTrainMFCC{num, speaker}(:, j);
                    dtwMFCC{num}(:, i + 1, speaker) = dataTrainMFCC{num, speaker}(:, j);
                    i = i + 2;
                    j = j + 1;
                elseif dynamic_map(k) == 2  % Copy
                    dtwMFCC{num}(:, i, speaker) = dataTrainMFCC{num, speaker}(:, j);
                    i = i + 1;
                    j = j + 1;
                elseif dynamic_map(k) == 3  % Average
                    dtwMFCC{num}(:, i, speaker) = mean(dataTrainMFCC{num, speaker}(:, j:j+1), 2);
                    i = i + 1;
                    j = j + 2;
                else
                    break;
                end
                %plot(i, j, 'k*');
            end  
            
            %if dynamic_map(end) ~= 1%%%%%%%%%%%%%%%%%%%
            dtwMFCC{num}(:, end, speaker) = dataTrainMFCC{num, speaker}(:, end);
            %end
        end
        
        if MKM
            % MKM CodeBookLength per record
            NumsCodeBook{num} = dtwMFCC{num}(:, :, mkm(dtwMFCC{num}, CodeBookSize));
        else
            % LBG/Kmeans CodeBookLength per frame
            NumsCodeBook{num} = zeros(MFCCs + 1, TypicalWordFrames, CodeBookSize);
            for k = 1:TypicalWordFrames
                if LBG
                    NumsCodeBook{num}(:, k, :) = vqlbg(squeeze(dtwMFCC{num}(:, k, :)), CodeBookSize);
                else
                    [~, centers] = kmeans(squeeze(dtwMFCC{num}(:, k, :))', CodeBookSize);
                    NumsCodeBook{num}(:, k, :) = centers';    
                end
            end
        end
        
        display(['CodeBook for number ' num2str(num-1) ' is ready!']);
    end   
    
    save(['CB_MKM_' num2str(MKM) '_'  datestr(now,'dd-mm-yy_HH-MM') '.mat'], 'NumsCodeBook');

end

%% Train Accuracy
if TrainAccuracy
    
    disp('---------------- Compute Train Accuracy ----------------');
    
    Accuracy = evaluateCB(dataTrain, NumsCodeBook, MKM, Fs, NumberOfSamplesAtEachWindow, WindowsLength);

    disp('Train Accuracy:');
    disp(mean(Accuracy));

end


%% Validation Accuracy

if ValidationAccuracy
    
    disp('---------------- Compute Validation Accuracy ----------------');
    
    Accuracy = evaluateCB(dataVal, NumsCodeBook, MKM, Fs, NumberOfSamplesAtEachWindow, WindowsLength);

    disp('Validation Accuracy:');
    disp(mean(Accuracy));

end


%% Test Accuracy

if TestAccuracy
    
    disp('---------------- Compute Test Accuracy ----------------');
    
    Accuracy = evaluateCB(test_data, NumsCodeBook, MKM, Fs, NumberOfSamplesAtEachWindow, WindowsLength);

    disp('Test Accuracy:');
    disp(mean(Accuracy));

end






