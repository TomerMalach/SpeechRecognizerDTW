function [Accuracy] = evaluateCB(data, CB, MKM, Fs, SamplesPerWindow, WindowsLength)
    
    Numbers = size(data, 1);
    Speakers = size(data, 2);

    Prediction = zeros(Numbers, Speakers);
    Accuracy = zeros(Numbers, 1);
    
    K = 2; 

    for num = 1:Numbers
        
        MFCCs = size(CB{num}, 1) - 1;
        
        for speaker = 1:Speakers
            
            % Edge Detector
            [StartPoint, EndPoint] = edge_point_detect(data{num,speaker}, Fs, 0);
            data{num,speaker} = data{num,speaker}(StartPoint:EndPoint);
            
            for i=1:Numbers % change to parfor!!!!!!
            
                TypicalWordFrames = size(CB{i}, 2);
                
                % Adjust overlap to the typical word length
                RecordLength = size(data{num, speaker}, 1);
                StepSizeBetweenFrames = round(RecordLength/(TypicalWordFrames + 1)); 

                % Framing
                FramesSig = enframe(data{num,speaker}, SamplesPerWindow, StepSizeBetweenFrames);

                % Hamming Window
                NumberOfFrame = size(FramesSig, 1);
                HammingWindow = hamming(SamplesPerWindow); % how much windows to create
                FramesSig = (FramesSig .* repmat(HammingWindow', NumberOfFrame, 1))'; 

                coeffs = squeeze(mfcc(FramesSig ,Fs, 'WindowLength', round(Fs*WindowsLength), 'OverlapLength', round(Fs*WindowsLength*0.8), 'NumCoeffs', MFCCs));

                if MKM
                    dd = zeros(size(CB{i}, 3), 1);
                    parfor c = 1:size(CB{i}, 3)
                        dd(c) = dynamic_dist(coeffs, squeeze(CB{i}(:, :, c)), 0);
                    end
                    dd = sort(dd);
                    Prediction(i, speaker) = mean(dd(1:K)); 
                else
                    Prediction(i, speaker) = dynamic_dist(coeffs, CB{i}, 0);
                end
            end
        end

        [~, argmin] = min(Prediction, [], 1);
        Accuracy(num) = sum(argmin == num)/length(data(num, :));
        
        display(['Accuracy for number ' num2str(num-1) ' is ' num2str(Accuracy(num))] );
    end
end

