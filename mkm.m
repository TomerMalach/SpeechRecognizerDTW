function [centers_indecies] = mkm(data, num_of_centers)

    j = 2;
    K = 10;
    
    % TODO ADD parfor!!!!!!!!!!!!!!!!!

   
    % Distance between each 2 MFCCs
    main_distances = zeros(size(data, 3));
    for speaker1 = 1:size(data, 3)
        parfor speaker2 = 1:speaker1 - 1 
            if speaker1 ~= speaker2
                main_distances(speaker1, speaker2) = dynamic_dist(squeeze(data(:, :, speaker1)), squeeze(data(:, :, speaker2)), 0);
                %main_distances(speaker2, speaker1) = main_distances(speaker1, speaker2);
            end
        end
    end
    main_distances = main_distances + main_distances';
    
    max_distance = max(main_distances, [], 'all');
    [c1_index c2_index] = find(main_distances == max_distance);
    
    % Remain only two vecs
    if length(c1_index) > 1
        c1_index = c1_index(1);
        c2_index = c2_index(1);
    end
    
    centers_indecies = [c1_index c2_index];
    
    while 1 == 1
    
        k = 0;
        prev_center_index = zeros(size(data, 2), 1);
        
        while 1 == 1
            
            % Compute distance between centers to data for classify
            distances = main_distances(:, centers_indecies);
            
            % Classify each sample to certain center index
            [~, center_index] = min(distances, [], 2);

            % Check if some sample changed its cluster
            if ~isequal(center_index, prev_center_index)
                k = k + 1;
                if k > K
                    break;
                end
                prev_center_index = center_index;
            else
                break;
            end

            % Recompute the centers values
            for i = 1:size(centers_indecies, 2)
                samples = find(center_index == i);
                distances = main_distances(samples, samples);
                
                [~, new_center_index] = min(max(distances)); % MinMax dynamic distance
                centers_indecies(i) = samples(new_center_index);
            end
        end

        j = j + 1;
        if j > num_of_centers
            break;
        end

        max_dist_cluster_index = 0;
        max_dist_cluster_value = 0;

        % Choose which cluster to split
        for i = 1:size(centers_indecies, 2)
            samples =find(center_index == i);
            max_distance = mean(main_distances(samples, centers_indecies(i)));
                 
            if max_distance > max_dist_cluster_value
                max_dist_cluster_index = i;
                max_dist_cluster_value = max_distance;
            end
        end

        samples =find(center_index == max_dist_cluster_index);
        max_dist_cluster_value = max(main_distances(samples, samples), [], 'all');
        distances = main_distances(samples, samples);
        
        % Split the centers
        [c1_index c2_index] = find(distances == max_dist_cluster_value);
        
        % Remain only two vecs
        if length(c1_index) > 1
            c1_index = c1_index(1);
            c2_index = c2_index(1);
        end
        
        c1_index = samples(c1_index);
        c2_index = samples(c2_index);
        
        
        centers_indecies(max_dist_cluster_index) = c1_index;
        centers_indecies = [centers_indecies c2_index];
    
    end
end

