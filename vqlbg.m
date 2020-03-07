function r = vqlbg(d, k)
% VQLBG Vector quantization using the Linde-Buzo-Gray algorithme
%
% Inputs: d contains training data vectors (one per column)
%         k is number of centroids required
%
% Output: r contains the result VQ codebook (k columns, one for each centroids)
%
%
%%%%%%%%%%%%%%%%%%
% Mini-Project: An automatic speaker recognition system
%
% Responsible: Vladan Velisavljevic
% Authors:     Christian Cornaz
%              Urs Hunkeler

    e   = .001;  % stoping error
    r   = mean(d, 2); % initial center 
    dpr = 10000; % distance from previous run


    for i = 1:log2(k)

        r = [r*(1+e), r*(1-e)]; % split each center into 2 centers

        while (1 == 1)
            z = dist(d, r, 1); % compute distance between vectors to centers
            [m, ind] = min(z, [], 2); % get closest centers
            t = 0;
            for j = 1:2^i % for all the centers
                if length(find(ind == j)) > 0
                    r(:, j) = mean(d(:, find(ind == j)), 2); % compute the new center value based only on vectors that belog to the cluster
                    x = dist(d(:, find(ind == j)), r(:, j), 1); % compute distance between vetros to the updated center
                    t = t + sum(x); % sum all the distances to compute convergence %%%% t
                else
                    v = 3;
                end
            end

            % check if converged
            if (((dpr - t)/t) < e)
                break;
            else
                dpr = t;
            end
       end    
    end

end


