function d = dist(x, y, type) 

    % x - centers
    % y - frames coeffs

    [M, N] = size(x); 
    [M2, P] = size(y); 
    
    if (M ~= M2) 
        error('Matrix dimensions do not match.') 
    end 
    
    d = zeros(N, P);
    
    for ii=1:N 
        for jj=1:P 
            % Euclidian
            if type == 1 
                d(ii,jj) = sum((x(:,ii)-y(:,jj)).^2).^0.5; 
            
            % Distortion
            elseif type == 0       
                Ry = toeplitz(y(:, jj));
                ay = levinson(y(:, jj))';
                %ay = [1; -Ry(1:end-1,1:end-1)^-1*y(2:end, jj)];
                
                %Rx = toeplitz(x(1:end-1, ii));
                %ax = [1; -Rx^-1*x(2:end, ii)];
                
                ax = levinson(x(:, ii))';
                
                den = ax'*Ry*ax;
                num = ay'*Ry*ay;
                
                d(ii,jj) = log(den/num); 
            end
        end
    end
end 