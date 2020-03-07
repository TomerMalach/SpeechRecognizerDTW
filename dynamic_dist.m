function [ddist, L] = dynamic_dist(A, B, f)
    
    I = size(A, 2);
    J = size(B, 2);
        
    Fh = 1:2:(4*I-2);
    Fh = Fh(1:I);
    
    Fl = 1:0.5:I;
    Fl = Fl(1:I);
    
    Gh = (J-0.5*(size(Fh, 2)-1)):0.5:J;
    Gh = Gh(1:I);
    
    Gl = (J-2*(size(Fh, 2)-1)):2:J;
    Gl = Gl(1:I);
    
    Lower_Bound = max(Fl, Gl);
    Higher_Bound = min(Fh, Gh);
      
    D = inf*ones(J, I);
    L = zeros(J, I);
     
    if ndims(B) == 3
        D(1, 1) = min(vecnorm(squeeze(B(:, 1, :)) - repmat(A(:, 1), 1, size(B, 3))));
    else
        D(1, 1) = norm(B(:, 1) - A(:, 1));
    end
    
    for i=2:I
        for j=ceil(Lower_Bound(i)):floor(Higher_Bound(i))
            if i > 2
                if ndims(B) == 3
                    D12 = D(j - 1, i - 2) + ...
                        2*min(vecnorm(squeeze(B(:, j, :)) - repmat(A(:, i - 1), 1, size(B, 3)))) + ...
                        min(vecnorm(squeeze(B(:, j, :)) - repmat(A(:, i), 1, size(B, 3))));
                else
                    D12 = D(j - 1, i - 2) + 2*norm(B(:, j) - A(:, i - 1)) + norm(B(:, j) - A(:, i));
                end
            else
                D12 = inf;
            end
            
            if ndims(B) == 3
                D11 = D(j - 1, i - 1) + ...
                    2*min(vecnorm(squeeze(B(:, j, :)) - repmat(A(:, i), 1, size(B, 3))));
            else
                D11 = D(j - 1, i - 1) + 2*norm(B(:, j) - A(:, i));
            end
            
            if j > 2
                if ndims(B) == 3
                    D21 = D(j - 2, i - 1) + ...
                        2*min(vecnorm(squeeze(B(:, j - 1, :)) - repmat(A(:, i), 1, size(B, 3)))) + ...
                        min(vecnorm(squeeze(B(:, j, :)) - repmat(A(:, i), 1, size(B, 3))));
                else
                    D21 = D(j - 2, i - 1) + 2*norm(B(:, j - 1) - A(:, i)) + norm(B(:, j) - A(:, i));
                end
            else
                D21 = inf;
            end
            
            [D(j, i), L(j, i)] = min([D12, D11, D21]);
        end
    end
    
    if f
        figure;
        plot(I, J, 'r*');
        ylim([1 J]);
        xlim([1 I]);
        hold on;
        grid on;
        plot(Fh);
        plot(Fl);
        plot(Gh);
        plot(Gl);
        plot(Lower_Bound, 'b*');
        plot(Higher_Bound, 'g*');
        i = I;
        j = J;
        while i > 1
            plot(i, j, 'm*');
            if L(j, i) == 1  % D12
                i = i - 2;
                j = j - 1;
            elseif L(j, i) == 2  % D11
                 i = i - 1;
                 j = j - 1;
            elseif L(j, i) == 3  % D21
                i = i - 1;
                j = j - 2;
            else
                break;
            end
        end
    end
    
    ddist = D(J, I)/(I + J);
    
end

