classdef MinimalLearningMachine

    properties
        refPoints, B, nRefs
    end
    
    methods
        function obj = MinimalLearningMachine(data, refPoints)
            assert(size(refPoints.y, 2) == 1);
            Dx = pdist2(data.x, refPoints.x);
            Dy = pdist2(data.y, refPoints.y);
            obj.nRefs = size(refPoints.x, 1);
            obj.refPoints = refPoints;
            obj.B = pinv(Dx'*Dx)* Dx'* Dy;
        end
        
        function yh = predict(MLM, X, method)
            Dx = pdist2(X, MLM.refPoints.x);
            Dyh = Dx*MLM.B;
            if strcmp(method, 'lsqnonlin') % Dumped Least-Squares
                options_lsq = optimoptions('lsqnonlin','Algorithm', ...
                    'levenberg-marquardt', 'Jacobian','on', ...
                    'FunValCheck', 'on', 'Display', 'off' );
                yh = zeros(size(X, 1), size(MLM.refPoints.y, 2));
                yh0 = mean(MLM.refPoints.y); 
                for i = 1: size(X, 1),   
                    yh(i, :) = lsqnonlin(@(x)(fun(x, MLM.refPoints.y, ...
                        Dyh(i, :))), yh0, [], [], options_lsq);
                end
            else % Linear Program
                options = optimoptions('linprog','Algorithm', ...
                    'dual-simplex', 'Display', 'off');
                yh = zeros(size(X, 1), size(MLM.refPoints.y, 2));
                k = MLM.nRefs;
                c = [ones(k, 1); zeros(k, 1) ; 0];                
                A = [zeros(k, k)         eye(k)      -ones(k, 1);    ...
                     zeros(k, k)         eye(k)      ones(k, 1);     ...
                     eye(k)              -eye(k)     zeros(k, 1);    ...
                     eye(k)              eye(k)      zeros(k, 1)];                 
                for i = 1: size(X, 1)
                    b = [-MLM.refPoints.y ; MLM.refPoints.y; ...
                        -Dyh(i,:)'; Dyh(i,:)'];
                    sol =  linprog(c, -A, -b, [], [], [], [], [], options);
                    yh(i, :) = sol(end);
                end
            end
        end
    end
    
end

function [F, J] = fun(x, refY, DYh)
    
    for i = 1: size(refY, 1),
        F(i) = norm(x-refY(i, :), 2)^2 - DYh(i)^2;
    end
    
    if nargout > 1   % Two output arguments
        for i = 1: size(refY, 1),
            for j = 1: size(refY, 2),
                J(i, j) = 2*(x(j) - refY(i, j));
            end
        end
    end
    
end