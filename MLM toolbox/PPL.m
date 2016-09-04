function[yh] = PPL(T,D)

       k = size(T,1);
       dime =[k 1];
       C = [ones(k, 1); zeros(k, 1) ; 0];
       
       A = [zeros(k, k) eye(k) -ones(k, 1);
           zeros(k, k) eye(k) ones(k, 1);
           eye(k) -eye(k) zeros(k, 1);
           eye(k) eye(k) zeros(k, 1)];      
       B = [-eye(k)*T ; eye(k)*T; -eye(k)*D; eye(k)*D];
    
       resultado = linprog(C, -A, -B);
       yh = resultado(end:end);
%        sugestão
%        for i=1:size(D,1),
%           linprog(C,A,B(:,i))
%        end
%       yh
end