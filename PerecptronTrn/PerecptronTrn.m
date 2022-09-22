function [w,b,pass,wevol]=PerecptronTrn(x,y,n);
% %Rosenblatt's Perecptron
[l,p]=size(x);   % l=number of patterns, p=input vector size
x=[-ones(l,1) x];
w=zeros(p+1,1); % initialize weights
wevol=w;
ier=1;        % initialize a misclassification indicator
pass=0;    % number of iterations
while ier==1, %repeat until no error
       ier=0;
       e=0; % number of training errors
       for i=1:l  % a pass through x           
           xx=x(i,:);
           ey=xx*w;       % estimated y
           if (ey>=0 && y(i)==-1);
              w=w-n*xx'; % the only rule works for me             
              e=e+1 ; % number of training errors
              wevol=[wevol w];
           end;
           if (ey<0 && y(i)==1);
              w=w+n*xx'; % the only rule works for me             
              e=e+1 ; % number of training errors
              wevol=[wevol w];
           end;
       end;  
       ee=e;    % number of training errors
       if ee>0  % cuntinue if there is still errors
          ier=1;           
       end
       pass=pass+1; % stop after 10000 iterations
       if pass==20000
          ier=0;
          pass=0;
       end;
 end;
 b=w(1,1);
 w=w(2:p+1,:);
 
disp(['Training_Errors=' num2str(e) '     Training data Size=' num2str(l)])
