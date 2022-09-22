clc
clear all
close all

x1 = [1 -1 -2 -4; 
      2  1  1  0]
de = [1  0  0  1]

x2 = [-2;1]
saida = [0]

% (tamanho/ estrutura do MLP)
N1=4;                  % Neurônios da camada oculta
N2=1;                   % Neurônios da camada de saída 
N0=2;    % Neurônios da camada de entrada 

% Inicialização de pesos
w1=randn(N1,N0);    % Pesos iniciais das conexões de entrada e camada oculta
w2=randn(N2,N1);    % Pesos iniciais das conexões da camada oculta e de saída

biasW1 = randn(4,2);
biasW2 = randn(1,4);

bias = rand;
bias2 = rand;
eta = 0.1;
numEpoch = 2000;

A = [0 0 0 0];

for epoch=1:numEpoch    
    
        % Camada de Entrada       
        saidaN1 = w1 * x2 + biasW1; 
        resultEntrada = hardlim(saidaN1);
        
        % Camada Oculta        
        saidaN2 = w2 * resultEntrada + biasW2';
        resultFinal = hardlim(saidaN2);
        
        % Camada de Saída                      
        e = resultFinal - saida;  
        
        if isequal(A,e)
            break
        end
                        
        %Retropropagação        
        gradY2 = dhardlim(saidaN2,resultFinal).*e;        
        w2 = w2 - eta*gradY2*saidaN1' + bias2; 
                
        gradY1 = dhardlim(saidaN1,resultEntrada)+w2'*gradY2;                                  
        w1 = w1 - eta*gradY1*x2 + bias;
        
        biasW1 = biasW1 * e' - eta;        
        biasW2 = biasW2 * e - eta;        
    
end

figure
plotpv(x2,resultFinal);
grid
xlim ([-5 2]);
ylim ([-2 4]);
title('Resultado do treinamento')


  