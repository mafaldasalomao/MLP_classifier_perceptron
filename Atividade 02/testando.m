clc
clear all
close all


valorEntrada= [0.01 0.01 0.99 0.99];
objetivo = [0.01 0.99 0.99 0.01];  % patrones objetivos


N0=2; % Neurônios da camada de entrada (feature length + bias)
N1=2; % Neurônios da camada oculta
N2=1; % Neurônios da camada de saída 
eta=0.5;

epoch=10;

w1=rand(N0+1,N1); %incluye el bias
w2=rand(N1+1,N2); %incluye el bias


for j=1:epoch
    
    
    for k = 1:length(objetivo)
        
        Input=[1 valorEntrada(1,k)];  % {1} é para o bias
        
                % Camada de Entrada
        n1 = w1*Input';
        a1=tansig(n1);       % Função de ativação sigmóide tangente hiperbólica

         % Camada Oculta
        n2 = w2*a1;
        a2=logsig(n2);       % Função de ativação log-sigmóide
        
        % Camada de Saída
        Output(k,:)=a2';    
        e = valorEntrada(ind(1,k),:) - Output(k,:);    % Cálculo do erro, diferença do resultado 
                                                      % desejado para o resultado obtido.
        
        % Após o cálculo do erro, pode-se ajustar o peso utilizando a regra delta, 
        % utilizado para ajustar os pesos dos neurônios
        % da camada de saída.
        
        Y2 = 2*dlogsig(n2,a2).*e';  % Onde o Gradiente equivale ao valor do erro * 
                                    % derivada da função de ativação em função de V.
    
        Y1 = diag(dtansig(n1,a1),0)*w2'*Y2; % Gradiente local da camada oculta / 
                                            % Cálculo da derivada da função de ativação de em função de V * 
                                            % peso * Gradiente do neurônio de saida ou camada anterior
        
        w1 = w1 + eta*Y1*Input;  % Atualização de peso de neurônios da camada de entrada
                                 % Equivale ao peso atual + taxa de aprendizagem * delta * o valor de entrada
                                 
        w2 = w2 + eta*Y2*a1';    % Atualização de peso de neurônios da camada de saída
                                 % Equivale ao peso atual + taxa de aprendizagem * delta
                                 % * valor de saída intermediária do neurônio da camada anterior
        
        SE(j,k)= e*e';      % erro quadrado
end
MSE(j)=mean(SE(j,:));       % função objetivo (erro quadrático médio)
                            % Treine a rede e avalie o desempenho.


end

MSE(j)=mean(SE(j,:));       % função objetivo (erro quadrático médio)
                            % Treine a rede e avalie o desempenho.
