close all;clear all;clc % Limpa tudo
%file = 'base_atividade_1.xlsx';
%[Amostras, data] = xlsread(file) % lê os dados da tabela excel
% y = size(Amostras); % pega o tamanho da matriz Amostras
% n1 = y(1); % n1 igual a quantidade de linhas
% n2 = y(2); % n2 igual a quantidade de colunas
x1 = [0.75 0.75 0.25 0.25 0.70 0.70 0.20 0.20]
x2 = [0.75 0.25 0.75 0.25 0.70 0.30 0.70 0.20]
classe = ['A' 'B' 'B' 'A' 'A' 'B' 'B' 'A'] % A = 0 B =1 

outputs = [0 1 1 0 0 1 1 0];

n1 = 8 ; n2 = 3 ; 

% Classe_a = [0.75 0.25 0.70 0.20; 0.75 0.25 0.70 0.20]
% output
% Classe_b = [0.75 0.25 0.70 0.20; 0.25 0.75 0.30 0.70]

%plotpv([x1;x2], outputs);
amostra = [x1' x2' outputs'];
w = rand (1,n2) % cria um vetor w com valores randômicos
n = (n1*n2)+(n1*n2);% taxa de aprendizagem é a soma de dois produtos
n = 1/n % taxa de aprendizagem igual a uma porcentagem de seu valor
e = 0.05; % taxa de precisão requerida
eqm = 0; % Erro quadrático médio
epoca = 0; % inicia epoca com valor zero
i=1;
target = outputs';
for i = 1:n1;
    eqmant = eqm;
%     flag = true;
        flag = 1;
        while flag > e % enquanto erro maior que a taxa de precisão pedida
        u = amostra(i,:)*w';
        w = w + (n*((target(i,1)-u)*amostra(i,:)))
        epoca= epoca +1;
        eqm = eqm + ((target(i,1)-u)^2);
        eqm = eqm / n1;
        eqmatual = eqm;
        erro = eqmatual - eqmant;
        erro = abs(erro);
        flag = erro;
        end
        i = i + 1
end

net = perceptron;
net = configure(net,[x1; x2],outputs);
net.iw{1,1} = [-1.2 -0.5];
net.b{1} = 1;
plotpc(net.iw{1,1},net.b{1})

%% 
% Run_XOR_MLP_Newff.m
x1 = [0.75 0.75 0.25 0.25 0.70 0.70 0.20 0.20]
x2 = [0.75 0.25 0.75 0.25 0.70 0.30 0.70 0.20]

Classe_a = [0.75 0.25 0.70 0.20; 0.75 0.25 0.70 0.20]
% output
Classe_b = [0.75 0.25 0.70 0.20; 0.25 0.75 0.30 0.70]
outputs = [0 1 1 0 0 1 1 0];
P = [x1; x2]; % XOR Function
T = outputs;
%plotpv(P,T,[-1, 2, -1, 2]); % plot data
PR = [min(P(1,:)) max(P(1,:));
min(P(2,:)) max(P(2,:))];
S1 = 2;
S2 = 1;
TF1 = 'logsig';
TF2 = 'logsig';
PF = 'mse';
%
net = newff(PR,[S1 S2],{TF1 TF2});
%
net.trainParam.epochs = 100;
net.trainParam.goal = 0.001;
net = train(net,P,T);

a = sim(net, P);
%hold on;
plot(P, a, '.');