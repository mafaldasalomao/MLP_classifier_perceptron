clc, clear
X1= [0.01 0.01 0.99 0.99;
       0.01 0.99 0.01 0.99];
Yd = [0.01 0.99 0.99 0.01];  % patrones objetivos

Nin=2; % cantidad de entradas
Noc=2; % cantidad nodos en capa oculta
Nout=1; % cantidad de salidas
eta=0.5; alpha=0.003;
errorPermitido=1;

W1=rand(Nin+1,Noc); %incluye el bias
W2=rand(Noc+1,Nout); %incluye el bias

DW1=0*W1; DW2=0*W2;
W1prev=0*W1; W2prev=0*W2;

B1=ones(1,4);
X1of=[X1; B1]; % entradas con bias

iter=1;
while errorPermitido ( 0.2 & iter ) 10 %error permitido
    sumError = 0;
    for muestra = 1:length(Yd)
        Y1 = 1 ./ (1 + exp(  -(  W1(1:Nin+1,:)' * X1of(:,muestra)   ) )   ); % salida de neuronas de la capa 1
        Y1of=[Y1; 1]; % vector Y1 con bias
        Y2 = 1 ./ (1 + exp( -( W2(1:Noc+1,:)' * Y1of)  )   ); % salida de neuronas de la capa 2

        % CALCULA CAMBIOS A LOS PESOS DE CAPA DE SALIDA W2
        for nk = 1:Nout %es un vector de un elemento por cada salida
            deltaJsalida(nk) = Y2(nk) * ( 1 - Y2(nk) ) * ( Yd(nk,muestra) - Y2(nk) );
            for nj = 1:Noc+1
               DW2(nj , nk) = deltaJsalida(nk) * Y1of(nj); %define la matriz de diferencia (delta) con tantas columnas como salidas
            end
        end

        % CALCULA CAMBIOS A PESOS DE CAPA DE OCULTA W1
        for nj = 1:Noc %calcula el error de cada neurona nj de la capa oculta
            SUMk = 0;
            for k = 1:Nout
                SUMk = SUMk + deltaJsalida(k) * W2(nj,k);
            end
            deltaJoculta(nj) = Y1(nj) * ( 1 - Y1(nj) ) * SUMk;
            for ni = 1:Nin+1
                DW1(ni , nj) = deltaJoculta(nj) * X1of(ni,muestra); % el error calculado se asigna a la variacion que tendrá la matriz
            end
        end
% actualiza los pesos
        W1nuevo = W1 + eta*DW1 + alpha*(W1 - W1prev);
        W1prev=W1;
        W1=W1nuevo;
        W2nuevo = W2 + eta*DW2 + alpha*(W2 - W2prev);
        W2prev=W2;
        W2=W2nuevo;

        sumError = sumError + (Yd(muestra) - Y2)^2;
    end %muestra
    errorTotal(iter)=sumError;
    errorPermitido = sumError;
    iter = iter + 1;
end % while

%mostrar los resultados con las muestras en plano 3d
for muestra = 1:length(Yd);
    y1 = 1 ./ (1 + exp(  -(  W1(1:Nin+1,:)' * X1of(:,muestra)   ) )   );
    Y1of=[Y1; 1]; 
    y2(muestra) = 1 ./ (1 + exp( -( W2(1:Noc+1,:)' * Y1of)  )   );
end
fig2=figure;
fig2.InnerPosition= [21 61 418 354]; 
plot3([0 0 1 1], [0 1 0 1], y2, '*'), 

ejes2=gca;
ejes2.XLim = [-0.2 1.2];
ejes2.YLim= [-0.3 1.3];
ejes2.PlotBoxAspectRatio= [1 .4 .4];


fig=figure;
fig.InnerPosition=[16 356 535 323]; %posicion en la pantalla
plot(errorTotal,'*'),
