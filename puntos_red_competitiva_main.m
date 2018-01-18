close all; clear all; clc;

m = 2000;
pt = 0.5;
r = randi([-100,100],2,m);
val_max = 100;
y = zeros(1,m);
drojo = 1;
dazul = 1;

%%% Etiquetado de datos
for im = 1:1:m
    h = sqrt( r(1,im)*r(1,im) + r(2,im)*r(2,im) );    
    if h <= 80
        y(1,im)= 1;
        datosrojo(:,drojo) = r(:,im);
        drojo = drojo + 1;
    else
        y(1,im)= -1;
        datosazul(:,dazul) = r(:,im);
        dazul = dazul + 1;
    end
end

% %%% Se muestran los datos clasificados
% figure
% plot(datosrojo(1,:),datosrojo(2,:),'ro','linestyle','none')
% hold on
% plot(datosazul(1,:),datosazul(2,:),'bo','linestyle','none')
% hold off
% xlim([-100 100])
% ylim([-100 100])
% pause

X1 = r(:,1:pt*m);
Y1 = y(1,1:pt*m);

[R,n] = size(X1);

epocas = 1000;

for C = 10:1:10 %neuronas
    
    W = randn(R,C);
    W = W/max(max(W))*val_max;
    
    figure
    plot(datosrojo(1,:),datosrojo(2,:),'ro','linestyle','none')
    hold on
    plot(datosazul(1,:),datosazul(2,:),'bo','linestyle','none')
    plot(W(1,1:5),W(2,1:5),'k*','linestyle','none')
    plot(W(1,6:10),W(2,6:10),'g*','linestyle','none')
    hold off
    xlim([-100 100])
    ylim([-100 100])
    pause

    val_test = zeros (1,epocas);
    val_training = zeros (1,epocas);

    for rt = 1:1:epocas

        %%% Entrenamiento
        for nx = 1:1:n
            X = X1(:,nx);
            Y = Y1(:,nx);
            [W,aciertos,memo] = training_red_competitiva(X,Y,C,W,0.05);
    %         val_training(1,rt) = aciertos;

%             figure
%             plot(datosrojo(1,:),datosrojo(2,:),'ro','linestyle','none')
%             hold on
%             plot(datosazul(1,:),datosazul(2,:),'bo','linestyle','none')
%             plot(W(1,1:5),W(2,1:5),'k*','linestyle','none')
%             plot(W(1,6:10),W(2,6:10),'g*','linestyle','none')
%             plot(X(1,:),X(2,:),'m+','linestyle','none')
%             hold off
%             xlim([-100 100])
%             ylim([-100 100])
%             pause
        end

        res = 0;
        dnegro = 1;
        dcyan = 1;
        
        %%% Evaluación
        for xi = pt*m+1:1:m
            
            etiqueta = test_red_competitiva(r(:,xi),W);

            if etiqueta == 1
                datosnegro(:,dnegro) = r(1:2,xi);
                dnegro = dnegro +1;
            else
                datoscyan(:,dcyan) = r(1:2,xi);
                dcyan = dcyan + 1;
            end

            if etiqueta == y(1,xi)
                res = res + 1;
            end
        end

        %%% Resultados gráficos
        if mod(rt,epocas/2)==0
            memo
            figure
            plot(datosrojo(1,:),datosrojo(2,:),'ro','linestyle','none')
            hold on
            plot(datosazul(1,:),datosazul(2,:),'bo','linestyle','none')
            plot(datosnegro(1,:),datosnegro(2,:),'k*','linestyle','none')
            plot(datoscyan(1,:),datoscyan(2,:),'g*','linestyle','none')
            plot(W(1,1:5),W(2,1:5),'m+','linestyle','-')
            plot(W(1,6:10),W(2,6:10),'c+','linestyle','-')

            xlabel(C)
            hold off
            xlim([-100 100])
            ylim([-100 100])
            
            figure
            plot(datosnegro(1,:),datosnegro(2,:),'r*','linestyle','none')
            hold on
            plot(datoscyan(1,:),datoscyan(2,:),'b*','linestyle','none')
            xlabel(C)
            hold off
            xlim([-100 100])
            ylim([-100 100])
            pause

        end

        val_test(1,rt) = res;
    end
    
    %%% Resultados por cada época
    figure
    plot(val_training)
    title('Aciertos Entrenamiento')
    xlabel(C)
    figure
    plot(val_test)
    title('Aciertos Evaluación')
    xlabel(C)    
end