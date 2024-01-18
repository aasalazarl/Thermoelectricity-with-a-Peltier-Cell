function internalResistancAsymp
% Scattered plot of values of the internal resistance of the cell

% Import data
% data = csvread(filename,Ri,Ci,[Ri Ci Rf Cf]) - Starts counting from 0
% naming: Ri means "measurements of the internal resistance of trial i".
%         Ni means "measurement number for the values of the internal
%         resistance in Ri"
R3 = csvread('internalResistanceAsympT3.csv',1,5,[1 5 36 5]);
N3 = csvread('internalResistanceAsympT3.csv',1,4,[1 4 36 4]) ;
R4 = csvread('internalResistanceAsympT4.csv',1,5,[1 5 84 5]);
N4 = csvread('internalResistanceAsympT4.csv',1,4,[1 4 84 4]);
R5 = csvread('internalResistanceAsympT5.csv',1,5,[1 5 91 5]);
N5 = csvread('internalResistanceAsympT5.csv',1,4,[1 4 91 4]);
R6 = csvread('internalResistanceAsympT6.csv',1,5,[1 5 90 5]);
N6 = csvread('internalResistanceAsympT6.csv',1,4,[1 4 90 4]);
R7 = csvread('internalResistanceAsympT7.csv',1,5,[1 5 85 5]);
N7 = csvread('internalResistanceAsympT7.csv',1,4,[1 4 85 4]);

figure(1)
plot(N3,R3,'.')
title('internalResistance, trial3')
xlabel('Measurement number');ylabel('Internal resistance (ohms)');
legend('dataPoints')
saveas(gcf,'internalResistanceAsympT3','eps')
saveas(gcf,'internalResistanceAsympT3','png')

figure(2)
plot(N4,R4,'.')
title('internalResistance, trial4')
xlabel('Measurement number');ylabel('Internal resistance (ohms)');
legend('dataPoints')
saveas(gcf,'internalResistanceAsympT4','eps')
saveas(gcf,'internalResistanceAsympT4','png')

figure(3)
plot(N5,R5,'.')
title('internalResistance, trial5')
xlabel('Measurement number');ylabel('Internal resistance (ohms)');
legend('dataPoints')
saveas(gcf,'internalResistanceAsympT5','eps')
saveas(gcf,'internalResistanceAsympT5','png')

figure(4)
plot(N6,R6,'.')
title('internalResistance, trial6')
xlabel('Measurement number');ylabel('Internal resistance (ohms)');
legend('dataPoints')
saveas(gcf,'internalResistanceAsympT6','eps')
saveas(gcf,'internalResistanceAsympT6','png')

figure(5)
plot(N7,R7,'.')
title('internalResistance, trial7')
xlabel('Measurement number');ylabel('Internal reistance (ohms)');
legend('dataPoints')
saveas(gcf,'internalResistanceAsympT7','eps')
saveas(gcf,'internalResistanceAsympT7','png')