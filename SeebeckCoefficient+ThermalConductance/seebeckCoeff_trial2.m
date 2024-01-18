function seebeckCoeff_trial2
% Determination of the Seebeck Coefficient

% Import data
% data = csvread(filename,Ri,Ci,[Ri Ci Rf Cf]) - Starts counting from 0
DelT = csvread('seebeckCoeff_trial2_decreasingVh.csv',1,5,[1 5 109 5])
uncDelT = csvread('seebeckCoeff_trial2_decreasingVh.csv',1,11,[1 11 109 11]);
Vs = csvread('seebeckCoeff_trial2_decreasingVh.csv',1,2,[1 2 109 2]);
uncVs = csvread('seebeckCoeff_trial2_decreasingVh.csv',1,9,[1 9 109 9]);

% Perform weighted (vertical unc only) linear fit
c = linefiterr(DelT,Vs,uncVs)
x = linspace(min(DelT),max(DelT),100);
y = linspace(min(Vs),max(Vs),100);

% Plot data
figure(1)
hold on
errorbar(DelT,Vs,uncVs,'.')
plot(x,y,'k')
xlabel('{\Delta}T (kelvin)'); ylabel('V_{s} (volts)');
legend('dataPoints','linearFit: y = 1.0509x + 7.7780','location','best')
title('Seebeck coefficient S from V_{s} vs. {\Delta}T')
hold off

% Save figure(1) = graph1
% saveas(gcf,filename,format)
%saveas(gcf,"seebeckCoefficient_trial2_wErrBars",'png')

figure(2)
hold on
plot(DelT,Vs,'.')
xlabel('{\Delta}T (kelvin)'); ylabel('V_{s} (volts)');
legend('dataPoints','location','best')
%title('Seebeck coefficient S from V_{s} vs. {\Delta}T')
hold off

% Save figure(2) = graph2
%saveas(gcf,"seebeckCoefficient_trial2_noErrBars",'png')
saveas(gcf,"seebeckCoefficient_trial2_noErrBars_4Report",'png')