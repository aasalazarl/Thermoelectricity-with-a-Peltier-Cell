function thermalCtrial2
% Plots data behaviour in time for calculation of thermal conductance

% Import data
% data = csvread(filename,Ri,Ci,[Ri Ci Rf Cf]) - Starts counting from 0
DelT = csvread('seebeckCoeff_trial2_decreasingVh.csv',1,5,[1 5 109 5]);
Qhpower = csvread('seebeckCoeff_trial2_decreasingVh.csv',1,12,[1 12 109 12]);

% Plot data
plot(DelT,Qhpower,'.')
xlabel('{\Delta}T (kelvin)'); ylabel('Heat power Q_{h-power} (watts)');
legend('dataPoints','location','best')

saveas(gcf,'thermalCtrial2_evoInTime.png','png')

