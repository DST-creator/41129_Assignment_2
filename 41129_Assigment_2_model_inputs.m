clear, clc

%% Preparation
ang = 0:15:180;
n = 0:length(ang)-1;
ang_dict = dictionary(ang, n);

T = 9.72;        %[s] Time period
nu = 1.14e-6;    %[m^2/s] - kinematic viscosity
h_m = 0.145;     %[m] - model height

case_tbl = table([.23; .68; 2.0; 2.0], ...
                 [0; 0; 0; 8.4e-4],...
                 [7.2e4; 6.3e5; 5.4e6; 5.4e6], ...
                 [Inf; Inf; Inf; 3700],...
                 [0;0;0;0],...
                 'VariableNames', {'U_0m', 'k_s', 'Re', 'a_ks', 'f_w'},...
                 'RowNames', {'1';'2';'3';'4'});

clear ang n

%% Task 1
for i = 1:height(case_tbl)
    Re_i = case_tbl{i, 'Re'};
    k_s_i = case_tbl{i, 'k_s'};
    if Re_i <= 1.5e5  %Laminar flow
        case_tbl{i, 'f_w'} = 2./sqrt(Re_i); %Eq. 5.59
    elseif Re_i >=1.5e5 %Turbulent flow
        if k_s_i==0 %smooth wall 
            case_tbl{i, 'f_w'} = .035./(Re_i.^.16); %Eq. 5.60
        else  %Rough wall
            case_tbl{i, 'f_w'} = ...
                exp(5.5.*(case_tbl{i, 'a_ks'}).^(-.16)-6.7); %Eq. 5.69
        end
    else  %Transitional flow
        case_tbl{i, 'f_w'} = .005;  %Eq. 5.61
    end    
end
clear i Re_i k_s_i

case_tbl.('U_fm') = sqrt(case_tbl.('f_w')./2).*case_tbl.('U_0m');

%% Task 2
case_tbl.('dy_max') = nu./case_tbl.('U_fm');  %Eq. 9.49

case_tbl.('k_s_plus') = zeros(height(case_tbl),1) + .1;
i_rough = case_tbl.('k_s')~=0;
case_tbl{i_rough, 'k_s_plus'} = case_tbl{i_rough, 'k_s'}...
                                .*case_tbl{i_rough, 'U_fm'}./nu; %Eq. from Assignment
if case_tbl{i_rough, 'k_s_plus'} >70
    disp ("Rough case is hydraulically rough")
end

i_smooth = case_tbl.('k_s')==0;
case_tbl{i_smooth, 'k_s'} = .1.*nu./case_tbl{i_smooth, 'U_fm'}; %Eq. from Assignment

clear i_rough i_smooth

%% Task 3
% WBL(c).omegat(n)