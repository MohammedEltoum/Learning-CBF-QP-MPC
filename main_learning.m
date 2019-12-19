%% ESE 680, Learning CBFs from Expert Demonstrations - Two aircrafts collision avoidance
clear all; close all; clc
setenv('SNOPT_LICENSE','/Users/haiminhu/licenses/snopt7.lic');

global v w b c db dc h f g

%% Parameters
v = 0.1;
w = 1;
Ds = 0.5;
ts = 0.3;


%% Define ZCBF
% x = [px_a py_a theta_a px_b py_b theta_b]
b  = @(px,theta) px - (v/w)*sin(theta);
c  = @(py,theta) py + (v/w)*cos(theta);
db = @(x) b(x(1),x(3)) - b(x(4),x(6));
dc = @(x) c(x(2),x(3)) - c(x(5),x(6));

A1 = @(x) db(x)^2 + dc(x)^2 + 2*(v/w)^2 - 2*(v/w)^2*cos(x(3)-x(6));
A2 = @(x) compute_A2(x);
h  = @(x) A1(x) - A2(x) - Ds^2;


%% Define dynamics
f = @(x) zeros(6,1);
g = @(x) [cos(x(3)),0,0,0; sin(x(3)),0,0,0; 0,1,0,0; 0,0,cos(x(6)),0; 0,0,sin(x(6)),0; 0,0,0,1];


%% Closed loop simulation
N_rollout = 1;
% Input (x', u')
X = [];
% Output (h, dhdt, dhdx')
y = [];

for kk = 1:N_rollout

close all
    
% Generate initial condition
% R = rand+0.6;
R = 0.6;

% angle = pi/6 + 2*pi*((kk-1)/N_rollout);
% angle = rand*2*pi;
angle = 0;
temp = R*exp(angle*1j);

x  = [ real(temp);imag(temp);pi+angle+0.1; -real(temp);-imag(temp);angle+0.1];
if real(temp)>= 0
    xg = [ -2;0;pi+angle; 2;0;angle];
else
    xg = [ 2;0;angle; -2;0;pi+angle];
end

x_traj = x;
u_traj = [];
x_MPC  = {};
N_sim = 25;
count = 0;

for t = 1:N_sim
    
    % Visualization
    figure(1)
    L = 0.3;
    % agent 1
    px1 = x(1);
    py1 = x(2);
    theta1 = x(3);
    px1_t = px1+(L*cos(theta1));
    py1_t = py1+(L*sin(theta1));
    quiver(px1,py1,px1_t-px1,py1_t-py1,0,'r','LineWidth',1.5,'MaxHeadSize',2.0)
    grid on
    hold on
    % agent 2
    px2 = x(4);
    py2 = x(5);
    theta2 = x(6);
    px2_t = px2+(L*cos(theta2));
    py2_t = py2+(L*sin(theta2));
    quiver(px2,py2,px2_t-px2,py2_t-py2,0,'b','LineWidth',1.5,'MaxHeadSize',2.0)
    xlim([-2,2])
    ylim([-2,2])
    grid on

    % Nonlinear tracking MPC as the nominal controller
    [ x_NMPC, u_NMPC, ~ ] = CFTOC_NMPC(x, xg, f, g, 10, ts);
    x_MPC{end+1} = x_NMPC;
    u_hat = u_NMPC(:,1);
    
    % Numerical gradient of h(x) w.r.t. x
%     dhdx = num_grad(x);
    [dhdx, h_x] = retrieve_grad(x);
    dhdx = dhdx';
    
    % Solve CBF-QP
    u = sdpvar(4,1); % u = [va wa vb wb]
    objective = norm(u-u_hat)^2;
    constraints = [ dhdx'*(f(x)+g(x)*u) >= -(h(x))^2;   % CBF condition !!TODO: extended class K function alpha
                    0.1 <= u(1) <= 1;
                    -1  <= u(2) <= 1;
                    0.1 <= u(3) <= 1;
                    -1  <= u(4) <= 1]; 
    options = sdpsettings('verbose',0,'solver','quadprog','usex0',0,'cachesolvers',1 );
    sol = optimize(constraints, objective, options);
    uOpt = value(u);
    uDev = sqrt(value(objective));
    
    % Return feasibility
    if sol.problem ~= 0
        sol.info
    end
    
    % Evolve the system
    x = x + ts*(f(x) + g(x)*uOpt);

    % Collect data
    x_traj = [x_traj x];
    u_traj = [u_traj uOpt];
    if uDev >= 0.01
        X = [X; [x'   uOpt'] ];
        y = [y; [h_x dhdx'*(f(x)+g(x)*uOpt) dhdx'] ];
        count = 0;
    else
        count = count + 1;
    end

    % Report
    D = sqrt((x(1)-x(4))^2 + (x(2)-x(5))^2);
    fprintf('Timestep: %d, Distance: %f, uDev: %f\n', [t, D, uDev]);
    
%     if count >= 5
%         break
%     end
    
end

fprintf('Iteration: %d finished\n', kk);


end



%% Functions

function xNext = RK4(x, u, ts, n)
    ts = ts/n;
    for rki = 1: n
        k1 = ode_nominal(x, u);
        k2 = ode_nominal(x + ts/2*k1, u);
        k3 = ode_nominal(x + ts/2*k2, u);
        k4 = ode_nominal(x + ts*k3, u);
        x  = x + (ts/6)*(k1 + 2*k2 + 2*k3 + k4);
    end
    xNext = x;
end

function dxdt = ode_nominal(x, u)
    global f g
    dxdt = f(x) + g(x)*u;
end

function Kt = succ_lqr(x)
% LQR with successive linearization
    Ac = zeros(6,6);
    Bc = [cos(x(3)),0,0,0; sin(x(3)),0,0,0; 0,1,0,0; 0,0,cos(x(6)),0; 0,0,sin(x(6)),0; 0,0,0,1];
    try
        [Kt,~,~] = lqr(Ac,Bc,eye(6),eye(4));
    catch
        Kt = [];
    end
end


% function dhdx = num_grad(x)
%     global h
%     eps = 1e-10;
%     dhdx = [];
%     for i = 1:length(x)
%         Eps = zeros(length(x),1);
%         Eps(i) = eps;
%         dhdx = [dhdx; (h(x+Eps)-h(x-Eps))/(2*eps)];
%     end
% end


function A2 = compute_A2(x)
    global v w db dc
    [A21,theta21] = phasor_add(-2*db(x)*(v/w), 2*db(x)*(v/w), x(3)+pi/2, x(6)+pi/2);
    [A22,theta22] = phasor_add(-2*dc(x)*(v/w), 2*dc(x)*(v/w), x(3), x(6));
    [A2,~]        = phasor_add(A21, A22, theta21, theta22);
end


function [A3,theta3] = phasor_add(A1,A2,theta1,theta2)
%Compute the phasor addition of two cosines
%   Reference: https://en.wikipedia.org/wiki/Phasor#Addition
A3 = sqrt( (A1*cos(theta1)+A2*cos(theta2))^2 + (A1*sin(theta1)+A2*sin(theta2))^2 );

% !!! theta3 needs to be in [-pi/2,3pi/2]
if A1*cos(theta1)+A2*cos(theta2)==0
    theta3 = sign(A1*sin(theta1)+A2*sin(theta2))*pi/2;
elseif A1*cos(theta1)+A2*cos(theta2)>0
    theta3 = atan((A1*sin(theta1)+A2*sin(theta2))/(A1*cos(theta1)+A2*cos(theta2)));
else
    theta3 = atan((A1*sin(theta1)+A2*sin(theta2))/(A1*cos(theta1)+A2*cos(theta2))) + pi;
end

end

%     % Design a nominal controller using LQR
%     if t > 1
%         Kt_old = Kt;
%     end
%     Kt = succ_lqr(x);
%     if isempty(Kt)
%         Kt = Kt_old;
%     end
%     u_hat = -Kt*x;


%% Attribution
% Haimin Hu, Alex Robey, ESE 680, UPenn, Fall 2019