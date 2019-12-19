function [ xOpt, uOpt, JOpt ] = CFTOC_NMPC(x0, xg, f, g, N, ts)
% Solve the CFTOC for the NMPC problem

% Define system dimensions
NX = 6;
NU = 4;
Q  = eye(NX);
R  = eye(NU);
P  = 10*eye(NX);

% Define state variables
x = sdpvar(NX,N+1);

% Define decision variables
u = sdpvar(NU,N);

% Define objective, constraints and terminal set

% Terminal cost
objective = (x(:,N+1)-xg)'*P*(x(:,N+1)-xg);
for i = 1:N
    objective = objective + (x(:,i)-xg)'*Q*(x(:,i)-xg) + u(:,i)'*R*u(:,i);
end

constraints = [ x(:,1) == x0 ];
for i = 1:N
    % dynamics
    constraints = [constraints x(:,i+1) == x(:,i) + ts*(f(x(:,i)) + g(x(:,i))*u(:,i));
                    0.1 <= u(1,i) <= 1;
                    -1  <= u(2,i) <= 1;
                    0.1 <= u(3,i) <= 1;
                    -1  <= u(4,i) <= 1];
%     constraints = [constraints x(:,i+1) == RK4(x(:,i), u(:,i), ts, 5, f, g)];
end

% Set options for YALMIP and solver
options = sdpsettings('verbose',0,'solver','snopt','usex0',0,'cachesolvers',1 );
% Solve
sol = optimize(constraints, objective, options);

% Retrieve solutions for plotting
xOpt = value(x);
uOpt = value(u);
JOpt = value(objective);

% Return feasibility
if sol.problem ~= 0
    sol.info
end

function xNext = RK4(x, u, ts, n, f, g)
    ts = ts/n;
    for rki = 1: n
        k1 = ode_nominal(x, u, f, g);
        k2 = ode_nominal(x + ts/2*k1, u, f, g);
        k3 = ode_nominal(x + ts/2*k2, u, f, g);
        k4 = ode_nominal(x + ts*k3, u, f, g);
        x  = x + (ts/6)*(k1 + 2*k2 + 2*k3 + k4);
    end
    xNext = x;
end

function dxdt = ode_nominal(x, u, f, g)
    dxdt = f(x) + g(x)*u;
end

end