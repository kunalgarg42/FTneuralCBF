clc
clearvars
close all

solv_quadprog = 1;

options = optimoptions('quadprog','Display','off','ConstraintTolerance',1e-12);
ops = sdpsettings('verbose',0,'solver','gurobi','cachesolvers',1,'usex0',1);


num = 50000;
dt = 0.001;

x = zeros(4,num);
u = zeros(2,num);

x(:,1) = [10; 0.05; 0.05; 0.01];
m = 23;

cla = 2*pi;
Cd0 = 0.02;
k1 = 0.05;
k2 = 0.05;

cm0 = -0.1;
cma = -0.1;
cmq = -9;
cmd = -1;
I = 4.51;
grav = 9.8;

rho = 1.225;
S = 0.99;
c = 0.33;
Vr = zeros(num,1);
gr = Vr;
time = 0:dt:dt*num;
i = 1;

syms a

u_opt = sdpvar(2,1);
gamma1 = sdpvar(1);
gamma2 = sdpvar(1);
delta = sdpvar(1);
z_opt = [u_opt;gamma1;gamma2;delta];


while i < num+1
    if i<num/5
        Vr(i) = 21;
    elseif i<2*num/5
        Vr(i) = 23;
    elseif i<4*num/5
        Vr(i) = 19;
    else
        Vr(i) = 25;
    end
    
    if i<num/5
        gr(i) = 0.1;
    elseif i<2*num/5
        gr(i) = 15*pi/180;
    elseif i<4*num/5
        gr(i) = 30*pi/180;
    else
        gr(i) = 0*pi/180;
    end
    i = i+1;
end
Ftr = zeros(num,1);
alphar = zeros(num,1);

syms Ft alpha
f1 = -1/2*rho*Vr(1)^2*S*(Cd0+cla*alpha+cla^2*alpha^2)+Ft*cos(alpha)-m*grav*sin(gr(1));
f2 = 1/2*rho*Vr(1)^2*S*(cla*alpha)+Ft*sin(alpha)-m*grav*cos(gr(1));
sol = vpasolve([f1; f2], Ft, alpha,[0 Inf; -2*pi 2*pi]);
Ftr(1:num/5) = double(sol.Ft)*ones(num/5,1);
alphar(1:num/5) = double(sol.alpha)*ones(num/5,1);

f1 = -1/2*rho*Vr(num/5+1)^2*S*(Cd0+cla*alpha+cla^2*alpha^2)+Ft*cos(alpha)-m*grav*sin(gr(num/5+1));
f2 = 1/2*rho*Vr(num/5+1)^2*S*(cla*alpha)+Ft*sin(alpha)-m*grav*cos(gr(num/5+1));
sol = vpasolve([f1; f2], Ft, alpha,[0 Inf; -2*pi 2*pi]);
Ftr(num/5+1:2*num/5) = double(sol.Ft)*ones(num/5,1);
alphar(num/5+1:2*num/5) = double(sol.alpha)*ones(num/5,1);

f1 = -1/2*rho*Vr(2*num/5+1)^2*S*(Cd0+cla*alpha+cla^2*alpha^2)+Ft*cos(alpha)-m*grav*sin(gr(2*num/5+1));
f2 = 1/2*rho*Vr(2*num/5+1)^2*S*(cla*alpha)+Ft*sin(alpha)-m*grav*cos(gr(2*num/5+1));
sol = vpasolve([f1; f2], Ft, alpha,[0 Inf; -2*pi 2*pi]);
Ftr(2*num/5+1:4*num/5) = double(sol.Ft)*ones(2*num/5,1);
alphar(2*num/5+1:4*num/5) = double(sol.alpha)*ones(2*num/5,1);

f1 = -1/2*rho*Vr(end)^2*S*(Cd0+cla*alpha+cla^2*alpha^2)+Ft*cos(alpha)-m*grav*sin(gr(end));
f2 = 1/2*rho*Vr(end)^2*S*(cla*alpha)+Ft*sin(alpha)-m*grav*cos(gr(end));
sol = vpasolve([f1; f2], Ft, alpha,[0 Inf; -2*pi 2*pi]);
Ftr(4*num/5+1:end) = double(sol.Ft)*ones(num/5,1);
alphar(4*num/5+1:end) = double(sol.alpha)*ones(num/5,1);

for iter = 1:1
    Q = 10*eye(5);
    Q(3,3) = 10000;
    Q(4,4) = 1000;
    Q(5,5) = 1000;
    F = 2*diag(Q)';
    F(1) = 0;
    F(2) = 0;
    %     F(5) = 0;
    cost = z_opt'*Q*z_opt+F*z_opt;
    
    for i = 1:num-1
        %         i
        if  mod(i,1000) == 0
            i
        end
        alpha = x(3,i)-x(2,i);
        theta = x(3,i);
        v = x(1,i);
        q = x(4,i);
        gamma = x(2,i);
        
        Cl = cla*alpha;
        Cd = Cd0 + k1*Cl + k2*Cl^2;
        Cmf = cm0 + cma*alpha+cmq*q;
        Cmg = cmd;
        
        L = 1/2*rho*v^2*S*Cl;
        D = 1/2*rho*v^2*S*Cd;
        
        Mf = 1/2*rho*v^2*S*c*Cmf;
        Mg = 1/2*rho*v^2*S*c*Cmg;
        
        
        V1 = 1/2*10*(v-Vr(i))^2;
        gradV1 = 10*[v-Vr(i); 0; 0; 0];
        
        thetar = gr(i) + alphar(i);
        V2 = 0.5*10*((theta-thetar)^2 + (q+theta-thetar)^2);
        
        gradV2  = 10*[0;0; 2*theta-2*thetar+q; q+theta-thetar];
        
        
        %         if i>1
        %             Ft = u(1,i-1);
        %         else
        %             Ft = 0;
        %             %         Ft = 370.8124;
        %             %         a0 = 0.1581;
        %         end
        
        %         %     falpha = 1/2*rho*v^2*S*cla*a+Ft*sin(a)-m*grav*cos(gr(i));
        %     a0 = vpasolve(falpha);
        %
        %     a0 = double(a0);
        
        %         a0 = m*grav*cos(gr(i))/(1/2*rho*v^2*S*cla+Ft);
        %         falpha = 1/2*rho*v^2*S*cla*a0+Ft*sin(a0)-m*grav*cos(gr(i));
        %         k = 1;
        %         while abs(falpha)>0.1 && k<1000
        %             gradfalpha = 1/2*rho*v^2*S*cla+Ft*cos(a0);
        %             a0 = a0-0.1*falpha/gradfalpha;
        %             falpha = 1/2*rho*v^2*S*cla*a0+Ft*sin(a0)-m*grav*cos(gr(i));
        %             k = k+1;
        %         end
        %         k
        %         if a0>0.1
        %             a0 = vpasolve(1/2*rho*v^2*S*cla*a+Ft*sin(a)-m*grav*cos(gr(i)));
        %             a0 = double(a0);
        %         end
        %         z1 = gamma-gr(i);
        %         z2  = theta-gr(i)-a0;
        %         z3 = q;
        %         V2 = 10/2*(z1^2+(z2-z1)^2+(z2+z3)^2);
        
        %         Cmzf = cm0 + cma*(z2-z1+a0)+cmq*q;
        %
        %         Mfz = 1/2*rho*v^2*S*c*Cmzf;
        %         fz = [1/m/v*(1/2*rho*v^2*S*cla*(z2-z1+a0));
        %             z3;
        %             Mfz/I];
        %         gz = [0 0;0 0;0 Mg/I];
        
        
        
        %         h1 = alpha-alphar(i)-0.01;
        %         h2 = 0.01+alphar(i)-alpha;
        
        %         h = max(h1,h2);
        %         if h1>h2
        %             gradH = [0;-1; 1; 0];
        %         else
        %             gradH = [0; 1; -1; 0];
        %         end
        
        h = 0.5*20*(alpha-alphar(i))^2;
        
        gradH = 20*[0; alpha-alphar(i); -alpha+alphar(i); 0];
        
        f = [-1/m*D-grav*sin(gamma);
            1/m/v*L-grav/v*cos(gamma);
            q;
            Mf/I];
        
        g = [1/m*cos(alpha) 0; 1/m*sin(alpha)/v 0; 0 0; 0 Mg/I];
        
        LfV1 = gradV1'*f;
        LgV1 = gradV1'*g;
        LfV2 = gradV2'*f;
        LgV2 = gradV2'*g;
        LfH = gradH'*f;
        LgH = gradH'*g;
        
        A1 = [LgV1 -V1 0 0];
        A2 = [LgV2 0 -V2 0];
        A3 = [LgH 0 0 h];
        
        if V1>0.1
            B1 = -20*pi*V1^0.8-20*pi*V1^1.2-LfV1;
        else
            B1 = -LfV1;
        end
        if V2>0.0001
            B2 = -50*pi*V2^0.8-50*pi*V2^1.2-LfV2;
        else
            B2 = -LfV2;
        end
        if h>0.00001
            B3 = -50*pi*h^0.8-50*pi*h^1.2-LfH;
        else
            B3 = -LfH;
        end
        %         A_ineq = [A1; A2; A3; 1 0 0 0 0; -1 0 0 0 0; 0 1 0 0 0; 0 -1 0 0 0];
        %         B_ineq = [B1; B2; B3; 400; 0; 1; 1];
        A_ineq = [A2; A3; A1;  1 0 0 0 0; -1 0 0 0 0; 0 1 0 0 0; 0 -1 0 0 0];
        B_ineq = [B2; B3; B1; 400; 0; 5; 5];
        if i>1
            F(1) = -Q(1,1)*u(1,i-1);
            F(2) = -Q(2,2)*u(2,i-1);
        end
        %     cost = cost_t+F(1)*z_opt(1)+F(2)*z_opt(2);
        %
        if solv_quadprog == 1
            k_opt =  quadprog(Q, F, A_ineq,B_ineq,[],[],[],[],[],options);
        else
            constraint = A_ineq*z_opt<= B_ineq;
            solution = solvesdp(constraint,cost+F(1)*z_opt(1)+F(2)*z_opt(2),ops);
            k_opt = double(u_opt);
        end
        %         
        
        
        u(:,i) = k_opt(1:2);
        x(:,i+1) = x(:,i) + (f+g*u(:,i))*dt;
        
    end
    
end

%% plotting

figure(1)
plot(x(1,1:i))
hold on
plot(Vr(1:i))

figure(2)
plot(x(2,1:i))
hold on
plot(gr(1:i))


figure(3)
plot(x(3,1:i));
hold on
% plot(alphar(1:i));
plot(alphar(1:i)+gr(1:i))

figure(4)
plot(u(1,1:i))
hold on
plot(Ftr(1:i))

deltar = (-cm0-cma*alphar)/cmd;
figure(5)
plot(u(2,1:i))
hold on
plot(deltar(1:i))
% hold off