function [Re] = navier_stokes_mass_eqn(e,testsp, teste, vars);

v(:,1) = vars.vel.u(vars.vel.dm * (vars.vel.t(e,:)-1) + 1);
v(:,2) = vars.vel.u(vars.vel.dm * (vars.vel.t(e,:)-1) + 2);
% evaulating the weighted sum of divergence of velocity at quadrature points 
divv = vars.vele.dy(:,:,1) * v(:,1) + vars.vele.dy(:,:,2) * v(:,2);
% Getting local row / local column sizes 
ne = size(testsp.t,2);
Re = zeros(ne, 1); 
% Pressure space has one degree of freedom per node
for i = 1:ne

    Re(i) = dot(teste.gw, divv .* teste.y(:,i));
end
