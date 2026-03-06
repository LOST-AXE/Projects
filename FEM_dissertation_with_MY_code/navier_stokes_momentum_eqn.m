function [Re] = navier_stokes_momentum_eqn(e,testsp, teste, vars);

  % Problem parameters 
  rho = 1e-3;   
  mu = 4e-3; 
  

  vl(:,1) = vars.vel.u (vars.vel.dm * (vars.vel.t(e,:)-1) + 1);
  vl(:,2) = vars.vel.u (vars.vel.dm * (vars.vel.t(e,:)-1) + 2);
  pl(:,1) = vars.pres.u(vars.pres.t(e,:)); 

  % evaulating the weighted sum of velocity and pressure variables at quadrature points 
  v = [vars.vele.y(:,:) * vl(:,1) vars.vele.y(:,:) * vl(:,2)];
  p = vars.prese.y(:,:) * pl(:,1);

  % Velocity gradients, fixed indices 
  dv1_dx1 = vars.vele.dy(:,:,1) * vl(:,1);  
  dv1_dx2 = vars.vele.dy(:,:,2) * vl(:,1);  
  dv2_dx1 = vars.vele.dy(:,:,1) * vl(:,2); 
  dv2_dx2 = vars.vele.dy(:,:,2) * vl(:,2);  
  
  % Getting local row / local column sizes 
  ne=size(testsp.t,2);
  Re = zeros(testsp.dm * ne, 1);

  for i = 1:ne 
    % getting the local element residual indices ordered 
    vei = (testsp.dm * (i - 1) + 1):(testsp.dm * i);
    
    % compute advection
    advection1 = rho * ( v(:,1).*dv1_dx1 + v(:,2).*dv1_dx2);
    advection2 = rho * ( v(:,1).*dv2_dx1 + v(:,2).*dv2_dx2);
   % Computing final terms
    Re(vei) =  [dot(teste.gw, advection1 .* teste.y(:,i) ...
                     + mu * (dv1_dx1 .* teste.dy(:,i,1) + dv1_dx2 .* teste.dy(:,i,2)) ...
                     - p .* teste.dy(:, i, 1));
               
               dot(teste.gw, advection2 .* teste.y(:,i) ...
                     + mu * (dv2_dx1 .* teste.dy(:,i,1) + dv2_dx2 .* teste.dy(:, i, 2)) ...
                     - p .* teste.dy(:, i,2))];
  end
end