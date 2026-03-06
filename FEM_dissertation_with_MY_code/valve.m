% valve.m  -- Vascular Flow Project (FINAL)
clear; clc;

quad  = 10;
etype = 'triangle';

load("vesseltree/vesseltree0.mat")

% Solver params
nonlinear_tol = 1e-6;
iter = 0;
residual_vector_norm = 1;

% Make basis objects
Omega2.e = fem_get_basis(Omega2.p, quad, etype);
Omega.e  = fem_get_basis(Omega.p,  quad, etype);

% Velocity space (quadratic)
vel = Omega2;
vel.name = 'vel';
vel.u = zeros(vel.dm * size(vel.x,1), 1);
n_vel = size(vel.u,1);

% Pressure space (linear)
pres = Omega;
pres.name = 'pres';
pres.dm = 1;
pres.u = zeros(size(pres.x,1), 1);
n_pres = size(pres.u,1);

% Outlet lengths (for reporting)
outlets = [2 3 4 5];
Lpatch = zeros(size(outlets));
for k = 1:size(Omega.b,1)
    nA = Omega.b(k,2);
    nB = Omega.b(k,3);
    patch = Omega.b(k,4);
    idx = find(outlets == patch, 1);
    if isempty(idx), continue; end
    Lpatch(idx) = Lpatch(idx) + norm(Omega.x(nB,:) - Omega.x(nA,:), 2);
end
disp('Total outlet boundary lengths:')
for i = 1:numel(outlets)
    fprintf('Patch %d length = %.6f\n', outlets(i), Lpatch(i));
end

% Newton-Raphson loop
while (iter == 0) || (residual_vector_norm > nonlinear_tol)

    % Store unknowns in vars 
    vars.vel  = vel;
    vars.pres = pres;

    % Residual: momentum
    R1 = fem_assemble_block_residual(@navier_stokes_momentum_eqn, Omega, vel, vars);

    % Apply velocity Dirichlet BCs using vel.b patches
    for i = 1:size(vel.b,1)
        patch = vel.b(i,5);

        % Outlets: do nothing 
        if any(patch == [2 3 4 5])
            continue
        end

        % Inlet: v = (-50, 0) on patch 1
        if patch == 1
            for j = 2:4
                nn = vel.dm * (vel.b(i,j) - 1);
                R1(nn+1:nn+2) = -[ -50 - vel.u(nn+1); 0 - vel.u(nn+2) ];
            end
        end

        % Wall: v = (0,0) on patch 6
        if patch == 6
            for j = 2:4
                nn = vel.dm * (vel.b(i,j) - 1);
                R1(nn+1:nn+2) = -[ 0 - vel.u(nn+1); 0 - vel.u(nn+2) ];
            end
        end
    end

    % Residual: mass
    R2 = fem_assemble_block_residual(@navier_stokes_mass_eqn, Omega, pres, vars);

    % Pressure gauge fix
    R2(1) = 0;

    % Global residual and norm
    R = [R1; R2];
    residual_vector_norm = norm(R,2);
    disp(['Current residual (iter=' num2str(iter) '): ' num2str(residual_vector_norm)])

    if residual_vector_norm < nonlinear_tol
        break
    end

    % Jacobian blocks 
    A = fem_assemble_block_matrix_perturbation(@navier_stokes_momentum_eqn, Omega, vel,  vel,  vars);
    B = fem_assemble_block_matrix_perturbation(@navier_stokes_momentum_eqn, Omega, vel,  pres, vars);
    C = fem_assemble_block_matrix_perturbation(@navier_stokes_mass_eqn,     Omega, pres, vel,  vars);
    D = sparse(size(pres.u,1), size(pres.u,1));

    % Gauge fix in Jacobian
    C(1,:) = 0;
    D(1,1) = 1;

    % Dirichlet for patch 1 and 6
    for i = 1:size(vel.b,1)
        patch = vel.b(i,5);
        if (patch == 1 || patch == 6)
            for j = 2:4
                nn = vel.dm * (vel.b(i,j) - 1);
                A(nn+1:nn+2,:) = 0;
                A(nn+1:nn+2, nn+1:nn+2) = eye(2);
                B(nn+1:nn+2,:) = 0;
            end
        end
    end

    % Global Jacobian and Newton update
    J = [A B; C D];
    U = J \ R;

    vel.u  = vel.u  - U(1:n_vel);
    pres.u = pres.u - U(n_vel+1:end);

    iter = iter + 1;
    disp(' ')
end

% Plot velocity
figure(1);
quiver(vel.x(:,1), vel.x(:,2), vel.u(1:2:end), vel.u(2:2:end))
title('Velocity field'); axis equal tight; grid on;
xlabel("X-axis (horizontal): Distance (mm)")
ylabel("Y-axis (Vertical): Distance (mm)")

% Plot pressure
figure(2);
trisurf(pres.t(:,1:3), pres.x(:,1), pres.x(:,2), pres.u, ...
        'Facecolor','interp','LineStyle','none')
title('Pressure field');
view(2);

% Create colorbar and store handle
cb = colorbar;

% Add label to colorbar
ylabel(cb, 'Pressure (g/(mm • s^2 ))', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('x (mm)'); ylabel('y (mm)');
axis equal tight;

% Post-processing 
results = compute_outflows_and_dP(Omega, vel, pres, [2 3 4 5], 1);

fprintf('--- FLOW RESULTS ---\n');
fprintf('Inlet flux Q_in = %.6f\n', results.Q_in);
fprintf('Sum outlet flux = %.6f\n', sum(results.Q_out));
fprintf('Mass balance (Q_in + sum(Q_out)) = %.6e\n', results.mass_balance);

fprintf('\nOutflow fractions (abs-normalized):\n');
for i = 1:numel(results.outletPatches)
    fprintf('Patch %d: %.4f\n', results.outletPatches(i), results.outletFractions(i));
end

fprintf('\nMean inlet pressure  = %.6f\n', results.p_in_mean);
fprintf('Mean outlet pressure = %.6f\n', results.p_out_mean);
fprintf('Total pressure drop = %.6f\n', results.deltaP);
fprintf('\nOutlet mean pressures and pressure drops (per patch):\n');
for i = 1:numel(results.outletPatches)
    fprintf('Patch %d: p_out_mean = %.6f,  DeltaP = %.6f\n', ...
        results.outletPatches(i), results.p_out_mean_i(i), results.deltaP_i(i));
end


disp('Outlet fluxes (signed):');
for i = 1:numel(results.outletPatches)
    fprintf('Patch %d: Q = %.6f\n', results.outletPatches(i), results.Q_out(i));
end

%% =======================
%  MESH RESOLUTION METRICS

nElem = size(Omega.t,1);          % number of triangles
nNodes = size(Omega.x,1);         % number of nodes

nVelDOF  = length(vel.u);         % P2 velocity DOFs
nPresDOF = length(pres.u);        % P1 pressure DOFs
nTotalDOF = nVelDOF + nPresDOF;

% Compute characteristic mesh size h
edges = zeros(3*nElem,1);
k = 1;
for e = 1:nElem
    nodes = Omega.t(e,:);
    edges(k)   = norm(Omega.x(nodes(1),:) - Omega.x(nodes(2),:)); k = k+1;
    edges(k)   = norm(Omega.x(nodes(2),:) - Omega.x(nodes(3),:)); k = k+1;
    edges(k)   = norm(Omega.x(nodes(3),:) - Omega.x(nodes(1),:)); k = k+1;
end

h_mean = mean(edges);
h_min  = min(edges);
h_max  = max(edges);

fprintf('\n--- MESH METRICS ---\n');
fprintf('Elements           = %d\n', nElem);
fprintf('Nodes              = %d\n', nNodes);
fprintf('Velocity DOFs (P2) = %d\n', nVelDOF);
fprintf('Pressure DOFs (P1) = %d\n', nPresDOF);
fprintf('Total DOFs         = %d\n', nTotalDOF);
fprintf('Mean mesh size h   = %.6f\n', h_mean);
fprintf('Min edge length    = %.6f\n', h_min);
fprintf('Max edge length    = %.6f\n', h_max);
