function results = compute_outflows_and_dP(Omega, vel, pres, outletPatches, inletPatch)
% Computes inlet flux, outlet fluxes, mean inlet/outlet pressures and pressure difference.
% Mean pressures follow notes:
%   p_in_mean  = (∫_{inlet} p ds) / (length(inlet))
%   p_out_mean = (∫_{outlets} p ds) / (total outlet length)

b = Omega.b; T = Omega.t; X = Omega.x;

Q_out = zeros(numel(outletPatches),1);
Q_in  = 0;

Pin_int = 0; Lin = 0;
Pout_int = zeros(numel(outletPatches),1);
Lout     = zeros(numel(outletPatches),1);


for k = 1:size(b,1)

    e = b(k,1); nA = b(k,2); nB = b(k,3); patch = b(k,4);

    isOutlet = any(patch == outletPatches);
    isInlet  = (patch == inletPatch);
    if ~(isOutlet || isInlet), continue; end

    te = T(e,:); Xe = X(te,:);
    ia = find(te == nA, 1); ib = find(te == nB, 1);
    if isempty(ia) || isempty(ib)
        error('Boundary nodes not found in Omega.t for element %d', e);
    end
    face = local_face_triangle(ia, ib);

    xA = X(nA,:); xB = X(nB,:);
    tvec = xB - xA;
    L = norm(tvec,2);
    if L == 0, continue, end

    % outward normal 
    n1 = [ tvec(2), -tvec(1) ] / L;
    xC = mean(Xe,1); xM = 0.5*(xA + xB);
    if dot(n1, xC - xM) < 0, n = n1; else, n = -n1; end

    % ds quadrature weights
    fw = Omega.e.f{face}.gw;
    wds = fw * (L / master_edge_len(face));

    % edge velocity 
    fyV = vel.e.f{face}.y;
    tve = vel.t(e,:);
    v1l = vel.u(2*(tve-1) + 1);
    v2l = vel.u(2*(tve-1) + 2);
    vx = fyV * v1l; vy = fyV * v2l;
    vn = vx*n(1) + vy*n(2);

    % edge pressure
    fyP = pres.e.f{face}.y;
    tpe = pres.t(e,:);
    pqp = fyP * pres.u(tpe);

    flux = sum(wds .* vn);
    pint = sum(wds .* pqp);
    ell  = sum(wds);

if isOutlet
    idx = find(outletPatches == patch, 1);
    Q_out(idx) = Q_out(idx) + flux;

    % Store pressure integral and length per outlet
    Pout_int(idx) = Pout_int(idx) + pint;
    Lout(idx)     = Lout(idx)     + ell;
else

        Q_in = Q_in + flux;
        Pin_int = Pin_int + pint;
        Lin     = Lin     + ell;
    end
end

results.outletPatches = outletPatches(:);
results.Q_out = Q_out;
results.Q_in  = Q_in;
results.mass_balance = Q_in + sum(Q_out);
results.outletFractions = abs(Q_out) / sum(abs(Q_out));

results.p_in_mean  = Pin_int  / Lin;
% Mean pressure on each outlet separately
results.p_out_mean_i = Pout_int ./ Lout;

% Also keep the total outlet-mean (length-weighted across all outlets)
results.p_out_mean = sum(Pout_int) / sum(Lout);

% Pressure drops per outlet and total
results.deltaP_i = results.p_in_mean - results.p_out_mean_i;
results.deltaP   = results.p_in_mean - results.p_out_mean;


end

function f = local_face_triangle(i,j)
pair = sort([i j]);
if isequal(pair,[1 2]), f=1;
elseif isequal(pair,[1 3]), f=2;
elseif isequal(pair,[2 3]), f=3;
else, error('Invalid triangle edge (%d,%d)', i, j);
end
end

function L = master_edge_len(face)
if face == 3, L = sqrt(2); else, L = 1.0; end
end
