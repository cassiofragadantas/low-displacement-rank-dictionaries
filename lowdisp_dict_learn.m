function [D,D_structured,X] = lowdisp_dict_learn(params)

% D_structured : before column normalization

D = params.initdict;
Y = params.data;

%% Parameter setting

% iteration count %

if (isfield(params,'iternum'))
  iternum = params.iternum;
else
  iternum = 10;
end

% structure type %

if (~isfield(params,'struct_type'))
    % Default structure type is Toeplitz (if not specified)
    params.struct_type = 'toeplitz';
end
    
switch lower(params.struct_type)
    case 'toeplitz'
        F1 = diag(ones(1,size(D,1)-1),-1);
        F2 = diag(ones(1,size(D,2)-1),-1)';
%         F2(end,1) = 1; % Using Z1 instead of Z0 (see Takahashi2013)
    case 'hankel'
        F1 = diag(ones(1,size(D,1)-1),-1);
        F2 = diag(ones(1,size(D,2)-1),-1);
%         F2(1,end) = 1; % Using Z1 instead of Z0 (see Takahashi2013)
    otherwise
        error('Invalid structure type specified');
end

%% Sparse Coding parameter setting
CODE_SPARSITY = 1;
CODE_ERROR = 2;

MEM_LOW = 1;
MEM_NORMAL = 2;
MEM_HIGH = 3;

%%%%% parse input parameters %%%%%

ompparams = {'checkdict','off'};

% coding mode %

if (isfield(params,'codemode'))
  switch lower(params.codemode)
    case 'sparsity'
      codemode = CODE_SPARSITY;
      thresh = params.Tdata;
    case 'error'
      codemode = CODE_ERROR;
      thresh = params.Edata;
    otherwise
      error('Invalid coding mode specified');
  end
elseif (isfield(params,'Tdata'))
  codemode = CODE_SPARSITY;
  thresh = params.Tdata;
elseif (isfield(params,'Edata'))
  codemode = CODE_ERROR;
  thresh = params.Edata;

else
  error('Data sparse-coding target not specified');
end


% max number of atoms %

if (codemode==CODE_ERROR && isfield(params,'maxatoms'))
  ompparams{end+1} = 'maxatoms';
  ompparams{end+1} = params.maxatoms;
end


% memory usage %

if (isfield(params,'memusage'))
  switch lower(params.memusage)
    case 'low'
      memusage = MEM_LOW;
    case 'normal'
      memusage = MEM_NORMAL;
    case 'high'
      memusage = MEM_HIGH;
    otherwise
      error('Invalid memory usage mode');
  end
else
  memusage = MEM_NORMAL;
end

% omp function %

if (codemode == CODE_SPARSITY)
  ompfunc = @omp;
else
  ompfunc = @omp2;
end


% data norms %

XtX = []; XtXg = [];
if (codemode==CODE_ERROR && memusage==MEM_HIGH)
  XtX = sum(Y.^2);
end

err = zeros(1,iternum);
gerr = zeros(1,iternum);

if (codemode == CODE_SPARSITY)
  errstr = 'RMSE';
else
  errstr = 'mean atomnum';
end

%% Sparse Coding
G = [];
if (memusage >= MEM_NORMAL)
    G = D'*D;
end

if (memusage < MEM_HIGH)
  X = ompfunc(D,Y,G,thresh,ompparams{:});

else  % memusage is high

  if (codemode == CODE_SPARSITY)
    X = ompfunc(D'*Y,G,thresh,ompparams{:});

  else
    X = ompfunc(D'*Y,XtX,G,thresh,ompparams{:});
  end

end

%%%%%%%%%%%%%%%%%%

%% Optimization %%
%%%%%%%%%%%%%%%%%%
lambda = params.alpha*norm(Y);

%% Ajuste automatico para step size
step = 1e-10/norm(D);
u = 1e-10;

found = false;

iter = 0;
clear lbound rbound
while ~found
    Z = (D - F1*D*F2)/u; 
    D_hat = zeros(size(D));
    % Min dispD
    [U, S, V] = svd(D_hat - F1*D_hat*F2 - Z ,'econ');
    diagS = diag(S);
    svp = length(find(diagS > lambda/u));
    temp_dispD = U(:, 1:svp) * diag(diagS(1:svp) - lambda/u) * V(:, 1:svp)';
    % Min D_hat
    grad1 = D_hat - F1*D_hat*F2 - temp_dispD - Z + F1'*(temp_dispD - D_hat + F1*D_hat*F2 + Z)*F2';
    grad2 = (D_hat*X - Y)*X.';
    norm1 = norm(step*(u*grad1 + grad2), 'fro');
    D_hat = D_hat - step*(u*grad1 + grad2);
    grad1 = D_hat - F1*D_hat*F2 - temp_dispD - Z + F1'*(temp_dispD - D_hat + F1*D_hat*F2 + Z)*F2';
    grad2 = (D_hat*X - Y)*X.';
    norm2 = norm(step*(u*grad1 + grad2), 'fro');

    if norm2 < norm1 % Converges
       lbound = step;
       if exist('rbound','var')
           step = (lbound + rbound)/2;
       else
           step = step*1e3;
       end
    else % Diverges
        rbound = step;
        if exist('lbound','var')
           step = (lbound + rbound)/2;
        else
           step = step/1e3;
        end
    end

    if exist('lbound','var') && exist('rbound','var') 
        if (rbound - lbound)/lbound < 1e-6
            found = true;
            if params.sigma > 50
                step = step/4;
            else
                step = step/3*(size(Y,2)/40000);
            end
        end
    end
    iter = iter + 1;
end

u = 1e7;

%% Alternating Optimization
success = false;
fprintf('Iteration:              ')
while ~success
    try
        Z = (D - F1*D*F2)/u;  % It is faster if  these variables are not reseted every iteration
        D_hat = zeros(size(D));
        tic
        for k = 1:iternum
            fprintf(repmat('\b',1,13));
            fprintf('%4d / %4d  ',k,iternum); 
            
            %% Dictionary update step
            tol = 1e-4*norm(D,'fro');
            if k == iternum, tol = 1e-7*norm(D,'fro'); end % Better accuracy on the last iteration
            converged = false;
            iter = 1;

            % ADMM Loop
            while ~converged
                % Min dispD
                [U, S, V] = svd(D_hat - F1*D_hat*F2 - Z ,'econ');
                diagS = diag(S);
                svp = length(find(diagS > lambda/u));
                temp_dispD = U(:, 1:svp) * diag(diagS(1:svp) - lambda/u) * V(:, 1:svp)';
                % Min D_hat
                grad1 = D_hat - F1*D_hat*F2 - temp_dispD - Z + F1'*(temp_dispD - D_hat + F1*D_hat*F2 + Z)*F2';
                grad2 = (D_hat*X - Y)*X.';
                temp_D_hat = D_hat - step*(u*grad1 + grad2);
                D_hat = temp_D_hat;

                dispD = temp_dispD;


                temp_Z = Z + dispD - D_hat + F1*D_hat*F2;

                % stop Criterion    
                if norm(temp_Z - Z, 'fro') < tol
                    converged = true;
                    % disp(['Total nº of iterations: ' num2str(iter)]);
                end

                Z = temp_Z;

                iter = iter+1;
                if iter > 1e6, error('ALM did not converge, probably due to bad parameter setting. Retrying...'),end
            end

            D = normc(D_hat);
            
            %% Sparse coding step
            G = [];
            if (memusage >= MEM_NORMAL)
                G = D'*D;
            end

            if (memusage < MEM_HIGH)
              X = ompfunc(D,Y,G,thresh,ompparams{:});

            else  % memusage is high

              if (codemode == CODE_SPARSITY)
                X = ompfunc(D'*Y,G,thresh,ompparams{:});

              else
                X = ompfunc(D'*Y,XtX,G,thresh,ompparams{:});
              end

            end     
        end
        total_time = toc; fprintf('    Elapsed time: %.1fs\n',total_time);
        success = true;
    catch err
        disp(err.message); disp('Trying smaller step.')
        step = step/2
    end 
end

D_structured = D_hat;
end