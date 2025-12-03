%%
%testing commits
%%

% ------------------------------------------------------------------------- 
% MCTS VS HUMAN
% -------------------------------------------------------------------------

clc
clear
clf
close all

%mcts_vs_human(1, [], 'time')
mcts_vs_human([], 50000, 'iteration')

%% 
% -------------------------------------------------------------------------
% MCTS VS RANDOM
% -------------------------------------------------------------------------

clc
clear
clf
close all

n = 1000;
iters = [5 15 25 100 1000];
for i = 1:length(iters)
    results(i,:) = mcts_vs_random(n, [], iters(i), 'iteration');
    results_sum(i,:) = cumsum(results(i,:));
    sums(i) = sum(results(i,:) == 1);
end

figure;
plot(1:n, results_sum,'-','LineWidth',2)
xlabel('$Games \; Played$','FontSize',16,'Interpreter','latex')
ylabel('$Cumulative \; Sum \; of \; Games \; Won$','FontSize',16,'Interpreter','latex')
grid on
labels = arrayfun(@(k) sprintf('$iterations: %d$', k), iters, ...
                  'UniformOutput', false);
legend(labels{:},'Fontsize',16,'interpreter','latex','location','best');

%%
% -------------------------------------------------------------------------
% MCTS PLAY VS MCTS
% -------------------------------------------------------------------------

clc
clear
clf
close all
%{
n = 100;

results = mcts_vs_mcts(n, 10,10);
results_sum = cumsum(results);
sums = sum(results == 1);

figure(1)
plot(1:n,results_sum,'-k','LineWidth',2)
xlabel('$Games \; Played$','FontSize',16,'Interpreter','latex')
ylabel('$Cumulative \; Sum \; of \; Games \; Won$','FontSize',16,'Interpreter','latex')
grid on
%}

n = 100;
for i = 1:50
    results(i,:) = mcts_vs_mcts(n,21,19);
    results_sum(i,:) = cumsum(results(i,:));
    sums(i) = nnz(results(i,:) == 1);
    
    figure(1)
    hold on
    plot(1:n,results_sum(i,:),'-','LineWidth',1)
end
xlabel('$Games \; Played$','FontSize',16,'Interpreter','latex')
ylabel('$Cumulative \; Sum \; of \; Games \; Won$','FontSize',16,'Interpreter','latex')
grid on
means = mean(results_sum,1);
plot(1:n,means,'-k','LineWidth',2)

%%

function results = mcts_vs_random(numGames, thinkTime, iterations, mode)
% mcts_vs_random  Play MCTS vs a purely random opponent.
%
% results(g) = +1  -> MCTS wins game g
% results(g) = -1  -> Random wins game g
% results(g) =  0  -> Draw
%
% Inputs:
%   numGames   : number of games to play
%   thinkTime  : time in seconds per MCTS move (used if mode = 'time')
%   iterations : number of MCTS iterations per move (used if mode = 'iteration')
%   mode       : 'time' or 'iteration'

    if nargin < 4
        error('Usage: mcts_vs_random(numGames, thinkTime, iterations, ''time'' or ''iteration'')');
    end

    mode = lower(string(mode));
    if mode ~= "time" && mode ~= "iteration"
        error('mode must be ''time'' or ''iteration''.');
    end

    % Convention: MCTS is -1, Random is +1
    MCTS   = -1;
    RANDOM =  1;

    results = zeros(numGames, 1);

    for g = 1:numGames
        fprintf('Current game: %d / %d\n', g, numGames);

        board = new_board();
        % Choose who starts; here RANDOM starts
        turn = RANDOM;

        while true
            w = winner(board);
            if w ~= 0 || is_full(board)
                break;
            end

            if turn == MCTS
                % MCTS move, using either time or iterations
                if mode == "time"
                    move = mcts_best_move(board, MCTS, [], thinkTime, 1.1, []);
                else % "iteration"
                    move = mcts_best_move(board, MCTS, iterations, [], 1.1, []);
                end
            else
                % Random move
                moves = legal_moves(board);
                if isempty(moves)
                    break;
                end
                move = moves(randi(numel(moves)));
            end

            if isempty(move)
                break;
            end

            board = make_move(board, move, turn);
            turn  = -turn;  % switch player
        end

        w = winner(board);
        if w == MCTS
            results(g) = 1;
        elseif w == RANDOM
            results(g) = -1;
        else
            results(g) = 0;
        end
    end
end

function results = mcts_vs_mcts(numGames, iterationsP1, iterationsP2)
% mcts_vs_mcts  Play MCTS vs MCTS in Connect-4.
%
% results(g) = +1  -> Player 1 (MCTS1) wins game g
% results(g) = -1  -> Player 2 (MCTS2) wins game g
% results(g) =  0  -> Draw
%
% Inputs:
%   numGames    : number of games to play
%   iterationsP1: number of MCTS iterations per move for Player 1
%   iterationsP2: number of MCTS iterations per move for Player 2
%
% Notes:
%   - Player 1 is encoded as -1
%   - Player 2 is encoded as +1
%   - We alternate who starts: odd games -> Player 1 starts,
%                               even games -> Player 2 starts.

    if nargin < 2
        error('Usage: mcts_vs_mcts(numGames, iterationsP1, [iterationsP2])');
    end
    if nargin < 3
        % If not given, use same iterations for both players
        iterationsP2 = iterationsP1;
    end

    % Convention: P1 = -1, P2 = +1 (same style as your other code)
    P1 = -1;
    P2 =  1;

    results = zeros(numGames, 1);

    for g = 1:numGames
        fprintf('Current game: %d / %d\n', g, numGames);

        board = new_board();

        % Alternate starting player for fairness
        if mod(g, 2) == 1
            turn = P1;
        else
            turn = P2;
        end

        while true
            w = winner(board);
            if w ~= 0 || is_full(board)
                break;
            end

            % MCTS move for the current player, with its own iteration budget
            if turn == P1
                move = mcts_best_move(board, P1, iterationsP1, [], 1.1, []);
            else
                move = mcts_best_move(board, P2, iterationsP2, [], 1.1, []);
            end

            if isempty(move)
                % If MCTS fails to return a move, abort this game as draw
                break;
            end

            board = make_move(board, move, turn);
            turn  = -turn;  % switch player
        end

        % Determine outcome
        w = winner(board);
        if w == P1
            results(g) = 1;
        elseif w == P2
            results(g) = -1;
        else
            results(g) = 0;
        end
    end
end

function mcts_vs_human(thinkTime, iterations, mode)
% mcts_vs_human  Human vs MCTS AI Connect-4.
%
% Inputs:
%   thinkTime  : time in seconds per MCTS move (used if mode = 'time')
%   iterations : number of MCTS iterations per move (used if mode = 'iteration')
%   mode       : 'time' or 'iteration'

    if nargin < 3
        error('Usage: mcts_vs_human(thinkTime, iterations, ''time'' or ''iteration'')');
    end

    mode = lower(string(mode));
    if mode ~= "time" && mode ~= "iteration"
        error('mode must be ''time'' or ''iteration''.');
    end

    board = new_board();
    HUMAN =  1;
    AI    = -1;
    turn  = HUMAN;   % change to AI if you want AI to start

    fprintf('You are X, AI is O. Enter column (1-7). ''q'' to quit.\n');
    print_board(board);

    while true
        if winner(board) ~= 0 || is_full(board)
            break;
        end

        if turn == HUMAN
            moves = legal_moves(board);  % columns 1..7
            while true
                s = input('Your move (1-7): ', 's');
                s = strtrim(s);
                if any(strcmpi(s, {'q','quit','exit'}))
                    return;
                end

                col = str2double(s);
                if ~isnan(col) && isfinite(col) && mod(col,1)==0 && col >= 1 && col <= 7
                    if any(moves == col)
                        break;
                    else
                        fprintf('Column %d is not legal. Legal: ', col);
                        fprintf('%d ', moves);
                        fprintf('\n');
                    end
                else
                    fprintf('Please enter an integer 1-7.\n');
                end
            end

            board = make_move(board, col, HUMAN);
            print_board(board);
            turn = AI;

        else
            fprintf('AI thinking with MCTS (%s mode)...\n', mode);

            % MCTS move, using either time or iterations
            if mode == "time"
                move = mcts_best_move(board, AI, [], thinkTime, 1.1, []);
            else  % "iteration"
                move = mcts_best_move(board, AI, iterations, [], 1.1, []);
            end

            if isempty(move)
                break;
            end

            board = make_move(board, move, AI);
            fprintf('AI plays column %d\n', move);
            print_board(board);
            turn = HUMAN;
        end
    end

    w = winner(board);
    if w == HUMAN
        fprintf('You win!\n');
    elseif w == AI
        fprintf('AI wins.\n');
    else
        fprintf('Draw.\n');
    end
end

%%
% ------------------------------------------------------------------------- 
% GAME SETUP
% -------------------------------------------------------------------------

function board = new_board()
    ROWS = 6;
    COLS = 7;
    board = zeros(ROWS, COLS);   % 0 empty, +1 human, -1 AI
end

function print_board(board)
    [ROWS, COLS] = size(board);

    % Column headers
    fprintf('   \n     ');
    for c = 1:COLS
        fprintf('%3d', c);   % columns 1–7, fixed width
    end
    fprintf('\n');

    % Rows
    for r = 1:ROWS
        fprintf('%2d ', r);  % row index 1–6, fixed width
        for c = 1:COLS
            v = board(r,c);
            if v == 0
                ch = '~';
            elseif v == 1
                ch = 'X';
            else
                ch = 'O';
            end
            fprintf('%3s', ch);  % each cell fixed width
        end
        fprintf('\n');
    end

    fprintf('\n');
end

function moves = legal_moves(board)
    % Returns columns (1..7) that are not full
    moves = find(board(1,:) == 0);
end

function nb = make_move(board, col, player)
    % Drop a piece in column "col" (1..7) for "player"
    [ROWS, ~] = size(board);
    r = ROWS;
    while r >= 1 && board(r,col) ~= 0
        r = r - 1;
    end
    nb = board;
    nb(r,col) = player;
end

function w = winner(board)
    % Returns 1 or -1 if someone won; 0 otherwise
    [ROWS, COLS] = size(board);
    dirs = [1 0; 0 1; 1 1; 1 -1];

    for r = 1:ROWS
        for c = 1:COLS
            p = board(r,c);
            if p == 0
                continue;
            end
            for d = 1:size(dirs,1)
                dr = dirs(d,1);
                dc = dirs(d,2);
                cnt = 0;
                rr = r;
                cc = c;
                while rr >= 1 && rr <= ROWS && cc >= 1 && cc <= COLS && board(rr,cc) == p
                    cnt = cnt + 1;
                    if cnt == 4
                        w = p;
                        return;
                    end
                    rr = rr + dr;
                    cc = cc + dc;
                end
            end
        end
    end
    w = 0;
end

function tf = is_full(board)
    tf = all(board(1,:) ~= 0);
end

% -------------------------------------------------------------------------
% MCTS NODE REPRESENTATION
% -------------------------------------------------------------------------

function node = create_node(board, player, parent_idx, move)
    node.board    = board;
    node.player   = player;      % player to move at this node
    node.parent   = parent_idx;  % parent index in the nodes array (0 for root)
    node.move     = move;        % move (column) that led here from parent
    node.children = [];          % indices of children in nodes array
    node.untried  = legal_moves(board);  % columns yet to expand
    node.N        = 0;           % visits
    node.W        = 0.0;         % total reward from root player's perspective
end

function child_idx = uct_select_child_idx(nodes, node_idx, Cp)
    % Upper Confidence bound: Q + Cp * sqrt(ln(Np) / Nc)
    parentN = nodes(node_idx).N;
    lnNp    = log(max(parentN, 1));
    best_score  = -1e18;
    child_idx   = [];

    child_indices = nodes(node_idx).children;
    for k = 1:numel(child_indices)
        ci = child_indices(k);
        child = nodes(ci);
        Q = child.W / (child.N + 1e-9);
        U = Cp * sqrt(lnNp / (child.N + 1e-9));
        s = Q + U;
        if s > best_score
            best_score = s;
            child_idx  = ci;
        end
    end
end

% -------------------------------------------------------------------------
% ROLLOUT POLICY AND SIMULATION
% -------------------------------------------------------------------------

function move = policy_rollout_move(board, player)
    % Lightly biased rollout:
    %  1) take immediate win if available
    %  2) block opponent's immediate win
    %  3) prefer center/near-center
    %  4) otherwise random legal
    moves = legal_moves(board);
    if isempty(moves)
        move = [];
        return;
    end
    %{
    % 1) win now
    for k = 1:numel(moves)
        m = moves(k);
        b2 = make_move(board, m, player);
        if winner(b2) == player
            move = m;
            return;
        end
    end

    % 2) block opponent's immediate win
    opp = -player;
    for k = 1:numel(moves)
        m = moves(k);
        b2 = make_move(board, m, player);
        opp_moves = legal_moves(b2);
        for j = 1:numel(opp_moves)
            o = opp_moves(j);
            if winner(make_move(b2, o, opp)) == opp
                if any(moves == o)
                    move = o;
                    return;
                end
            end
        end
    end

    % 3) prefer center/near-center (columns 1..7)
    % original pref in 0-based: [3,2,4,1,5,0,6] -> 1-based [4,3,5,2,6,1,7]
    pref = [4 3 5 2 6 1 7];
    for k = 1:numel(pref)
        if any(moves == pref(k))
            move = pref(k);
            return;
        end
    end
    %}
    % 4) fallback random
    move = moves(randi(numel(moves)));
end

function result = rollout(board, player, root_player)
    % Random (biased) playout until terminal
    b = board;
    p = player;

    while true
        w = winner(b);
        if w ~= 0
            result = get_result(w, root_player);
            return;
        end
        if is_full(b)
            result = 0.0;
            return;
        end

        m = policy_rollout_move(b, p);
        if isempty(m)
            result = 0.0;
            return;
        end

        b = make_move(b, m, p);
        p = -p;
    end
end

function r = get_result(winner_flag, root_player)
    if winner_flag == 0
        r = 0.0;
    elseif winner_flag == root_player
        r = 1.0;
    else
        r = -1.0;
    end
end

% -------------------------------------------------------------------------
% TREE EXPANSION & BACKPROP
% -------------------------------------------------------------------------

function [nodes, child_idx] = expand_node(nodes, node_idx)
    node = nodes(node_idx);
    m = node.untried(end);
    node.untried(end) = [];

    child_board = make_move(node.board, m, node.player);
    child_idx   = numel(nodes) + 1;

    nodes(child_idx) = create_node(child_board, -node.player, node_idx, m);
    node.children    = [node.children, child_idx];

    nodes(node_idx)  = node;
end

function nodes = backpropagate(nodes, node_idx, reward)
    % Reward is from root player's perspective (no sign flip)
    n = node_idx;
    while n ~= 0
        nodes(n).N = nodes(n).N + 1;
        nodes(n).W = nodes(n).W + reward;
        n = nodes(n).parent;
    end
end

%--------------------------------------------------------------------------
% TOP-LEVEL MCTS DRIVER
%--------------------------------------------------------------------------

function move = mcts_best_move(board, player, iterations, time_limit, Cp, seed)
    % Return best column (1..7) for `player` using MCTS (iterations or time budget).

    if nargin < 3 || isempty(iterations)
        iterations = 6000;
    end
    if nargin < 4
        time_limit = [];
    end
    if nargin < 5 || isempty(Cp)
        Cp = 1.41;
    end
    if nargin < 6
        seed = [];
    end
    if ~isempty(seed)
        rng(seed);
    end

    % Initialize root node
    nodes(1) = create_node(board, player, 0, []); %#ok<AGROW>
    root_idx = 1;

    start = tic;
    it = 0;

    while true
        if ~isempty(time_limit)
            if toc(start) >= time_limit
                break;
            end
        else
            if it >= iterations
                break;
            end
        end
        it = it + 1;

        % 1) Selection
        node_idx = root_idx;
        while isempty(nodes(node_idx).untried) && ~isempty(nodes(node_idx).children)
            node_idx = uct_select_child_idx(nodes, node_idx, Cp);
        end

        % 2) Expansion
        if ~isempty(nodes(node_idx).untried)
            [nodes, node_idx] = expand_node(nodes, node_idx);
        end

        % 3) Simulation (rollout)
        result = rollout(nodes(node_idx).board, nodes(node_idx).player, player);

        % 4) Backpropagation
        nodes = backpropagate(nodes, node_idx, result);
    end

    % Pick child with max visits
    if isempty(nodes(root_idx).children)
        moves = legal_moves(board);
        if isempty(moves)
            move = [];
        else
            move = moves(randi(numel(moves)));
        end
        return;
    end

    child_indices = nodes(root_idx).children;
    Ns = arrayfun(@(idx) nodes(idx).N, child_indices);
    [~, best_idx] = max(Ns);
    move = nodes(child_indices(best_idx)).move;
end
