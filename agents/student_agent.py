import time
import numpy as np
from copy import deepcopy
from agents.agent import Agent
from store import register_agent
import random

# import logging

# logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
# logger = logging.getLogger(__name__)


# Important: you should register your agent with a name
@register_agent("student_agent")
class StudentAgent(Agent):
    """
    Example of an agent which takes random decisions
    """

    def set_barrier(self, r, c, dir, chess_board, barrier):
        # Set the barrier to True
        chess_board[r, c, dir] = barrier
        # Set the opposite barrier to True
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = barrier

    def check_valid_pos(self, p0_pos, end_pos, p1_pos, chess_board):
        """
        Check if the step the agent takes is valid (reachable and within max steps).

        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """
        # Endpoint already has barrier or is boarder
        if np.array_equal(p0_pos, end_pos):
            return True

        # Get position of the adversary
        adv_pos = p1_pos

        # BFS
        state_queue = [(p0_pos, 0)]
        visited = {tuple(p0_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self.max_step:
                break
            for dir, move in enumerate(self.moves):
                if chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached
    
    def check_valid_dir(self, end_pos, barrier_dir, chess_board):
        r, c = end_pos
        return not chess_board[r, c, barrier_dir]

    def check_valid_step(self, p0_pos, end_pos, barrier_dir, p1_pos, chess_board):
        """
        Check if the step the agent takes is valid (reachable and within max steps).

        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """
        return self.check_valid_dir(end_pos, barrier_dir, chess_board) and self.check_valid_pos(p0_pos, end_pos, p1_pos, chess_board)

    def check_endgame(self, chess_board, p0_pos, p1_pos):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        # Union-Find
        father = dict()
        for r in range(self.board_size):
            for c in range(self.board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(self.board_size):
            for c in range(self.board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(self.board_size):
            for c in range(self.board_size):
                find((r, c))
        p0_r = find(tuple(p0_pos))
        p1_r = find(tuple(p1_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
        if player_win >= 0:
            # logging.info(
            #     f"Game ends! Player {self.player_names[player_win]} wins having control over {win_blocks} blocks!"
            # )
            pass
        else:
            # logging.info("Game ends! It is a Tie!")
            pass
        return True, p0_score, p1_score

    def check_boundary(self, pos):
        r, c = pos
        return 0 <= r < self.board_size and 0 <= c < self.board_size

    def find_new_move(self, my_pos, adv_pos, chess_board, step_remaining = 0):
        my_pos = np.array(my_pos)
        op_pos = np.array(adv_pos)

        moves = list(enumerate(self.moves))
        random.shuffle(moves)

        for (dir, move) in moves:
            # no more steps, just pick any direction
            if step_remaining == 0:
                r, c = my_pos
                dirs = list(range(4))
                random.shuffle(dirs)
                for dir in dirs:
                    if chess_board[r, c, dir]:
                        continue
                    else:
                        return my_pos, dir

            # check if walking into a wall
            r, c = my_pos
            if chess_board[r, c, dir]:
                continue

            # found new pos, so make another move
            new_pos = my_pos + move
            if self.check_boundary(new_pos) and not np.array_equal(new_pos, op_pos):
                return self.find_new_move(new_pos, op_pos, chess_board, step_remaining - 1)


    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.autoplay = True
        self.moves = [np.array([-1, 0]), np.array([0, 1]), np.array([1, 0]), np.array([0, -1])]
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}

    def step(self, chess_board, my_pos, adv_pos, max_step):
        # Moves (Up, Right, Down, Left)
        self.max_step = max_step
        self.board_size = chess_board.shape[0]

        start_time = time.time_ns()
        end_time = start_time + 1 * 10**9

        new_moves = []

        for _ in range(100):
            step = random.randint(0, max_step)
            new_pos, new_dir = self.find_new_move(my_pos, adv_pos, chess_board, step)
            r, c = new_pos
            self.set_barrier(r, c, new_dir, chess_board, True)
            end_game, my_score, adv_score = self.check_endgame(chess_board, new_pos, adv_pos)
            self.set_barrier(r, c, new_dir, chess_board, False)
            new_moves.append((new_pos, new_dir, my_score - adv_score))
        
        new_moves = sorted(new_moves, key=lambda x: x[-1])
        my_pos, my_dir, _ = new_moves[-1]
        return my_pos, my_dir
