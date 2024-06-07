"""
@Project: code
@File: agent.py
@Description:
@Author: xiaoyihong
@Email: xiaoyh20@mails.tsinghua.edu.cn
@Date: 2024/6/7
"""
import random
import numpy as np


class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((self.env.h, self.env.w, env.n_actions))
        self.epsilon = 0.1 # exploration rate
        self.alpha = 0.1 # learning rate
        self.gamma = 0.9 # discount factor

    def train_action(self, state, epsilon):
        """
        Choose action based on epsilon-greedy policy while training
        """
        x_pixel, y_pixel = state
        x = int((x_pixel - 50) // 100)
        y = int((y_pixel - 50) // 100)
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            return np.argmax(self.q_table[x, y])

    def test_action(self, state):
        """
        Choose action based on greedy policy while testing
        """
        x_pixel, y_pixel = state
        x = int((x_pixel - 50) // 100)
        y = int((y_pixel - 50) // 100)
        return np.argmax(self.q_table[x, y])

    def update_qtable(self, s, a, r, ns):
        """
        Update Q-table
        """
        x_pixel, y_pixel = s
        nx_pixel, ny_pixel = ns
        x = int((x_pixel - 50) // 100)
        y = int((y_pixel - 50) // 100)
        nx = int((nx_pixel - 50) // 100)
        ny = int((ny_pixel - 50) // 100)

        q_cur = self.q_table[x, y, a]
        max_q_next = self.q_table[nx, ny].max()
        # iterative update
        self.q_table[x, y, a] = q_cur + self.alpha * (r + self.gamma * max_q_next - q_cur)

    def train(self, episodes):
        """
        Train the agent
        :param episodes: number of episodes
        """
        self.env.label.config(text="Training the agent")
        for e in range(episodes):
            num_step = 0
            s = self.env.reset() # initial state
            done = False
            cur_qtable = self.q_table.copy()
            # explore the environment until done in each episode
            while not done:
                self.env.render()
                if e < 30:
                    epsilon = 0.1
                    a = self.train_action(s, epsilon=0.1)
                elif e >= 30 and e < 100:
                    epsilon = 0.05
                    a = self.train_action(s, epsilon=0.05)
                elif e >= 100 and e < 200:
                    epsilon = 0.02
                    a = self.train_action(s, epsilon=0.02)
                elif e >= 200:
                    epsilon = 0.0
                    a = self.train_action(s, epsilon=0.0)

                ns, r, done = self.env.step(a)
                self.update_qtable(s, a, r, ns)
                s = ns
                num_step += 1
                if done:
                    self.env.label.config(text=f"Episode {e} finished after {num_step} steps (epsilon={epsilon})")
                    print(f"Episode {e} finished after {num_step} steps (epsilon={epsilon})")
            # check convergence
            if np.allclose(cur_qtable, self.q_table):
                self.env.label.config(text="Q-Learning result is Converged!")
                print("Q-Learning result is Converged!")
                break

    def show_q_table(self):
        """
        Show the learned Q-table
        """
        h, w, n_actions = self.q_table.shape

        header = f'| {"x":^6} | {"y":^6} |' + ' | '.join([f'{action:^6}' for action in range(n_actions)]) + ' |'
        print(header)
        print('-' * len(header))
        for x in range(h):
            for y in range(w):
                row = f'| {x:^6} | {y:^6} |' + ' | '.join([f'{self.q_table[x,y,a]:^6.2f}' for a in range(n_actions)]) + ' |'
                print(row)
        print('-' * len(header))
    def test(self):
        """
        Test the learned policy
        """
        self.env.label.config(text="Testing the learned policy")
        s = self.env.reset()
        done = False
        while not done:
            self.env.render()
            a = self.test_action(s)
            ns, r, done = self.env.step(a)
            s = ns
            if done:
                self.env.label.config(text="Test finished with the learned policy.")
                break