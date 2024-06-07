import numpy as np
import time
import sys
import tkinter as tk
from tkinter import PhotoImage

from agent import QLearningAgent


UNIT = 100   # 迷宫中每个格子的像素大小
MAZE_H = 6  # 迷宫的高度（格子数）
MAZE_W = 6  # 迷宫的宽度（格子数）


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        # 构建迷宫
        self.h = MAZE_H
        self.w = MAZE_W
        self.title('Q-Learnign')
        self.geometry('{0}x{1}'.format((self.h + 1) * UNIT, (self.w + 1)* UNIT))
        self.button_frame = tk.Frame(self)
        self.button_frame.pack(side=tk.BOTTOM)
        # self._build_maze()
        self.label = tk.Label(self, text='', font=('Arial', 15), width=50, height=2)
        self.label.pack(side=tk.BOTTOM)
        # 动作空间
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        # 智能体
        self.agent = None

    def set_agent(self, agent):
        self.agent = agent

    def _build_maze(self):
        """
        迷宫初始化
        """
        self.canvas = tk.Canvas(self, bg='white', height=self.h * UNIT, width=self.w * UNIT)
        for r in range(0, (self.h + 1) * UNIT, UNIT):
            self.canvas.create_line(r, 0, r, self.h * UNIT, fill='black')
        for c in range(0, (self.w + 1) * UNIT, UNIT):
            self.canvas.create_line(0, c, self.w * UNIT, c, fill='black')

        origin_pos = np.array([UNIT/2, UNIT/2])
        self.stone_img = PhotoImage(file="imgs/obstacles.png")
        self.stone1 = self.canvas.create_image(origin_pos[0] + UNIT * 4, origin_pos[1] + UNIT, image=self.stone_img)
        self.stone2 = self.canvas.create_image(origin_pos[0] + UNIT, origin_pos[1] + UNIT * 4, image=self.stone_img)
        self.stone3 = self.canvas.create_image(origin_pos[0] + UNIT * 4, origin_pos[1] + UNIT * 3, image=self.stone_img)
        self.stone4 = self.canvas.create_image(origin_pos[0] + UNIT * 3, origin_pos[1] + UNIT * 4, image=self.stone_img)

        self.player_img = PhotoImage(file="imgs/character.png")
        self.player = self.canvas.create_image(origin_pos[0], origin_pos[1], image=self.player_img)

        self.candy_img = PhotoImage(file="imgs/candy.png")
        self.candy = self.canvas.create_image(origin_pos[0] + 4 * UNIT, origin_pos[1] + 4 * UNIT, image=self.candy_img)

        self.canvas.pack()

        # 功能按钮
        self.quit_button = tk.Button(self.button_frame, text='quit game', command=self.quit)
        self.quit_button.pack(side=tk.LEFT)

        self.test_button = tk.Button(self.button_frame, text='test policy', command=self.agent.test)
        self.test_button.pack(side=tk.LEFT)

    def update_label(self, text):
        self.label.config(text=text)
        
    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.player)
        
        origin_pos = np.array([UNIT/2, UNIT/2])
        self.player = self.canvas.create_image(origin_pos[0], origin_pos[1], image=self.player_img)
        return self.canvas.coords(self.player)

    def step(self, action):
        s_pixel = self.canvas.coords(self.player)
        move_pixel = np.array([0, 0])
        if action == 0:  # 向上移动
            if s_pixel[1] > UNIT:
                move_pixel[1] -= UNIT
        elif action == 1:  # 向下移动
            if s_pixel[1] < (self.h - 1) * UNIT:
                move_pixel[1] += UNIT
        elif action == 2:  # 向左移动
            if s_pixel[0] > UNIT:
                move_pixel[0] -= UNIT
        elif action == 3:
            if s_pixel[0] < (self.w - 1) * UNIT:
                move_pixel[0] += UNIT

        # update player position
        self.canvas.move(self.player, move_pixel[0], move_pixel[1])
        ns_pixel = self.canvas.coords(self.player)

        # reward
        if ns_pixel == self.canvas.coords(self.candy):
            reward = 10
            done = True
            ns_pixel =  ns_pixel
        elif ns_pixel in  [self.canvas.coords(self.stone1), self.canvas.coords(self.stone2), self.canvas.coords(self.stone3), self.canvas.coords(self.stone4)]:
            reward = -10
            done = False
            self.canvas.move(self.player, -move_pixel[0], -move_pixel[1])
            ns_pixel = s_pixel
        else:
            reward = -1
            done = False
            ns_pixel = ns_pixel

        return ns_pixel, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()

