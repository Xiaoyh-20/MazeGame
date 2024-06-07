"""
@Project: code
@File: main.py
@Description:
@Author: xiaoyihong
@Email: xiaoyh20@mails.tsinghua.edu.cn
@Date: 2024/6/7
"""
from maze_env import Maze
from agent import QLearningAgent

if __name__ == '__main__':
    # Initialize the environment
    env = Maze()
    agent = QLearningAgent(env)
    env.set_agent(agent)
    env._build_maze()

    # Train the agent
    env.after(100, agent.train(episodes=1000))
    print("Training completed.")
    print('-' * 50)
    # Test the learned policy
    print("Testing the learned policy...")
    agent.test()
    print("Testing completed.")
    print('-' * 50)
    # show the converged Q-table
    print('The converged Q-table is:\n')
    agent.show_q_table()
    env.mainloop()

