import enum
import random

import numpy as np


class State():
    def __init__(self, row=-1, column=-1):
        self.row = row
        self.column = column

    def __repr__(self):
        return "<State: [{}, {}]>".format(self.row, self.column)

    def clone(self):
        return State(self.row, self.column)

    def __hash__(self):
        return hash((self.row, self.column))

    def __eq__(self, other):
        return self.row == other.row and self.column == other.column


class Action(enum.Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


class Environment():
    def __init__(self, grid, move_prob=0.8):
        # grid 是一个二维数组，他的值可以看做是属性
        # 格子的属性如下：
        # 0 普通格子
        # 1 奖励格子
        # -1 惩罚格子
        # 9 屏蔽格子
        self.grid = grid
        self.agent_state = State()

        # 默认的奖励是负数，就像施加了初始位置的惩罚
        # 这意味着智能体必须快速到达终点
        self.default_reward = 0.4

        # 智能体能够以 move_prob 的概率向所选方向移动
        # 如果移动概率落在了 (1 - move_prob) 中，则随机运动
        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self):
        return len(self.grid)

    @property
    def column_length(self):
        return len(self.grid[0])

    @property
    def actions(self):
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                # state 中不包含被屏蔽的格子
                if self.grid[row][column] != 9:
                    states.append(State(row, column))
        return states

    def transit_func(self, state, action):
        transition_probs = {}
        if not self.can_action_at(state):
            # 已经到达游戏结束的格子
            return transition_probs

        opposite_direction = Action(action.value * -1)

        for a in self.actions:
            # 获取动作执行的可能性的数组
            if a == action:  # 正确运动的几率
                prob = self.move_prob
            elif a != opposite_direction:  # 侧向运动的几率
                prob = (1 - self.move_prob) / 2
            else:  # 反向运动几率
                prob = 0

            print("action {} has prob {}".format(a, prob))

            # 获取下一状态切换的可能性的数组
            next_state = self._move(state, a)
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob

        return transition_probs

    def can_action_at(self, state):
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False

    def _move(self, state, action):
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state = state.clone()

        # 执行行动（移动）
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        # 检查状态是否在 grid 外
        if not (0 <= next_state.row < self.row_length):
            next_state = state

        if not (0 <= next_state.column < self.column_length):
            next_state = state

        # 检查智能体是否到达了被屏蔽的格子
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    def reward_func(self, state):
        reward = self.default_reward
        done = False

        # 检查下一种状态的属性
        attribute = self.grid[state.row][state.column]
        if attribute == 1:
            # 获取奖励，游戏结束
            reward = 1
            done = True
        elif attribute == -1:
            # 遇到危险，游戏结束
            reward = -1
            done = True

        return reward, done

    def reset(self):
        # 将智能体放置到左下角
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    def transit(self, state, action):
        # 获取下一状态是XX发生的可能性数组
        transition_probs = self.transit_func(state, action)
        if len(transition_probs) == 0:
            return None, None, None

        next_states = []
        probs = []
        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])

        if np.sum(probs) != 0:
            probs = probs / np.sum(probs)

        print("next_states = {}, probs = {}".format(next_states, probs))

        # 随机选择函数
        next_states = np.random.choice(a=next_states, p=probs)
        reward, done = self.reward_func(next_states)
        return next_states, reward, done

    def step(self, action):
        next_state, reward, done = self.transit(self.agent_state, action)

        if next_state is not None:
            self.agent_state = next_state

        return next_state, reward, done


class Agent():
    def __init__(self, env):
        self.actions = env.actions

    def policy(self, state):
        # 策略1：完全随机
        return random.choice(self.actions)


def main():
    # 创建 grid
    grid = [
        [0, 0, 0, 1],
        [0, 9, 0, -1],
        [0, 0, 0, 0],
    ]

    env = Environment(grid)
    agent = Agent(env)

    # 尝试10次游戏
    for i in range(10):
        # 初始化智能体的位置
        state = env.reset()
        action = 0
        total_reward = 0
        done = False

        while not done:
            action = agent.policy(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

        print("Episode {}: Agent at state {} take action {} gets {} reward."
              .format(i, state, action, total_reward))


if __name__ == "__main__":
    main()
