import numpy as np
import random

# 定义环境参数
WORLD_HEIGHT = 4  # 网格高度
WORLD_WIDTH = 12  # 网格宽度
START = [3, 0]  # 起点坐标
GOAL = [3, 11]  # 终点坐标
CLIFF = [(3, i) for i in range(1, 11)]  # 悬崖坐标

# 动作定义 [上, 右, 下, 左]
ACTIONS = [0, 1, 2, 3]
ACTION_SYMBOLS = ['↑', '→', '↓', '←']

# Q-learning参数
EPSILON = 0.1  # 探索概率
ALPHA = 0.5  # 学习率
GAMMA = 1.0  # 折扣因子
EPISODES = 500  # 训练轮数

# 初始化Q表 (状态空间大小 x 动作空间大小)
Q = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, len(ACTIONS)))


def step(state, action):
    """执行动作，返回新状态和奖励"""
    y, x = state

    # 根据动作移动
    if action == 0:
        y = max(y - 1, 0)  # 上
    elif action == 1:
        x = min(x + 1, WORLD_WIDTH - 1)  # 右
    elif action == 2:
        y = min(y + 1, WORLD_HEIGHT - 1)  # 下
    elif action == 3:
        x = max(x - 1, 0)  # 左

    new_state = [y, x]

    # 判断状态类型
    if tuple(new_state) in CLIFF:  # 掉入悬崖
        reward = -100
        new_state = START  # 回到起点
    elif new_state == GOAL:  # 到达终点
        reward = 0
    else:  # 普通移动
        reward = -1

    return new_state, reward


def choose_action(state):
    """ε-贪婪策略选择动作"""
    if random.random() < EPSILON:
        return random.choice(ACTIONS)  # 随机探索
    else:
        y, x = state
        return np.argmax(Q[y, x, :])  # 选择当前最优动作


def train():
    """训练Q-learning代理"""
    for episode in range(EPISODES):
        state = START.copy()  # 初始化状态
        total_reward = 0

        while state != GOAL:  # 未到终点时继续
            # 选择并执行动作
            action = choose_action(state)
            next_state, reward = step(state, action)
            total_reward += reward

            # 更新Q值
            y, x = state
            ny, nx = next_state
            # Q-learning更新公式
            Q[y, x, action] += ALPHA * (reward + GAMMA * np.max(Q[ny, nx, :]) - Q[y, x, action])

            state = next_state  # 转移到新状态

        # 每50轮打印一次训练进度
        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}: Total reward = {total_reward}")


def show_policy():
    """显示最终策略"""
    policy_grid = [['' for _ in range(WORLD_WIDTH)] for _ in range(WORLD_HEIGHT)]

    for y in range(WORLD_HEIGHT):
        for x in range(WORLD_WIDTH):
            if [y, x] == GOAL:
                policy_grid[y][x] = 'G'  # 终点
            elif (y, x) in CLIFF:
                policy_grid[y][x] = 'X'  # 悬崖
            else:
                # 选择最优动作对应的符号
                best_action = np.argmax(Q[y, x, :])
                policy_grid[y][x] = ACTION_SYMBOLS[best_action]

    # 打印策略网格
    print("\n最终策略：")
    for row in policy_grid:
        print(' '.join(row))


# 主程序
if __name__ == "__main__":
    train()  # 训练模型
    show_policy()  # 显示学习到的策略

"""
代码解释：
环境设置：

4×12网格世界，起点在左下角(3,0)，终点在右下角(3,11)

底部中间的10个格子是悬崖（掉入悬崖得-100奖励并返回起点）

动作系统：

4个基本动作：上(0)、右(1)、下(2)、左(3)

移动时碰到边界会保持原位

奖励机制：

普通移动：-1（鼓励最短路径）

掉入悬崖：-100

到达终点：0

Q-learning核心：

Q表：三维数组[行][列][动作]存储状态-动作值

ε-贪婪策略：90%概率选择最优动作，10%概率随机探索

更新公式：

text
Q(s,a) ← Q(s,a) + α[R + γ·maxQ(s',a') - Q(s,a)]
其中α=学习率，γ=折扣因子

训练过程：

进行500轮训练

每轮从起点开始，直到到达终点

每50轮打印累计奖励（理想值≈-13）

策略可视化：

训练后打印网格策略

用箭头表示最优动作，'G'表示终点，'X'表示悬崖
"""