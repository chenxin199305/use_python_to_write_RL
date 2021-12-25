import numpy as np

def V(s, gamma=0.99):
    V = R(s) + gamma * max_V_on_next_state(s)
    return V

def R(s):
    # 只有 ending 时才有奖励
    if s == "happy_end":
        return 1
    elif s == "bad_end":
        return -1
    else:
        return 0

# 这种迭代策略，一直在计算状态的最大值的可能性
def max_V_on_next_state(s):
    # 如果游戏结束，则期望值是 0
    if s in ["happy_end", "bad_end"]:
        return 0

    actions = ["up", "down"]
    values = []
    for a in actions:
        transition_probs = transit_func(s, a)  # 状态迁移函数: 得到 <状态，迁移概率>
        v = 0
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            v += prob * V(next_state)  # 计算状态的期望价值，动态规划（迭代）计算
        values.append(v)
    return max(values)

def transit_func(s, a):
    """
    Make next state by adding action str to state
    :param s:
    :param a:
    :return:
    """
    actions = s.split("_")[1:]
    LIMIT_GAME_COUNT = 5
    HAPPY_END_BORDER = 4  # 如果有4次"up"就是"happy_end"
    MOVE_PROB = 0.9

    def next_state(state, action):
        return "_".join([state, action])

    if len(actions) == LIMIT_GAME_COUNT:
        up_count = sum([1 if a == "up" else 0 for a in actions])
        state = "happy_end" if up_count >= HAPPY_END_BORDER else "bad_end"
        prob = 1.0
        return {state:prob}
    else:
        opposite = "up" if a == "down" else "down"
        return {
            next_state(s, a): MOVE_PROB,
            next_state(s, opposite): 1 - MOVE_PROB
        }

if __name__ == "__main__":
    print(V("happy_end"))
    print(V("bad_end"))
    print(V("state"))
    print(V("state_up_up"))
    print(V("state_down_down"))
    print(V("state_up_up_up_up_up"))
    print(V("state_down_down_down_down_down"))
