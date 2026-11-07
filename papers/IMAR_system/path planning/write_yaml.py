import os
import yaml
import random
import numpy as np
# DM MAX路径格式：行，列；高，宽（界面的右上角即是芯片的左上角）

# 计算离此drop曼哈顿距离最近的点，并把此点pop掉
def get_goal(goal,accs):
    if goal[1]%2 == 0:
        for i in range(len(accs)):
            if accs[i][0] > 65:
                drop = accs[i]
                accs.remove(drop)
                break
    else:
        for i in range(len(accs)):
            if accs[i][0] <= 65:
                drop = accs[i]
                accs.remove(drop)
                break

    return drop,accs

def write_yaml(accs, goals, file_path, w, h):
    # 对输入格式做适配
    input = {"agents":[],
             'map':{'dimensions': [128, 128],'obstacles':[]},
             'dynamic_obstacles':{}}

    # 下面还能优化，如果bit排布效果不好的话，根据液滴的面积大小，选出液滴面积最小的几个液滴，来构成字母，这样减小碰撞的概率，同时也进一步体现出目标识别模型的必要性。
    # if by_area:

    # inter 是 goals跟accs的交集
    inter = []
    for goal in goals:
        if goal in accs:
            inter.append(goal)

    for i in range(len(inter)):
        acc = inter[i]
        goals.remove(acc)
        accs.remove(acc)
        input['agents'].append({'start': acc[0:2], 'goal': acc[0:2], 'name': 'agent' + str(i), 'width': acc[2], 'height': acc[3]})

    # 2024年5月15日写下面的部分：
    for i in range(len(goals)):
        goal = goals[i]
        # 需要为其分配一个目标
        drop, accs = get_goal(goal, accs)
        input['agents'].append({'start':drop[0:2],'goal': goal[0:2],'name':'agent'+str(i+len(inter)), 'width': drop[2], 'height': drop[3]})

        for n in range(h+2):
            input['map']['obstacles'].append(tuple([goal[0] - 1, goal[1] - 1 + n]))
            input['map']['obstacles'].append(tuple([goal[0] + w, goal[1] - 1 + n]))

        for m in range(w):
            input['map']['obstacles'].append(tuple([goal[0] + m, goal[1] - 1]))



    with open(file=file_path+'sipp_input.yaml', mode='w', encoding='utf8') as f:
        yaml.safe_dump(input, f)

# def read_yaml(data, file_path):
#     write_yaml(data, file_path)
#     f = open(file=file_path, mode='r', encoding='utf8')
#     return yaml.safe_load(f)
