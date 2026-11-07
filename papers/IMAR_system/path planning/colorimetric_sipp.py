
import yaml
from ultralytics import YOLO
import numpy as np
import pandas as pd
import re
from write_yaml import write_yaml
from sipp import SippPlanner

def bbox2accords(xywh,xyxy):
    accs = []
    accs_predicted = []
    scale = 15.95
    for i in range(len(xywh)):
        x,y,w,h = int(xyxy[i][0]/scale),int(xyxy[i][1]/scale),max(int(xywh[i][2]/scale),1),max(int(xywh[i][3]/scale),1)
        # 图片上的x轴方向与片上x轴方向不同，需要变换处理：
        x = 128 - w - (x - 1)
        accs.append([x,y,w,h])
        accs_predicted.append([x,y])
    return accs,accs_predicted


def path_export(file_path,w,h):

    with open(file_path + r'output.yaml','r') as file:
        data = yaml.safe_load(file)

    #获取 agent_list：
    agent_list = ['agent'+str(i) for i in range(len(data['schedule']))]
    max_t = 0
    for agent in agent_list:
        if len(data['schedule'][agent]) > max_t:
            max_t = len(data['schedule'][agent])
    print('Total timestep:{}'.format(max_t))
    res = [''for i in range(max_t)]
    for agent in agent_list:
        agent_t = len(data['schedule'][agent])-1
        for i in range(max_t):
            if i < agent_t:
                res[i] += '('+str(data['schedule'][agent][i]['x'])+','+str(data['schedule'][agent][i]['y'])+')({},{});'.format(w,h)
            else:
                res[i] += '('+str(data['schedule'][agent][agent_t]['x'])+','+str(data['schedule'][agent][agent_t]['y'])+')({},{});'.format(w,h)

    for i in range(len(res)):
        res[i] = res[i][:-1]
        res[i] += '-1000\n'

    with open(file_path + r'output_path.txt', 'w') as file:
        for i in range(len(res)):
            file.write(res[i])

def drop_sipp(accs, goals, file_path, w, h):
    write_yaml(accs, goals, file_path, w, h)
    with open(file_path + 'sipp_input.yaml','r') as map_file:
        map = yaml.load(map_file, Loader=yaml.FullLoader)
    t_max = 0
    # Output file
    output = dict()
    output["schedule"] = dict()
    for i in range(len(map["agents"])):
        # 判断是否起点和终点相同，若相同，则不需要规划，plan只有一步，添加后直接跳过
        # 若需添加障碍，在此处
        if map['agents'][i]['goal'] == map['agents'][i]['start']:
            plan = {map['agents'][i]['name']: [
                {'x': map['agents'][i]['goal'][0], 'y': map['agents'][i]['goal'][1], 't': 0}]}
            output["schedule"].update(plan)
            map["dynamic_obstacles"].update(plan)

        else:
            # w为液滴宽度，h为液滴高度，从左上角液滴开始
            sipp_planner = SippPlanner(map, i, w, h)
            if sipp_planner.compute_plan():
                plan = sipp_planner.get_plan()
                output["schedule"].update(plan)
                # 这步把每个agent都视作移动障碍物，并添加到map里面
                map["dynamic_obstacles"].update(plan)

            else:
                print("Plan not found")

        t_max_1 = t_max if plan['agent{}'.format(i)][-1]['t'] < t_max else plan['agent{}'.format(i)][-1]['t']

        if t_max_1 != t_max and i != 1:
            for j in range(i):
                t_temp = map["dynamic_obstacles"]['agent{}'.format(j)][-1]['t']
                for t in range(t_temp,t_max_1):
                    last_step = {'x':map["dynamic_obstacles"]['agent{}'.format(j)][-1]['x'],
                                 'y':map["dynamic_obstacles"]['agent{}'.format(j)][-1]['y'],
                                 't':t}
                    map["dynamic_obstacles"]['agent{}'.format(j)].append(last_step)
        t_max = t_max_1

    with open(file_path + r'output.yaml', 'w') as output_yaml:
        yaml.safe_dump(output, output_yaml)
    path_export(file_path,w,h)

# TODO: 此函数将液滴通过路径规划重排
def rearrange(accs):
    # 先由分割结果确定目标阵列
    n = len(accs)
    goals = []
    for i in range(n):
        goals.append([25 + int(i % 16) * 5, 83 + int(i // 16) * 5])

    file_path = './demo/rearrange_'
    goals_dup = goals.copy()
    drop_sipp(accs,goals,file_path, 3)

    return goals_dup


def txt2accs(goal_file):
    goal = []
    with open(goal_file,'r') as file:
        data = file.readline()
    data = data.split(';')
    for i in range(len(data)):
        drop = [int(re.findall(r"\((\w*),(\w*)\)",data[i])[0][0]),
                int(re.findall(r"\((\w*),(\w*)\)",data[i])[0][1]),
                int(re.findall(r"\((\w*),(\w*)\)",data[i])[1][0]),
                int(re.findall(r"\((\w*),(\w*)\)",data[i])[1][1])]
        goal.append(drop)
    return goal

def output_accs(accs,file):
    out = ''
    for acc in accs:
        out += '({},{})({},{});'.format(acc[0],acc[1],acc[2],acc[3])
    out += '-1000'
    with open(file,'w') as f:
        f.write(out)
    f.close()

def pic_to_accs(pic_path,output_path,save = False):
    # 加载模型
    model = YOLO('best1.pt')
    # 输入低倍镜图片，并检测
    result = model.predict(pic_path, conf=0.5, imgsz=1024, save=save)[0]
    xywh = result.boxes.xywh.cpu().numpy()
    xyxy = result.boxes.xyxy.cpu().numpy()

    # accs:[x,y,w,h]软件对应格式; accs_predicted:[x,y]以单点形式列出所有的点
    accs, accs_predicted = bbox2accords(xywh, xyxy)

    # 将accs输出，以便观察识别结果是否准确
    output_accs(accs, file=output_path)
    return accs, accs_predicted

if __name__ == "__main__":
    # 加载模型
    # model = YOLO('best1.pt')
    # accs, accs_predicted = pic_to_accs('path_test/test_pic1.jpg','path_test/start.txt',save=False)
    accs = txt2accs('path_test/start_32.txt')
    goals = txt2accs('path_test/goal_32.txt')

    # 此处的目标需要重新分配，将坐标分为左侧和右侧，左侧目标为奇数行，右侧目标为偶数行
    drop_sipp(accs, goals, 'path_test/', 2, 1)

    # transform(accs,goal_file)