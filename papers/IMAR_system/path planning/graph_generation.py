"""

Graph generation for sipp 

author: Ashwin Bose (@atb033)

See the article: 10.1109/ICRA.2011.5980306

"""

import argparse
import yaml
from bisect import bisect

class State(object):
    def __init__(self, position=(-1,-1), t=0, interval=(0,float('inf'))):
        self.position = tuple(position)
        self.time = t
        self.interval = interval

class SippGrid(object):
    def __init__(self):
        # self.position = ()
        self.interval_list = [(0, float('inf'))]
        self.f = float('inf')
        self.g = float('inf')
        self.parent_state = State()

    def split_interval(self, t, last_t = False):
        """
        Function to generate safe-intervals
        """
        for interval in self.interval_list:
            if last_t:
                if t <= interval[0]:
                    self.interval_list.remove(interval)
                elif t>interval[1]:
                    continue
                else:
                    self.interval_list.remove(interval)
                    self.interval_list.append((interval[0], t-1))
            else:
                if t == interval[0]:
                    self.interval_list.remove(interval)
                    if t+1 <= interval[1]:
                        self.interval_list.append((t+1, interval[1]))
                elif t == interval[1]:
                    self.interval_list.remove(interval)
                    if t-1 <= interval[0]:
                        self.interval_list.append((interval[0],t-1))
                elif bisect(interval,t) == 1:
                    self.interval_list.remove(interval)
                    self.interval_list.append((interval[0], t-1))
                    self.interval_list.append((t+1, interval[1]))
            self.interval_list.sort()


class SippGraph(object):
    def __init__(self, map, w, h):
        self.map = map
        self.dimensions = map["map"]["dimensions"]
        self.w = w
        self.h = h
        self.obstacles = [tuple(v) for v in map["map"]["obstacles"]]
        self.dyn_obstacles = map["dynamic_obstacles"]

        self.sipp_graph = {}
        self.init_graph()
        self.init_intervals()

    def init_graph(self):
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                grid_dict = {(i,j):SippGrid()}
                self.sipp_graph.update(grid_dict)

    def init_intervals(self):
        if not self.dyn_obstacles: return
        for schedule in self.dyn_obstacles.values():
            # for location in schedule:
            for i in range(len(schedule)):
                location = schedule[i]
                last_t = i == len(schedule)-1
                positions = pad(location,self.w, self.h)
                # w,h 分别为液滴宽高
                t = location["t"]
                for position in positions:
                    self.sipp_graph[position].split_interval(t, last_t)
                # print(str(position) + str(self.sipp_graph[position].interval_list))

    def is_valid_position(self, position):
        w = self.w
        h = self.h
        # 还应该有一个agent check，检查是否邻近其他agent
        dim_check = position[0] in range(self.dimensions[0]) and  position[1] in range(self.dimensions[1])
        obs_check = 1
        for i in range(w):
            for j in range(h):
                obs_check *= tuple([position[0]+i,position[1]+j]) not in self.obstacles
        # print(dim_check)
        return dim_check and obs_check

    def get_valid_neighbours(self, position):
        neighbour_list = [(position[0], position[1]+1),
                          (position[0], position[1]-1),
                          (position[0]+1, position[1]),
                          (position[0]-1, position[1])]
        #
        # up_left = (position[0]-1, position[1]+1)
        # if self.is_valid_position(up_left): neighbour_list.append(up_left)
        # up_right = (position[0]+1, position[1]+1)
        # if self.is_valid_position(up_left): neighbour_list.append(up_right)
        # down_left = (position[0]-1, position[1]-1)
        # if self.is_valid_position(up_left): neighbour_list.append(down_left)
        # down_right = (position[0]+1, position[1]-1)
        # if self.is_valid_position(up_left): neighbour_list.append(down_right)

        return [neighbour for neighbour in neighbour_list if self.is_valid_position(neighbour)]

def pad(position,w, h):
    #填充以postion为中心，k为半径的电极
    position = [position['x'],position['y']]
    res = []
    for i in range(-3,w+2):
        for j in range(-3,h+2):
            res.append((max(position[0]+i,0),max(position[1]+j,0)))
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("map", help="input file containing map and dynamic dyn_obstacles")
    args = parser.parse_args()
    
    with open(args.map, 'r') as map_file:
        try:
            map = yaml.load(map_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    graph = SippGraph(map)
    # print(graph.get_valid_neighbours((0,0)))
    # print(graph.sipp_graph[(1,1)].interval_list)
    # print(graph.get_valid_neighbours((1,2)))

if __name__ == "__main__":
    main()

