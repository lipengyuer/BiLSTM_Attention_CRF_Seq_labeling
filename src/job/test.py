'''
Created on 2019年9月11日

@author: Administrator
'''
#人工鱼群算法
import numpy as np
import matplotlib.pyplot as plt

#第一阶段的人工鱼群只有追寻食物的机制。只让人工鱼寻找食物浓度最高的位置，就可以实现对一些的损失函数的优化了。
class AFish():
    
    def __init__(self):
        self.location = None#描述鱼在参数空间中位置的向量
        self.current_food_density = None
        
class AFSA():
    
    def __init__(self, fish_num = 100, location_dim=2, visual=0.01, try_num_searching_food=3):
        self.location_dim = location_dim
        self.fishes = None
        self.bulletin_fish = None
        self.visual = visual#所有鱼的视野范围。可以为每条鱼设置不同的视野大小，模拟个体差异
        self.try_num_searching_food = try_num_searching_food#多试几次有助于找到更好的方向。
        self.create_fishes(fish_num, location_dim)

        
    def food_density(self, location):
        x, y = location
        score = -(x**2+y**2 +2*y)#z=x**2+y**2 +2*y,需要加一个符号，让函数是凸的。如果需要优化的目标函数比较复杂，就没有这么直观了。
        return score
    
    def distance(self, location1, location2):
        return np.dot(location1, location2)
    
    #生成一个步长，目标是视野范围内的随机一个点
    def generate_a_step(self):
        step_vector = np.random.uniform(-self.visual, self.visual, self.location_dim)
        return step_vector
    
    def create_fishes(self, fish_num, location_dim):
        self.fishes = []
        for _ in range(fish_num):
            a_fish = AFish()
            a_fish.location = np.random.random(location_dim) 
#             print("a_fish.location", a_fish.location)
            a_fish.current_food_density = self.food_density(a_fish.location)
            self.fishes.append(a_fish)
            if self.bulletin_fish==None:
                self.bulletin_fish = a_fish
            else:
                if a_fish.current_food_density > self.bulletin_fish.current_food_density:
                    self.bulletin_fish = a_fish
           
    def search_food(self, fish):
        for _ in range(self.try_num_searching_food):
            new_location = fish.location + self.generate_a_step()
            new_density = self.food_density(new_location)
            concentration = self.food_density(fish.location)
            if  new_density > concentration:
                fish.location = new_location
                concentration = new_density
            if concentration > self.bulletin_fish.current_food_density:
                self.bulletin_fish = fish
             
    #更新一条鱼的状态，并更新公示板        
    def update_a_fish(self, fish):
        self.search_food(fish)#模拟一条鱼找食物的动作
        
    def fit(self ,epoch_num=1000):
        
        #########将寻优过程可视化，不是必须的#########
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        x = list(map(lambda x: x.location[0], self.fishes))
        y = list(map(lambda x: x.location[1], self.fishes))
        self.ax.scatter(x, y)
        plt.ion()
        ############可视化部分####################
        
        for epoch in range(epoch_num):
            for fish in self.fishes:
                self.update_a_fish(fish)
            print("轮次是", epoch)
            print(self.bulletin_fish.location)
            
            self.show_locations()#可视化

    #把所有鱼的位置变化展示出来，参考了https://blog.csdn.net/omodao1/article/details/81223240
    def show_locations(self):
        x = list(map(lambda x: x.location[0], self.fishes))
        y = list(map(lambda x: x.location[1], self.fishes))
        try:
            self.ax.lines.remove(self.lines[0])
        except Exception as e:
            print(e)
        self.lines = self.ax.plot(x ,y, '*')
        plt.pause(0.1)

if __name__ == '__main__':
    afsa = AFSA()
    afsa.fit(epoch_num=1000)