import random  # 导入随机数模块
import math  # 导入数学计算模块
import matplotlib.pyplot as plt  # 导入绘图模块
import matplotlib.animation as animation  # 导入动画模块

class GeneticAlgorithm:
    def __init__(self, file_path, population_size=20, mutation_rate=0.6, max_iterations=100):
        """
        初始化遗传算法对象

        参数:
            file_path (str): 城市坐标文件路径
            population_size (int): 种群大小，默认为20
            mutation_rate (float): 变异概率，默认为0.5
            max_iterations (int): 最大迭代次数，默认为500
        """
        self.map_ = self.load_map(file_path)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.max_iterations = max_iterations
        self.population = self.initialize_population()

    def load_map(self, file_path):
        """
        加载城市坐标文件
        城市数据我选取了放在temp里
        参数:
            file_path (str): 城市坐标文件路径

        返回:
            list: 城市坐标列表，每个元素为(city_id, x, y)
        """
        city_coords = []
        with open(file_path, "r") as file:
            for line in file:
                fields = line.strip().split()
                city_coords.append((int(fields[0]), float(fields[1]), float(fields[2])))
        return city_coords

    def initialize_population(self):
        """
        初始化种群

        返回:
            list: 种群列表，每个个体为城市编号列表
        """
        population = []
        for _ in range(self.population_size):
            random_path = random.sample(range(1, len(self.map_)+1), len(self.map_))
            population.append(random_path)
        return population

    def distance(self, a, b):
        """
        计算两个城市之间的距离

        参数:
            a (int): 城市A的编号
            b (int): 城市B的编号

        返回:
            float: 城市A到城市B的距离
        """
        ax, ay = self.map_[a-1][1], self.map_[a-1][2]
        bx, by = self.map_[b-1][1], self.map_[b-1][2]
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    def evaluate(self, path):
        """
        计算路径的总距离

        参数:
            path (list): 城市编号列表，表示一条路径

        返回:
            float: 路径的总距离
        """
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += self.distance(path[i], path[i+1])
        total_distance += self.distance(path[-1], path[0])  # 回到起点
        return total_distance

    def reproduce_ox(self, p1, p2):
        """
        使用OX交叉法进行交叉

        参数:
            p1 (list): 父代1的城市编号列表
            p2 (list): 父代2的城市编号列表

        返回:
            list: 子代的城市编号列表
        """
        length = len(p1)
        sta, end = sorted(random.sample(range(1, length), 2))
        child = p1[sta:end]
        c1, c2 = [], []
        for i in range(0, length):
            if p2[i] not in child:
                if i < sta:
                    c1.append(p2[i])
                else:
                    c2.append(p2[i])
        return c1 + child + c2

    def neighbour3(self, path):
        """
        对路径进行三种变异操作之一
        1.部分随机交换：随机选择三个不同位置的城市，然后将这三个位置上的城市进行交换。
        2.部分逆转：随机选择两个不同位置的城市，然后将这两个位置之间的城市顺序逆转。
        3.部分插入：随机选择两个不同位置的城市，然后将这两个位置之间的城市插入到另一个位置之后。
        参数:
            path (list): 城市编号列表，表示一条路径

        返回:
            list: 变异后的路径
        """
        length = len(path)
        charge = random.random()
        if charge < 0.4:
            a1, a2, a3 = sorted(random.sample(range(1, length-1), 3))
            tmp = path[0:a1] + path[a2:a3] + path[a1:a2] + path[a3:length]
        elif charge < 0.6:
            i = random.randint(1, length-2)
            j = random.randint(2, length-1)
            if i != j:
                path[i], path[j] = path[j], path[i]
                tmp = path[:]
                path[i], path[j] = path[j], path[i]
            else:
                tmp = path[:]
        else:
            k1, k2 = sorted(random.sample(range(1, length-1), 2))
            tmp = path[0:k1] + path[k1:k2][::-1] + path[k2:length]
        return tmp

    def select_parent(self, population):
        """
        从种群中选择父代

        参数:
            population (list): 种群列表，每个个体为城市编号列表

        返回:
            tuple: 选中的两个父代
        """
        the_weight = [1000000 / self.evaluate(individual) for individual in population]
        ch1 = random.choices(population, the_weight, k=1)
        ch2 = random.choices(population, the_weight, k=1)
        return ch1[0], ch2[0]

    def iterate(self):
        """
        迭代函数，执行遗传算法

        返回:
            tuple: 最优路径和对应的最短距离
        """
        ims = []  # 用于存储动画帧
        dis_change = []  # 存储迭代过程中的最短距离
        iteration = 0  # 迭代次数计数器
        best_distance = float('inf')  # 最短距离初始值
        best_individual = None  # 最优路径初始值

        # 开始迭代
        while iteration < self.max_iterations:
            new_population = []  # 新种群

            # 生成新种群
            for count in range(self.population_size // 2):
                ch1, ch2 = self.select_parent(self.population)
                child1 = self.reproduce_ox(ch1, ch2)
                child2 = self.reproduce_ox(ch2, ch1)

                # 变异
                if random.random() < self.mutation_rate:
                    child1 = self.neighbour3(child1)
                    child2 = self.neighbour3(child2)

                # 防止两个子代相同
                if child1 == child2:
                    child1 = self.neighbour3(child1)
                    child2 = self.neighbour3(child2)

                new_population.append(child1)
                new_population.append(child2)

            # 更新种群
            self.population = new_population[:]
            # 计算最优个体和最短距离
            best_individual = min(self.population, key=self.evaluate)
            best_distance = min(best_distance, self.evaluate(best_individual))
            print(iteration, best_distance)
            dis_change.append(best_distance)

            # 每隔一定次数记录一次动画帧
            if iteration % 10 == 0:
                x1, y1 = zip(*[self.map_[idx-1][1:] for idx in best_individual])
                x1 += (self.map_[best_individual[0]-1][1],)
                y1 += (self.map_[best_individual[0]-1][2],)
                im = plt.plot(x1, y1, marker='.', color='red', linewidth=1)
                ims.append(im)

            iteration += 1

        print("loop end")
        # 绘制最优路径
        x1, y1 = zip(*[self.map_[idx-1][1:] for idx in best_individual])
        x1 += (self.map_[best_individual[0]-1][1],)
        y1 += (self.map_[best_individual[0]-1][2],)
        plt.plot(x1, y1, marker='.', color='red', linewidth=1)
        # 保存动画
        ani = animation.ArtistAnimation(plt.figure(1), ims, interval=200, repeat_delay=1000)
        ani.save("GA.gif", writer='pillow')
        # 绘制最短距离的变化过程
        plt.figure(2)
        plt.title('the evolution of the cost')
        x_ = [i for i in range(len(dis_change))]
        plt.plot(x_, dis_change)
        plt.show()

        return best_individual, best_distance

def main():
    tsp = GeneticAlgorithm("temp.txt")
    best_tour, best_distance = tsp.iterate()
    print("Best tour found:", best_tour)
    print("Best distance:", best_distance)

if __name__ == "__main__":
    main()
