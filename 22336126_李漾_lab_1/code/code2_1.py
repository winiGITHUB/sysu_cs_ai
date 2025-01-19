class StuData:
    def __init__(self, file_name):
        """
        构造函数，从文件中读取学生数据并存储到 self.data 中
        :param file_name: str, 学生数据文件名
        """
        self.data = []
        with open(file_name, 'r') as file:
            for line in file:
                student = line.strip().split()  # 将每一行的数据按空格分割成列表
                self.data.append(student)  # 将学生信息列表添加到 self.data 中

    def AddData(self, name, stu_num, gender, age):
        """
        添加新的学生数据到 self.data 中
        :param name: str, 学生姓名
        :param stu_num: str, 学号
        :param gender: str, 性别
        :param age: int, 年龄
        """
        student = [name, stu_num, gender, age]  # 创建学生信息列表
        self.data.append(student)  # 将学生信息列表添加到 self.data 中

    def SortData(self, a):
        """
        根据指定属性对学生数据进行排序
        :param a: str, 排序属性，可以是 'name', 'stu_num', 'gender' 或 'age'
        """
        if a == 'name':
            self.data.sort(key=lambda x: x[0])  # 根据姓名排序
        elif a == 'stu_num':
            self.data.sort(key=lambda x: x[1])  # 根据学号排序
        elif a == 'gender':
            self.data.sort(key=lambda x: x[2])  # 根据性别排序
        elif a == 'age':
            self.data.sort(key=lambda x: x[3])  # 根据年龄排序
#使用了 lambda 函数作为 key 参数，以实现按照指定属性排序的功能。
#lambda 函数根据输入的 x 值（即学生数据的一个列表），返回对应属性的值，然后 sort() 方法根据这个值进行排序。
    def ExportFile(self, filename):
        """
        将学生数据导出到文件
        :param filename: str, 导出文件名
        """
        with open(filename, 'w') as file:
            for student_info in self.data:
                file.write(' '.join(map(str, student_info)) + '\n')  # 将学生信息列表转换为字符串并写入文件中

# 测试
if __name__ == "__main__":
    # 创建 StuData 实例
    student_data = StuData("student_data.txt")

    # 打印原始数据
    print("Original Data:")
    for student_info in student_data.data:
        print(student_info)

    # 添加新数据
    student_data.AddData(name="Bob", stu_num="003", gender="M", age=20)

    # 打印添加新数据后的数据
    print("\nData after adding new student:")
    for student_info in student_data.data:
        print(student_info)

    # 按学号排序数据
    student_data.SortData('stu_num')

    # 打印排序后的数据
    print("\nData after sorting by student number:")
    for student_info in student_data.data:
        print(student_info)

    # 导出数据到文件
    student_data.ExportFile("new_student_data.txt")
