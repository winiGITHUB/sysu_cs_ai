def ReverseKeyValue(dict1):
    """
    :param dict1: dict
    :return: dict
    """
    r_dict = {}  # 创建一个空字典用于存放颠倒后的键值对
    for k, v in dict1.items():  # 遍历原字典的键值对
        r_dict[v] = k  # 将原字典的值作为新字典的键，原字典的键作为新字典的值
    return r_dict

# 测试
dict1 = {'Alice': '001', 'Bob': '002'}
r_dict = ReverseKeyValue(dict1)
print(r_dict)
