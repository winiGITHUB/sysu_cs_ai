def BinarySearch(nums, target):
    """
    :param nums: list[int]
    :param target: int
    :return: int
    """
    left= 0
    right=len(nums) - 1  # 初始化左右指针
    while left <= right:
        mid = (left + right) // 2  # 计算中间索引(注意是整除)
        if nums[mid] == target:  # 如果中间值等于目标值，则返回中间索引
            return mid
        elif nums[mid] < target:  # 如果中间值小于目标值，则将左指针移动到中间索引的右侧
            left = mid + 1
        else:  # 如果中间值大于目标值，则将右指针移动到中间索引的左侧
            right = mid - 1
    return -1  # 如果未找到目标值，则返回 -1

# 测试样例
nums = [2,3,7,11,35,56,236,293,778]
target = 35
print("Index of target:", BinarySearch(nums, target))

