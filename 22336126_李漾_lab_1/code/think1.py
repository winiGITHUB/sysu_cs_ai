import random

def guess_number():
    # 生成随机答案
    answer = random.randint(1, 100)
    attempts = 0

    while True:
        # 循环读取用户输入的猜测数字
        guess = int(input("请输入一个猜测的数字（1到100之间）："))
        attempts += 1

        # 判断猜测的数字是否等于答案
        if guess == answer:
            print("恭喜你猜对了！答案是", answer)
            break
        elif guess < answer:
            print("猜的数字偏小，请继续尝试。")
        else:
            print("猜的数字偏大，请继续尝试。")

    print("你一共猜测了", attempts, "次。")

if __name__ == "__main__":
    guess_number()
