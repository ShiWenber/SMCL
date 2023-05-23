# 读取 log 查看 acc 结果
import numpy as np
import os
import sys
# 读取文件中的字符串内容

def get_acc(path: str):
    """获取nohup日志中的acc值

    Args:
        path (str): 日志文件路径

    Returns:
        tuple: (acc_max, acc_mean, acc_minus, acc_plus, acc_ls)
    """
    # data = open("./nohup_logs/20230515002301/IMDB-BINARY.log").read()
    data = open(path).read()
    # 按照换行符和逗号分割字符串
    data_ls = data.split("\n")
    acc_ls = []
    for i in data_ls:
        for j in i.split(","):
            if (j.startswith("Acc")):
                # print(j)
                j.replace("Acc: ","")
                acc = float(j.split(": ")[1])
                acc_ls.append(acc)
    if len(acc_ls) == 0:
        print("acc_ls is null")
        return
    # 计算acc的平均值和+/-误差
    acc_mean = np.mean(acc_ls)
    print("acc_mean: ",acc_mean)
    acc_max = np.max(acc_ls)
    print("acc_max", acc_max)
    # 计算负误差
    acc_minus = acc_mean - np.min(acc_ls)
    acc_plus = np.max(acc_ls) - acc_mean
    print("acc_minus: ",acc_minus)
    print("acc_plus: ",acc_plus)
    return acc_max, acc_mean, acc_minus, acc_plus, acc_ls

if __name__ == "__main__":
    args = sys.argv
    if args.__len__() == 1:
        print("args is null") 
        print("[Usage]: python summary_logs.py <dir_path>")
        exit()
    else:
        dir_path = args[1]
        # 检查是否以/结尾
        if dir_path[-1] != "/":
            dir_path += "/"
        print(dir_path)
    path_list = os.listdir(dir_path)
    acc_dict = {}
    for i in path_list:
        print(f"====================================={i}")
        path = dir_path + i
        print(path)
        res = get_acc(path)
        if (res == None):
            continue
        acc_max = res[0]
        name = i.split(".")[0]
        acc_dict[name] = acc_max
    print()
    print()
    print("=====================================summary")
    print(acc_dict)
 