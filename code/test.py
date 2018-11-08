import math
profit = int(input("请输入利润（单位：万元）: "))
reward = 0
if profit <= 10:
    reward = profit * 0.1
elif profit <= 20:
    reward = 10*0.1 + ( reward - 10 ) * 0.075
elif profit <= 40:
    reward = 10 * 0.1 + 10 *  0.075 + ( reward - 20 ) * 0.05
elif profit <= 60:
    reward = 10 * 0.1 + 10 * 0.075 + 20 * 0.03 + ( profit - 40 ) * 0.03
elif profit <= 100:
    reward = 10 * 0.1 + 10 * 0.075 + 20 * 0.03 + 20 * 0.015 + ( profit - 60 ) * 0.015
elif profit > 100:
    reward = 10 * 0.1 + 10 * 0.075 + 20 * 0.03 + 20 * 0.015 + 40 * 0.015 + ( profit - 100 ) * 0.01
print("奖金是{}（单位：万元）".format(reward))