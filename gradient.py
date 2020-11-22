'''
梯度下降演示示例
'''
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.lines as mlines

fun = lambda x: (x**3) - 3 * (x**2) + 7

'''
表达式 (x**3) - 3 * (x**2) + 7 的导数
'''
def deriv(x): 
    x_dervi = 3 * (x ** 2) - 6 * x
    return x_dervi


'''
点x_0处切线
'''
def tangent_line(x_0): 
    x = np.linspace(-1,3,500)
    y = fun(x)
    y_0 = fun(x_0)
    y_tan = deriv(x_0) * (x - x_0) + y_0 
    plt.plot(x,y,'r-')
    plt.plot(x,y_tan,'b-')
    plt.show()

#tangent_line(0)

def step(x_new,x_prev,precision,l_r): 
    x_list,y_list = [x_new],[fun(x_new)]
    while abs(x_new - x_prev) > precision: 
        x_prev = x_new 
        d_x = -deriv(x_new)
        x_new = x_prev + d_x * l_r
        x_list.append(x_new)
        y_list.append(fun(x_new))

    for i in range(len(x_list)): 
        plt.clf()
        x = np.linspace(-1,3,500)
        y = fun(x)
        plt.plot(x,y,'r-')
        plt.scatter(x_list[i],y_list[i], c="g") #绘制当前点
        y_i = fun(x_list[i])
        tan = deriv(x_list[i])

        tanline_begin = tan * (-0.5) + y_i 
        tanline_end = tan * 0.5 + y_i 
        ax = plt.gca() #获取当前坐标轴
        l = mlines.Line2D([x_list[i] - 0.5, x_list[i] + 0.5], [tanline_begin, tanline_end]) 
        ax.add_line(l)
        plt.pause(0.1) #每隔0.1秒重绘下图形

    plt.show()

step(0.1,0, 0.001, 0.05)

