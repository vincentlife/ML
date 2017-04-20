# matplotlib
## general
引入包import matplotlib.pyplot as plt，

最后用plt.show()显示图像
基本画图方法：plt.plot(x,y)，plt.xlabel(‘x’)，plt.ylabel(‘y’)，plt.title(‘…’)
子图：plt.subplot(abc)，其中abc分别表示子图行数、列数、序号
创建绘图组件的顶层容器：fig = plt.figure()
添加子图：ax = fig.add_subplot(abc)
设置横轴上的主定位器：ax.xaxis.set_major_locator(…)
绘制方图：plt.hist(a,b)，a为长方形的左横坐标值，b为柱高。alpha是频率分布图的透明度, bins指定bin(箱子)的个数,也就是总共有几条条状图
绘制散点图：plt.scatter(x,y,c = ‘..’,s = ..)，c表示颜色，s表示大小
添加网格线：plt.grid(True)
添加注释：如ax.annotate('x', xy=xpoint, textcoords='offsetpoints',xytext=(-50, 30), arrowprops=dict(arrowstyle="->"))
   
增加图例：如plt.legend(loc='best', fancybox=True)
对坐标取对数：横坐标plt.semilogx()，纵坐标plt.semilogy()，横纵同时plt.log

# Seaborn Methods
## 