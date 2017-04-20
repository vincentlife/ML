# NumPy基本方法
## 数组方法（ndarray）
### 创建数组：
* arange()创建一维数组；
* array() 创建一维或多维数组，其参数是类似于数组的对象，如列表等
* ones(tuple)：全1 
* zeros(tuple)：全0 
* eye(3,k=1) 对角矩阵，从中间第k条对角线为1 其余为0
* empty(tuple)：随机数，取决于内存情况
* diag(v,k) v是向量,从中间第k条对角线起生成对角阵


### 读
* 读取数组元素 a[0],a[0,0]
* flat属性，这是一个遍历整个数组的迭代器 for element in b.flat:
* 获取数组每一维度的大小：a.shape
* 获取数组维度：a.ndim
* 元素个数：a.size
* 数组元素在内存中的字节数：a.itemsize
* 数组字节数：a.nbytes==a.size*a.itemsize
* 将np数组变为py列表：a.tolist()
* tofile()
* np.searchsorted(a,b)将b插入原有序数组a，并返回插入元素的索引值
* ndarray.min,max,sum 同时可以使用axis指定对哪一维进行操作：
* 类型转换：如a.astype(int)，np的数据类型比py丰富，且每种类型都有转换方法
* 条件查找，返回满足条件的数组元素的索引值：np.where(条件)
* 条件查找，返回下标：np.argwhere(条件)
* 条件查找，返回满足条件的数组元素：np.extract([条件],a)

### 数组变型
* reshape & resize
如b=a.reshape(2,3,4)将得到原数组变为2*3*4的三维数组后的数组；或是a.shape=(2,3,4)或a.resize(2,3,4)直接改变数组a的形状。
* newaxis c = b[np.newaxis]
* 数组转置：a.T
* 数组元素覆盖：a.flat=1，则a中数组元素都被1覆盖
* 将a中元素都置为b：a.fill(b)

### 数组组合
* 水平组合hstack((a,b))或concatenate（（a,b）,axis=1）;
* 垂直组合vstack((a,b))或concatenate（（a,b）,axis=0）;
* 深度组合dstack((a,b))

### 数组分割（与数组组合相反）
* hsplit 
* vsplit
* dsplit
* split(split与concatenate相对应)

### 数组排序（小到大）
* 列排列np.msort(a)
* 行排列np.sort(a)
* np.argsort(a)排序后返回下标
* 复数排序：np.sort_complex(a)按先实部后虚部排序

### 插入和查找

example = np.where(np.isnan(example), 0, example)
根据b中元素作为索引，查找a中对应元素：np.take(a,b)一维
数组中最小最大元素的索引：np.argmin(a)，np.argmax(a)
多个数组的对应位置上元素大小的比较：np.maximum(a,b,c,…..)返回每个索引位置上的最大值，np.minimum(…….)相反

### 数组计算
ndarray默认不使用矩阵运算，如果希望对数组进行矩阵运算的话需要调用相应的函数。而matrix默认使用矩阵运算。
* 指数：每个数组元素的指数 np.exp(a)
* ndarray.cumsum() cumulative sum along each row
* 求余：np.mod(a,n)相当于a%n，np.fmod(a,n)仍为求余且余数的正负由a决定
* 计算平均值：np.mean(a)
* 计算加权平均值：np.average(a,b),其中b是权重
* 计算数组的极差：np.pth(a)=max(a)-min(a)
* 计算方差（总体方差）：np.var(a)
* 标准差：np.std(a)
* 算术平方根，a为浮点数类型：np.sqrt(a)
* 对数：np.log(a)
* 点积（计算两个数组的线性组合）：np.dot(a,b)，即得到a*b（一维上是对应元素相乘，多维可将a*b视为矩阵乘法。注意：使用*符号仍然是按位置一对一相乘
* 修剪数组，将数组中小于x的数均换为x，大于y的数均换为y：a.clip(x,y)
* 所有数组元素乘积：a.prod()
* 数组元素的累积乘积：a.cumprod()
* 数组元素的符号：np.sign(a)，返回数组中各元素的正负符号，用1和-1表示
* 数组元素分类：np.piecewise(a,[条件]，[返回值])，分段给定取值，根据判断条件给元素分类，并返回设定的返回值。
* 判断两数组是否相等： np.array_equal(a,b)
* 判断数组元素是否为实数： np.isreal(a)
* 去除数组中首尾为0的元素：np.trim_zeros(a)
* 对浮点数取整，但不改变浮点数类型：np.rint(a)
 
## 矩阵方法
### 创建矩阵
* np.mat(‘…’)通过字符串格式创建，np.mat(a)
* 通过数组创建，也可用matrix或bmat函数创建
* 创建复合矩阵：np.bmat(‘A B’,’AB’)，用A和B创建复合矩阵AB（字符串格式）
* 创建n*n维单位矩阵：np.eye(n)

### 矩阵信息
矩阵的转置：A.T
矩阵的逆矩阵：A.I
计算协方差矩阵：np.cov(x)，np.cov(x,y)
计算矩阵的迹（对角线元素和）：a.trace()
相关系数：np.corrcoef(x,y) 默认的是皮尔森相关系数，也可以用scipy.stats.spearmanr
给出对角线元素：a.diagonal()
转换为list a.tolist()

## 多项式
多项式拟合：poly= np.polyfit(x,a,n),拟合点集a得到n级多项式，其中x为横轴长度，返回多项式的系数
多项式求导函数：np.polyder(poly),返回导函数的系数
得到多项式的n阶导函数：多项式.deriv(m = n)
多项式求根：np.roots(poly)
多项式在某点上的值：np.polyval(poly,x[n]),返回poly多项式在横轴点上x[n]上的值
两个多项式做差运算： np.polysub(a,b)

五、线性代数
估计线性模型中的系数：a=np.linalg.lstsq(x,b),有b=a*x
求方阵的逆矩阵：np.linalg.inv(A)
求广义逆矩阵：np.linalg.pinv(A)
求矩阵的行列式：np.linalg.det(A)
解形如AX=b的线性方程组：np.linalg.solve(A,b)
求矩阵的特征值：np.linalg.eigvals（A）
求特征值和特征向量：np.linalg.eig(A)
Svd分解：np.linalg.svd(A)

## 概率分布
* 均匀分布：
如np.linspace(1,6,10)则得到1到6之间的均匀分布，总共返回10个数
产生二项分布的随机数：np.random.binomial(n,p,size=…)，其中n,p,size分别是每轮试验次数、概率、轮数
产生超几何分布随机数：np.random.hypergeometric(n1,n2,n,size=…)，其中参数意义分别是物件1总量、物件2总量、每次采样数、试验次数
### 正态分布
产生N个正态分布的随机数：np.random.normal(均值，标准差，N)
产生N个对数正态分布的随机数：np.random.lognormal(mean,sigma,N)
 multivariate_normal