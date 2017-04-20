# pandas基本方法
## 工具类
### 分割合并
* pd.cut() 拆成bin： bins = [18, 25, 35, 60, 100] cats = pd.cut(ages, bins)
返回一个特殊的Categorical对象，levels为分类名称，labels为结果。
* pd.qcut(df['Column'], 4) 根据样本分位数对数据进行bin划分,大小基本相等
* pd.concat([example1, example2], axis=1) 沿轴拼接，axis=1为纵向
pieces = [df[:3], df[3:7], df[7:]]; pd.concat(pieces)
* pd.merge(left, right, on='key') 通过键拼接列，键为列名

## DataFrame
### IO
* DataFrame(np.ndarray,index=,columns=) # index 是行索引 columns是列名
* pd.read_csv()
* to_csv()
* dataframe转化为numpy.ndarray: example.values[:, :]


### 增删改查
* 添加行 s = df.iloc[3]  df.append(s, ignore_index=True)
* 添加列 


+ del()
+ df.drop(["column"],axis=1) axis=1是删除列

* df.at[dates[0],'A'] = 0 按标签赋值
* df.iat[0,1] = 0 按位置赋值
* df.loc[:,'D'] = np.array([5] * len(df)) 按列赋值
* df2[df2 > 0] = -df2 赋值

+ df['A'] 获取A列，等价于 df.A
+ df.loc[:,['A','B']] 按标签查询。df.at[rows[0],'A'] 获取标量值，等价于df.loc[rows[0],'A'] 需要注意的是，dataframe的索引[1:3]是包含1,2,3的，与一般python不同。
+ df.iloc[3:5,0:2] 按位置查询，或者df.iloc[[1,2,4],[0,2]]。df.iat[1,1]可获取标量
+ df.ix(1) df.ix('e') 混合索引,先用loc的方式来索引，索引失败就转成iloc的方式
+ df.iterrows() 返回(index, Series)对,可对DataFrame进行遍历。for循环遍历数据时一定要使用.iterrows()，或用.values()转换成ndarray再遍历，否则遍历的只是df的columns names。

### 处理数据
* df.dropna(how='any')
* df.fillna()
* pd.isnull(df)
* reset_index(drop=True) drop=True表示列中不包含索引
* astype(numpy.dtype) 将数据转换成numpy.dtype
* df.drop_duplicates() 去重
* df.map({}) 传入字典或函数。df['food'].map(str.lower).map({'A':'ab'}})
* df.replace() data.replace([-999, -1000], np.nan)

### 运算
计算时一般不包括丢失的数据
* df.T 转置
* df.sort_index(axis=1, ascending=False) 按轴排序
* df.sort_values(['A','B'],ascending=True) 按值排序
* df.apply() 在数据上使用函数
* df.mean() 

### 布尔索引
* df[df.A > 0] 布尔索引 df2[df2['E'].isin(['two','four'])]
*  data.loc[(data["Gender"]=="Female") & (data["Education"]=="Not Graduate") & (data["Loan_Status"]=="Y"), ["Gender","Education","Loan_Status"]] 



### 统计作图
* df.index df.columns df.values values是将原本dataframe数据强制转化为numpy格式的数据来索引
* df.describe()
* pd.crosstab(df["column1","column2"])
* df.value_counts()
* df.groupby() 生成一个groupby对象，可调用其mean方法来计算分组平均值,size()返回大小。 dict(list(df.groupby('key1')))可以做成字典。
*  data.boxplot(column="ApplicantIncome",by="Loan_Status") 
*  data.hist(column="ApplicantIncome",by="Loan_Status",bins=30) 


## Series
相当于是一维的DataFrame

### 增删改查
* 遍历 s.iteritems() 返回(index, value)对
对dataframe数据的index统一加一个后缀

比如对原本dataframe下的index=[‘aa’, ‘cc’, ‘dddddd’]的，统一加上一个_5m的后缀，通常的操作大家一般就是直接example.index = [x + ‘_5m’ for x in example.index]，这个其实会产生些小问题，因为默认的index是pandas.indexes.base.Index，这个格式可能会默认index里面数据的长度是确定的，导致加_5m后缀失败，所以需要先把格式强制转化为list, 像这样：example.index = [x + ‘_5m’ for x in list(example.index)]

## Categorical