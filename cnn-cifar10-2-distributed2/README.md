1、由于多个worker写同一个文件容易造成资源争夺的现象，原本打算采用socket通信的方式替代这个过程，后来发现socket的连接方式
并不适合这个过程，socket连接过程中下一次连接上一次连接中间必须中断，这样就会照成每次连接的端口号是不同的，不能用一次连接解决整个过程的写操作
，比较麻烦，而且容易照成端口号的资源消耗越来越大，

2、多个程序写一个文件并不是多进程通信，而是多个进程使用同一个资源，所以在最初解决这个问题的时候陷入了误区，一直在多进程通信的方式寻找答案

3、直接使用文件锁轻松解决这个问题，另一个问题是写文件的时候使用的函数是open().writelines()读文件的时候使用np.loadtxt(),容易报错，显示第二行读取的数不对
最后改掉方法，写的时候使用open.write()读的时候使用open().read(),轻松解决。

4 ps2 版本已经实现将种群更新，更新的意思就是每次变异之前都能保持当前中保留的个体是loss最小的weight
  实现的过程将ps中获得weight个体重新赋值给新的list,避免了之前的zipfile错误,现在初步实验了保持种群大小结果，后面将围绕扩充种群进行实验
  
5 ps2 之前的版本都是使用current best1变异策略，由于其它变异策略都不需要使用当前的种群索引，现在增加ps3

6 ps3开始对6种变异策略进行对比，并且把f的值由原来的0.5调整为了0.2，每一种变异策略分别进行更新种群和不更新种群的实验对比。

