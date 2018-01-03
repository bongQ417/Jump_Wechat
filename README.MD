# 微信小程序jump游戏辅助

# 安装WebDriverAgent
* WebDriverAgent
* XCode
* 真机测试

https://github.com/openatx/facebook-wda/blob/master/README.md
https://testerhome.com/topics/9854

# python图片解析学习
* opencv
* findContours、GaussianBlur

https://blog.callmewhy.com/2016/04/23/opencv-find-qrcode-position/
https://segmentfault.com/a/1190000003742461
http://python.jobbole.com/85600/
http://blog.csdn.net/sinat_31135199/article/details/51252393

# 算法与思路
* 通过wda可以截屏与长按操作
* 解析图片
* 识别棋子位置,深紫色可以很容易识别
* 识别后剔除棋子,避免影响后续棋盘的计算
* 通过轮廓识别获得棋盘轮廓
* 从上至下,获取棋盘顶点,再通过顶点为中心轴,计算距离两边的长度
* 当长度达到最长时,该Y轴就是棋盘中心位置
* 计算棋盘与棋子位置,获取距离,在通过测试获得距离与长按时间的线性关系