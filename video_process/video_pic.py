
import cv2
import os
# 获取一个视频打开cap 参数1 file name
#cap = cv2.VideoCapture("1.mp4")
#cv2.VideoCapture(0, cv2.CAP_DSHOW) # 摄像头截取
cap = cv2.VideoCapture('test.mp4')
isOpened = cap.isOpened # 判断是否打开‘
print(isOpened)
# 获取信息 宽高
n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('总帧数：',n_frame) # 整个视频的总帧数
fps = cap.get(cv2.CAP_PROP_FPS) # 帧率 每秒展示多少张图片
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # w
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # h
print('帧数、宽度、高度分别为：',fps,width,height) # 帧率 宽 高
i = 0 # 记录读取多少帧
frameFrequency = 1 # 每frameFrequency保存一张图片
x = 0
while(isOpened):

    # 结束标志是否读取到最后一帧
    if i == n_frame:
        break
    else:
        i = i+1
    (flag,frame) = cap.read() # read方法 读取每一张 flag是否读取成功 frame 读取内容
    fileName = str(x)+'_.jpg' # 名字累加

    # True表示读取成功 进行·写入
    # if 判断需要有冒号
    #if flag == True:
    outPutDirName = './outputs/' # 设置保存路径
    if not os.path.exists(outPutDirName):
        # 如果文件目录不存在则创建目录
        os.makedirs(outPutDirName)
    if i % frameFrequency == 0:
        print(fileName)
        cv2.imwrite(outPutDirName+fileName,frame,[cv2.IMWRITE_JPEG_QUALITY,100])# 质量控制 100最高
        x += 1
print('end!')
