import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8-MobileNetV3.yaml')
    # model.load('yolov8n.pt') # 是否加载预训练权重,科研不建议大家加载否则很难提升精度
    model.train(data=r'./ultralytics/cfg/datasets/data.yaml',
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                cache=False,
                imgsz=640,
                epochs=150,
                single_cls=True,  # 是否是单类别检测
                batch=16,
                close_mosaic=10,
                workers=0,
                device='0',
                optimizer='SGD', # using SGD
                # resume='', # 如过想续训就设置last.pt的地址
                amp=False,  # 如果出现训练损失为Nan可以关闭amp
                project='/tf_logs/train',
                name='v8_MobileNetV3',
                )