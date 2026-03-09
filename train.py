import warnings
#warnings.filterwarnings("ignore")
from ultralytics import YOLO
if __name__=='__main__':
    model=YOLO(model=r'my_model.yaml')
    #model.load('yolov12n.pt') #加载预训练权重，改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    model.train(data=r'jyz.yaml',
                epochs=300,
                batch=16,
                workers=0,
                imgsz=640,
                device="0",
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/my_model/train',
                name='exp',
                single_cls=False,
                cache=False,
                amp=False,
                )
