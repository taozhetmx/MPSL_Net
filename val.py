from ultralytics import YOLO
import warnings
#warnings.filterwarnings("ignore")
def main():
    model = YOLO("my_model.pt")
    metrics = model.val(data="jyz.yaml",
                        project='runs/val',
                        batch=1)

if __name__ == '__main__':
    main()