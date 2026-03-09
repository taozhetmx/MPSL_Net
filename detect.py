import warnings
# warnings.filterwarnings("ignore")

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'my_model.pt')

    model.predict(
        source=r'',
        save=True,
        show=False,
        project='runs/detect/',
    )