from ultralytics import YOLO


if __name__=='__main__':

    # load a model;
    model = YOLO(model=r'ultralytics-8.3.94/yolo11n.pt')
    model.predict(source=r'D:\python_project\Multimodal-Large-Model-Driven-Comprehensive-Road-Condition-Recognition'
                         r'\ultralytics-8.3.94\ultralytics\assets\acc5.jpg',
                  save=True,
                  show=True
                  )