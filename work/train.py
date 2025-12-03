from ultralytics import YOLO
from clearml import Task
import warnings
warnings.filterwarnings('ignore')

def train_yolov13_pose():
    # -------------------------- 核心路径配置（请根据实际情况修改） --------------------------
    model_config = 'yolov13n-hand-palm8.yaml'  # 模型配置文件路径
    data_yaml = 'hand_palm8.yaml'            # 数据集配置文件路径
    # -------------------------------------------------------------------------------------

    # 训练参数设置
    pretrained_weights = 'yolov13n.pt'  # 预训练权重路径
    epochs = 20
    batch = 4
    imgsz = 640
    device = "0"  # GPU: "0", 多GPU: "0,1", CPU: "cpu"
    project = 'runs/coco8_pose_train'
    name = 'yolov13n_coco8_pose'

    # 加载模型
    model = YOLO(model_config).load(pretrained_weights)

    # 启动训练
    print(f"开始训练 | 模型配置: {model_config} | 数据集配置: {data_yaml}")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        mosaic=0.0,  # 小数据集关闭Mosaic
        mixup=0.0,
        save=True,
        val=True,
        plots=True
    )

    # 训练结果简要信息
    print(f"训练完成 | 最优权重路径: {project}/{name}/best.pt")

if __name__ == "__main__":
    task = Task.init(project_name="YOLOv13_test2", 
                 task_name="test")
    train_yolov13_pose()