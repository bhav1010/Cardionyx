import os
from ultralytics import YOLO


def main():
    model = YOLO("yolo11s.pt") 

    data_yaml_path = r"c:\Users\10bha\OneDrive\Documents\projects\Cardionyx\Cardionyx\Cardionyx.v1i.yolo26\data.yaml"

    save_dir = r"c:\Users\10bha\OneDrive\Documents\projects\Cardionyx\Cardionyx\training_runs"
    os.makedirs(save_dir, exist_ok=True)


    epochs = 950          
    imgsz = 640           
    batch = 16            
    device = '0'            
    lr0 = 0.01            
    optimizer = 'auto'    

    print(f"Starting YOLO training for {epochs} epochs...")
    print(f"Models will be logically saved inside: {save_dir}")

    # 5. Start Training
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        lr0=lr0,
        optimizer=optimizer,
        project=save_dir,    
        name="yolo_run",      
        exist_ok=False        
    )
    
    print(f"\nTraining complete! Check '{save_dir}' for your models and validation results.")

if __name__ == "__main__":
    main()
