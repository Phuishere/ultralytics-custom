import argparse
from ultralytics import YOLO
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Training Pipeline")

    parser.add_argument("--model", type=str, required=True,
                        help="YOLO model YAML path (e.g., yolo11n.yaml)")

    parser.add_argument("--weights", type=str, default=None,
                        help="Optional pretrained weight file (e.g., yolo11n.pt)")

    parser.add_argument("--data", type=str, required=True,
                        help="Dataset YAML (e.g., coco8.yaml)")

    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")

    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size")

    parser.add_argument("--device", type=str, default="cpu",
                        help="Compute device: cpu, cuda, 0, 1, etc.")

    parser.add_argument("--predict", type=str, default=None,
                        help="Optional: run inference on an image after training")

    parser.add_argument("--export_format", type=str, default="onnx",
                        choices=["onnx", "engine", "torchscript", "saved_model"],
                        help="Export format")

    parser.add_argument("--export_path", type=str, default="exported",
                        help="Directory to save exported model")

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output folder
    export_dir = Path(args.export_path)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model YAML: {args.model}")
    model = YOLO(args.model)

    # Load weights if provided
    if args.weights:
        print(f"Loading weights: {args.weights}")
        model = model.load(args.weights)

    # Train
    print("\n=== Training Model ===")
    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
    )

    # Validate
    print("\n=== Validating Model ===")
    metrics = model.val()
    print(metrics)

    # Predict (optional)
    if args.predict:
        print(f"\n=== Predicting on image: {args.predict} ===")
        results = model(args.predict)
        results[0].show()

    # Export model
    print(f"\n=== Exporting Model to {args.export_format} ===")
    export_file = model.export(
        format=args.export_format,
        imgsz=args.imgsz,
        opset=12, # recommended for ONNX
        project=str(export_dir),
        name="model_export"
    )
    print(f"Exported model saved at: {export_file}")


if __name__ == "__main__":
    main()