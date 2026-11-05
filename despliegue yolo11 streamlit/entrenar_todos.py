"""Entrena los 6 modelos: YOLO11 [n|s] x [deteccion | seg | pose] con early stopping.
Uso: python entrenar_todos.py <carpeta_trabajo> [epocas]
Guarda best.pt de cada uno en modelos/ y un resumen de metricas en metricas.json.
"""
import json
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

def main():
    work = Path(sys.argv[1]).resolve()
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    here = Path(__file__).parent
    (here / "modelos").mkdir(exist_ok=True)

    DATA = {"detect": "dataset_detect", "seg": "dataset_seg", "pose": "dataset_pose"}
    SUFFIX = {"detect": "", "seg": "-seg", "pose": "-pose"}
    metrics_file = here / "metricas.json"
    metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}

    for size in ("n", "s"):
        for task in ("detect", "seg", "pose"):
            name = f"yolo11{size}{SUFFIX[task]}"
            if name in metrics:
                continue
            model = YOLO(f"{name}.pt")
            model.train(
                data=str(work / DATA[task] / "data.yaml"), epochs=epochs, imgsz=640,
                batch=16, patience=8, workers=4, device=0, plots=True,
                project=str(work / "runs"), name=name, exist_ok=True, amp=True,
            )
            best = work / "runs" / name / "weights" / "best.pt"
            shutil.copy2(best, here / "modelos" / f"{name}.pt")
            val = YOLO(str(best)).val(data=str(work / DATA[task] / "data.yaml"), split="test", device=0)
            metrics[name] = {k: round(float(v), 4) for k, v in val.results_dict.items()}
            metrics_file.write_text(json.dumps(metrics, indent=2))
            print("TERMINADO", name, metrics[name], flush=True)


if __name__ == "__main__":
    main()
