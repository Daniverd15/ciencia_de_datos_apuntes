"""Deriva los datasets de deteccion y pose a partir del dataset de segmentacion
(Roboflow cicatriz/danocarro v1, formato YOLO con poligonos).

- detect: caja = rectangulo que encierra el poligono.
- pose:   caja + 5 puntos clave derivados del poligono
          (superior, inferior, izquierdo, derecho y centroide).
          El dataset original NO trae puntos clave anotados a mano: son
          puntos geometricos del contorno, utiles para el ejercicio.

Uso: python preparar_datasets.py <carpeta_dataset_seg> <carpeta_salida>
"""
import shutil
import sys
from pathlib import Path

import numpy as np

SPLITS = {"train": "train", "valid": "valid", "test": "test"}


def parse(line):
    p = line.split()
    cls = int(p[0])
    pts = np.array(p[1:], dtype=float).reshape(-1, 2)
    return cls, pts


def keypoints(pts):
    top = pts[pts[:, 1].argmin()]
    bottom = pts[pts[:, 1].argmax()]
    left = pts[pts[:, 0].argmin()]
    right = pts[pts[:, 0].argmax()]
    center = pts.mean(axis=0)
    return [top, bottom, left, right, center]


def convert(src: Path, dst: Path, mode: str, names_block: str):
    for split in SPLITS:
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        for img in (src / split / "images").iterdir():
            shutil.copy2(img, dst / split / "images" / img.name)
        for lab in (src / split / "labels").glob("*.txt"):
            out = []
            for line in lab.read_text().splitlines():
                if len(line.split()) < 7:
                    continue
                cls, pts = parse(line)
                x0, y0 = pts.min(axis=0)
                x1, y1 = pts.max(axis=0)
                box = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
                row = [cls] + box
                if mode == "pose":
                    for kx, ky in keypoints(pts):
                        row += [kx, ky, 2]
                out.append(" ".join(f"{v:.6f}" if i else str(int(v)) for i, v in enumerate(row)))
            (dst / split / "labels" / lab.name).write_text("\n".join(out))
    extra = ""
    if mode == "pose":
        # superior, inferior, izquierdo, derecho, centro -> al voltear, izq<->der
        extra = "kpt_shape: [5, 3]\nflip_idx: [0, 1, 3, 2, 4]\n"
    (dst / "data.yaml").write_text(
        f"path: {dst.resolve().as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        f"{extra}{names_block}"
    )


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    yaml_lines = (src / "data.yaml").read_text().splitlines()
    names_block = "\n".join(l for l in yaml_lines if l.startswith(("nc:", "names:"))) + "\n"
    # el data.yaml de seg tambien necesita rutas absolutas
    (src / "data.yaml").write_text(
        f"path: {src.resolve().as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n{names_block}"
    )
    for mode in ("detect", "pose"):
        convert(src, out / f"dataset_{mode}", mode, names_block)
        print("listo", mode)
