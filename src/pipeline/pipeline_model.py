import sys
import cv2
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Any, Optional
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from classifier.model import BrailleDotNet
from pipeline.translator import BrailleTranslator

class EndToEndPipeline:
    def __init__(self, config_path: str | Path = "configs/pipeline_config.yaml") -> None:
        config_path = Path(config_path).resolve()
        project_root = config_path.parent.parent

        with config_path.open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        inf_cfg = self.config["inference"]
        geo_cfg = self.config["geometry"]
        trl_cfg = self.config["translation"]

        self.device = torch.device(inf_cfg["device"] if torch.cuda.is_available() else "cpu")
        self.target_shape = tuple(geo_cfg["target_shape"])
        self.margin_pct = geo_cfg["margin_pct"]
        self.logit_threshold = inf_cfg["classifier_logit_threshold"]
        self.batch_size = inf_cfg["batch_size"]

        self.translator = BrailleTranslator(project_root / trl_cfg["language_map_json"])
        
        self.detector = YOLO(str(project_root / self.config["models"]["detector_weights"]))

        self.classifier = BrailleDotNet(num_classes=6)
        cls_weights = project_root / self.config["models"]["classifier_weights"]
        self.classifier.load_state_dict(torch.load(cls_weights, map_location=self.device, weights_only=True))
        self.classifier.to(self.device).eval()

    def _extract_crop_fast(self, image_gray: np.ndarray, bbox: tuple[int, int, int, int]) -> Optional[np.ndarray]:
        th, tw = self.target_shape
        ih, iw = image_gray.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1

        if bw <= 0 or bh <= 0:
            return None

        if (bw / bh) > (tw / th):
            nh = round(bw * th / tw)
            cy = y1 + bh // 2
            y1, y2 = cy - nh // 2, cy + (nh - nh // 2)
        else:
            nw = round(bh * tw / th)
            cx = x1 + bw // 2
            x1, x2 = cx - nw // 2, cx + (nw - nw // 2)

        vx1, vy1, vx2, vy2 = max(0, x1), max(0, y1), min(iw, x2), min(ih, y2)
        crop = image_gray[vy1:vy2, vx1:vx2]

        if crop.size == 0:
            return None

        if any(v != p for v, p in zip((vx1, vy1, vx2, vy2), (x1, y1, x2, y2))):
            pad_t, pad_b = max(0, -y1), max(0, y2 - ih)
            pad_l, pad_r = max(0, -x1), max(0, x2 - iw)
            bg = int(np.median(crop)) if crop.size > 10 else 128
            crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=bg)

        interp = cv2.INTER_AREA if crop.shape[1] > tw else cv2.INTER_CUBIC
        return cv2.resize(crop, (tw, th), interpolation=interp)

    @staticmethod
    def _bits_to_dots(binary: np.ndarray) -> str:
        dots = "".join(str(i + 1) for i, v in enumerate(binary) if v == 1)
        return dots if dots else "0"

    @torch.inference_mode()
    def process_image(self, image_bgr: np.ndarray, translate: bool = False) -> list[dict[str, Any]]:
        det_res = self.detector.predict(
            image_bgr, 
            conf=self.config["inference"]["detector_conf_threshold"],
            iou=self.config["inference"]["detector_iou_threshold"],
            max_det=1500, 
            verbose=False
        )[0]

        if not det_res.boxes:
            return []

        boxes = det_res.boxes.xyxy.cpu().numpy().astype(int)
        confs = det_res.boxes.conf.cpu().numpy()
        
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        crops, valid_idx = [], []

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            mw, mh = int((x2 - x1) * self.margin_pct), int((y2 - y1) * self.margin_pct)
            if (crop := self._extract_crop_fast(image_gray, (x1 - mw, y1 - mh, x2 + mw, y2 + mh))) is not None:
                crops.append(crop)
                valid_idx.append(i)

        if not crops:
            return []

        batch = torch.from_numpy(np.stack(crops)).float().unsqueeze(1).to(self.device)
        batch = (batch / 127.5) - 1.0

        preds = []
        for s in range(0, batch.size(0), self.batch_size):
            logits = self.classifier(batch[s:s + self.batch_size])
            preds.append((logits > self.logit_threshold).int().cpu().numpy())
        
        preds_all = np.concatenate(preds, axis=0)
        
        results = []
        for local_i, orig_i in enumerate(valid_idx):
            dots = self._bits_to_dots(preds_all[local_i])
            res = {
                "bbox": tuple(boxes[orig_i]),
                "dots": dots,
                "det_conf": float(confs[orig_i])
            }
            if translate:
                res["char"] = self.translator.translate(dots)
            results.append(res)

        return results