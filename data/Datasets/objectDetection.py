import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import shutil
import multiprocessing as mp
from transformers import DetrImageProcessor, DetrForObjectDetection

SAVE_RESIZED_VIDEO_FOLRDER = ".datasets/video_resized"
SAVE_OBJFILTER_VIDEO_FOLDER = "./datasets/video_obj"
SAVE_OBJFILTER_RESIZED_VIDEO_FOLDER = "/./datasets/video_obj_resized"


class ObjectDetector:

    def __init__(self, device):
        self.device = device
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            revision="no_timm",
            use_safetensors=False
        ).to(device=device)

    @torch.no_grad()
    def detect(self, x):
        _size, _channel, _height, _width = x.shape
        inputs = self.processor(images=x, return_tensors='pt')
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        results = self.processor.post_process_object_detection(outputs, threshold=0.9)

        for imgidx in range(_size):
            result_dict = results[imgidx]
            mask = torch.zeros(_channel, _height, _width).to(self.device)
            for score, label, box in zip(result_dict["scores"], result_dict["labels"], result_dict["boxes"]):
                label = self.model.config.id2label[label.item()]
                if label in ['car', 'truck', 'bus', 'traffic light']:
                    boxes = [round(i, 2) for i in box.tolist()]
                    x0 = int(boxes[0] * _width)
                    y0 = int(boxes[1] * _height)
                    x1 = int(boxes[2] * _width)
                    y1 = int(boxes[3] * _height)
                    mask[:, y0:y1, x0:x1] = 1
            x[imgidx] *= mask
        
        return x


def objectFilteringWorker(video_list, device, frames=16):
    ObjDet = ObjectDetector(device)
    
    for _vid in tqdm(video_list, desc=f"ObjectFiltering-{device}"):
        _full_path = os.path.join(SAVE_RESIZED_VIDEO_FOLRDER, _vid)
        _save_path = os.path.join(SAVE_OBJFILTER_VIDEO_FOLDER, _vid)

        __frames = []
        cap = cv2.VideoCapture(_full_path)

        # Construct Video Reader & Get Video Setting
        _original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        _original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Contruct Video Writer
        _out = cv2.VideoWriter(
            _save_path,
            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
            frames // 8,
            (_original_w, _original_h),
            True
        )

        while True:
            ret, _frame = cap.read()
            if not ret:
                break

            # Read Video RGB Order
            _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
            __frames.append(_frame)

        cap.release()

        __video_length = len(__frames)
        __step_size = max(1, __video_length // frames)

        # Sampling Frames evenly from Video ( Total self.frame Frames)
        __frames = np.array(__frames)[:__step_size * frames:__step_size].astype(np.float32)
        __frames = __frames.transpose(0, 3, 1, 2)  # T C H W

        __frames_tensor = torch.from_numpy(__frames).to(device)
        _frames_obj = ObjDet.detect(__frames_tensor).cpu().detach().numpy().transpose(0, 2, 3, 1).astype(np.uint8)

        for _frame_idx in range(_frames_obj.shape[0]):
            _out.write(cv2.cvtColor(_frames_obj[_frame_idx], cv2.COLOR_RGB2BGR))
        
        _out.release()


def objectFiltering(frames=16):
    video_list = sorted(os.listdir(SAVE_RESIZED_VIDEO_FOLRDER))

    # Split video list into 2 parts for 2 GPUs
    video_list_gpu0 = video_list[0::2]
    video_list_gpu1 = video_list[1::2]

    ctx = mp.get_context("spawn")

    p0 = ctx.Process(target=objectFilteringWorker, args=(video_list_gpu0, "cuda:0", frames))
    p1 = ctx.Process(target=objectFilteringWorker, args=(video_list_gpu1, "cuda:1", frames))

    p0.start()
    p1.start()

    p0.join()
    p1.join()


def resizing(img_size=224):
    video_list = os.listdir(SAVE_OBJFILTER_VIDEO_FOLDER)
    
    for _vid in tqdm(video_list):
        _full_path = os.path.join(SAVE_OBJFILTER_VIDEO_FOLDER, _vid)
        _save_path = os.path.join(SAVE_OBJFILTER_RESIZED_VIDEO_FOLDER, _vid)

        cap = cv2.VideoCapture(_full_path)

        # Construct Video Reader & Get Video Setting
        _fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Contruct Video Writer
        _out = cv2.VideoWriter(
            _save_path,
            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
            _fps,
            (img_size, img_size),
            True
        )

        while True:
            ret, _frame = cap.read()
            if not ret:
                break

            # Read Video RGB Order
            _frame = _out.write(cv2.resize(_frame, (img_size, img_size)))

        cap.release()
        _out.release()
        

if __name__ == "__main__":
    
    if os.path.exists(SAVE_OBJFILTER_VIDEO_FOLDER):
        shutil.rmtree(SAVE_OBJFILTER_VIDEO_FOLDER)

    if os.path.exists(SAVE_OBJFILTER_RESIZED_VIDEO_FOLDER):
        shutil.rmtree(SAVE_OBJFILTER_RESIZED_VIDEO_FOLDER)
        
    os.mkdir(SAVE_OBJFILTER_VIDEO_FOLDER)
    os.mkdir(SAVE_OBJFILTER_RESIZED_VIDEO_FOLDER)
    
    objectFiltering()
    resizing()
    
    # result
    
    # video_obj 총 개수 : 14195개
    # video_obj_resized 총 개수 : 14195개