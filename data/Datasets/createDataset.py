import os
import cv2
import json
import shutil
import zipfile
from tqdm import tqdm

DATASET_ORIGINAL_FOLRDER_PATH = "./datasets/095.교통사고_영상_데이터/01.데이터"
VIDEO_ZIP_FOLDERS = ['1.Training/원천데이터_231108_add', '2.Validation/원천데이터_231108_add']
LABEL_ZIP_FOLDER = ['1.Training/라벨링데이터_231108_add', '2.Validation/라벨링데이터_231108_add']

SAVE_VIDEO_FOLRDER = "./datasets/video_original"
SAVE_RESIZED_VIDEO_FOLRDER = "./datasets/video_resized"
SAVE_LABEL_FOLRDER = "./datasets/label"


# Check Folder Exist
def checkFolder():

    # Check Video Zip Folder is Exist
    for _vid_folder in VIDEO_ZIP_FOLDERS:
        _full_path = os.path.join(DATASET_ORIGINAL_FOLRDER_PATH, _vid_folder)

        if os.path.isdir(_full_path):
            print(f"[createDataset.py] Folder {_full_path} Found!")
        
        else:
            print(f"[createDataset.py][Error] Folder {_full_path} Not Exist!")
            exit(-1)

    # Check Label Zip Folder is Exist
    for _label_folder in LABEL_ZIP_FOLDER:
        _full_path = os.path.join(DATASET_ORIGINAL_FOLRDER_PATH, _label_folder)

        if os.path.isdir(_full_path):
            print(f"[createDataset.py] Folder {_full_path} Found!")
        
        else:
            print(f"[createDataset.py][Error] Folder {_full_path} Not Exist!")
            exit(-1)

    # Check Save Folder for Videos is Empty
    if os.path.isdir(SAVE_VIDEO_FOLRDER):
        print(f"[createDataset.py][Error] Save Folder for Videos {SAVE_VIDEO_FOLRDER} is Not Empty")
        print("[createDataset.py][Error] Please Make Save folrder for Videos Empty!")
        exit(-1)
    else:
        os.mkdir(SAVE_VIDEO_FOLRDER)
        print(f"[createDataset.py] Create Save Folder for Videos {SAVE_VIDEO_FOLRDER}")

    # Check Save Folder for Resized Videos is Empty
    if os.path.isdir(SAVE_RESIZED_VIDEO_FOLRDER):
        print(f"[createDataset.py][Error] Save Folder for Resized Videos {SAVE_RESIZED_VIDEO_FOLRDER} is Not Empty")
        print("[createDataset.py][Error] Please Make Save folrder for Resized Videos Empty!")
        exit(-1)
    else:
        os.mkdir(SAVE_RESIZED_VIDEO_FOLRDER)
        print(f"[createDataset.py] Create Save Folder for Resized Videos {SAVE_RESIZED_VIDEO_FOLRDER}")
        
    # Check Save Folder for Labels is Empty
    if os.path.isdir(SAVE_LABEL_FOLRDER):
        print(f"[createDataset.py][Error] Save Folder for Labels {SAVE_LABEL_FOLRDER} is Not Empty")
        print("[createDataset.py][Error] Please Make Save folrder for Labels Empty!")
    else:
        os.mkdir(SAVE_LABEL_FOLRDER)
        print(f"[createDataset.py] Create Save Folder for Videos {SAVE_LABEL_FOLRDER}")


# Unzip Video Label .zip Files
def unzipFile():
    for _vid_folder in VIDEO_ZIP_FOLDERS:
        _curr_vid_folder = os.path.join(DATASET_ORIGINAL_FOLRDER_PATH, _vid_folder)

        print(f"[createDataset.py] Extract Videos .zip Files from {_curr_vid_folder} to {SAVE_VIDEO_FOLRDER}")

        for _file in tqdm(os.listdir(os.path.join(_curr_vid_folder))):
            _zip_path = os.path.join(_curr_vid_folder, _file)
            
            with zipfile.ZipFile(_zip_path, 'r') as zip_ref:
                zip_ref.extractall(SAVE_VIDEO_FOLRDER)
    
    for _label_folder in LABEL_ZIP_FOLDER:
        _curr_label_folder = os.path.join(DATASET_ORIGINAL_FOLRDER_PATH, _label_folder)

        print(f"[createDataset.py] Extract Videos .zip Files from {_curr_label_folder} to {SAVE_LABEL_FOLRDER}")

        for _file in tqdm(os.listdir(os.path.join(_curr_label_folder))):
            _zip_path = os.path.join(_curr_label_folder, _file)
            
            with zipfile.ZipFile(_zip_path, 'r') as zip_ref:
                zip_ref.extractall(SAVE_LABEL_FOLRDER)

                
def videoFiltering():

    total_removed_video = 0

    for _vid_file in tqdm(os.listdir(SAVE_VIDEO_FOLRDER)):
        _vid_file_name = _vid_file[:-4]
        _vid_path = os.path.join(SAVE_VIDEO_FOLRDER, _vid_file_name + ".mp4")
        _label_path = os.path.join(SAVE_LABEL_FOLRDER, _vid_file_name + ".json")

        with open(_label_path, 'r') as f:
            json_data = json.load(f)

            if 'filming_way' not in json_data['video'].keys():
                os.remove(_vid_path)
                os.remove(_label_path)
                total_removed_video += 1
                continue

            if 'video_point_of_view' not in json_data['video'].keys():
                os.remove(_vid_path)
                os.remove(_label_path)
                total_removed_video += 1
                continue

            if json_data['video']['filming_way'] != 'bb':
                os.remove(_vid_path)
                os.remove(_label_path)
                total_removed_video += 1
                continue

            if json_data['video']['video_point_of_view'] != 1:
                os.remove(_vid_path)
                os.remove(_label_path)
                total_removed_video += 1
                continue
    
    print(f"[createDataset.py][Alert] Video Filtering : Total {total_removed_video} Videos Removed !")


# Resize & Cut off (0~t sec) Video
def resizeAndCutOffVideos(w: int, h: int, t: int):
    for _vid_file in tqdm(os.listdir(SAVE_VIDEO_FOLRDER)):

        # Get Original Video Path and Save path
        _file_path = os.path.join(SAVE_VIDEO_FOLRDER, _vid_file)
        _save_path = os.path.join(SAVE_RESIZED_VIDEO_FOLRDER, _vid_file)

        # Construct Video Reader & Get Video Setting
        _cap = cv2.VideoCapture(_file_path)
        _original_w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        _original_h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _fps = _cap.get(cv2.CAP_PROP_FPS)
        _total_frame_cnt = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # if Video Length(sec) is not longer that 't' sec, we don't use this video
        if _total_frame_cnt < _fps * t:
            print(f"[createDataset.py][Alert] Video is not longer than {t} sec.")
            continue

        # if Video Already (1920x1080) Size, Just Copy File and Skip Resize Process
        if (_original_w == w) and (_original_h == h):
            shutil.copyfile(_file_path, _save_path)
            continue
        
        # Contruct Video Writer
        _out = cv2.VideoWriter(
                _save_path,
                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                _fps, 
                (w, h),
                True
            )
        
        _frame_cnt = 0

        while True:
            # Read Frame
            ret, frame = _cap.read()
            if not ret:
                break
            
            # Resize Frame and Write
            _resized_frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            _out.write(_resized_frame)
            _frame_cnt += 1

            # Save Resized Video during t sec
            if _frame_cnt > (_fps * t):
                break

        # Release All Reader, Writer
        _cap.release()
        _out.release()

# Check Resized & Cut off video's width, height, fps, number of frames
def checkResizedVideos(w: int, h: int, t: int):

    total_removed_video = 0

    for _vid_file in tqdm(os.listdir(SAVE_RESIZED_VIDEO_FOLRDER)):

        # Get Resized & Cut off Video's Setting
        _file_path = os.path.join(SAVE_RESIZED_VIDEO_FOLRDER, _vid_file)
        _cap = cv2.VideoCapture(_file_path)
        _resized_w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        _resized_h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _fps = _cap.get(cv2.CAP_PROP_FPS)
        _total_frame_cnt = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _cap.release()

        if (_resized_w != w) or (_resized_h != h):
            print(f"[createDataset.py][Error] Video Shape is Not Correct. ({_resized_w}, {_resized_h}) != ({w},{h}). {_file_path}")
        
        if _total_frame_cnt != (_fps * t):
            print(f"[createDataset.py][Error] Video's number of Frames is Not Correct. {_total_frame_cnt} != {_fps * t}. {_file_path}")
            os.remove(_file_path)
            total_removed_video += 1

    print(f"[createDataset.py][Alert] checkResizedVideos : Total {total_removed_video} Videos Removed !")
    


if __name__ == "__main__":
    print("[createDataset.py] Please Run createDataset.py in Utils Folder")
    checkFolder() # 위에 있는 모든 폴더 경로들 존재하는지 확인하는 함수
    unzipFile() # zip 파일 unzip하는 함수
    videoFiltering() # 필요한 영상만 필터링하는 함수
    resizeAndCutOffVideos(w=1920, h=1080, t=10) # resize(w, h) & 영상 길이 0~10초만 사용하도록 cutoff하는 함수 
    checkResizedVideos(w=1920, h=1080, t=10) # 싹 다 체크하는 함수
    
    
    # result
    
    # video_original 총 개수 : 14212개
    # label 총 개수 : 14212개
    # 영상 길이가 10초가 안되는 영상 : 3개
    # 비디오 프레임 개수 일치 X 영상 : 14개
    # video_resized 총 개수 : 14212-3-14 = 14195