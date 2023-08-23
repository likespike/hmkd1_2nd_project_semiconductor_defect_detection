# pipeline 실행
from controller.apply_album_aug import apply_aug
from controller.get_album_bb import get_bboxes_list
import cv2
import os
import yaml

with open("contants.yaml", 'r') as stream:
    CONSTANTS = yaml.safe_load(stream)  # 변수에 yaml파일 dictionary 정보가 들어감

def run_pipeline():
    imgs = os.listdir(CONSTANTS["inp_img_pth"])
    for img_file in imgs:
        # file_name = img_file.split('.')[0]    # 파일 이름 중간에 .이 있을 경우는 에러가 나서 아래와 같이 rsplit으로 변경해 줌
        file_name = img_file.rsplit('.', 1)[0]    # 변경 부분
        # aug_file_name = file_name + "_" + CONSTANTS["transformed_file_name"]    # 접미어 붙이고 싶을 때
        aug_file_name = CONSTANTS['pre'] + file_name    # 접두어 붙이고 싶을 때
        image = cv2.imread(os.path.join(CONSTANTS["inp_img_pth"], img_file))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        # 이 부분은 실험 결과 안해줘도 됨
        lab_pth = os.path.join(CONSTANTS["inp_lab_pth"], file_name + '.txt')
        album_bboxes = get_bboxes_list(lab_pth, CONSTANTS['CLASSES'])   # 바운딩 박스 정보(좌표 4개, 클래스)
        apply_aug(image, album_bboxes, CONSTANTS["out_lab_pth"],  CONSTANTS["out_img_pth"], aug_file_name, CONSTANTS['CLASSES'])
