# Albumentations Library with YoloV5 or YoloV8
- 여러가지 augmentation을 쉽게 구현해 줌
- augmentation 수행에 맞게 자동으로 bounding box도 알맞게 자동 수정하여 처리해 줌
- model 별로 bounding box 좌표를 처리하는 방식이 다르기 때문에 주의하여야 함
- Albumentations is a Python library for image augmentation that offers a simple and flexible way to perform a variety of image transformations.


# 원본 출처
- https://github.com/muhammad-faizan-122/yolo-data-augmentation
- 수행한 프로젝트에 알맞게 수정함


# 수정 내용
- workflow.py
    - 파일 이름 중간에 .이 있을 경우 에러가 날 수 있기에, rsplit으로 변경함
- apply_album_aug.py
    - single_obj_bb_yolo_conversion 함수 부분에 괄호 위치 잘못된 것 수정함
- draw_volo 메서드에서 colab용으로 cv2.imshow -> cv2_imshow로 바꿈 (colab안 쓸 경우는 다시 바꿔야 함)
- 하고자 하는 프로젝트에 맞게 자잘한 오류 등을 수정함


# 참고 사항
- windows 환경에서는(아마?) validate_results.py의 아래와 같은 cv2관련 에러가 날 때가 있음
    cv2.error: OpenCV(4.8.0) D:\a\opencv-python\opencv-python\opencv\modules\highgui\src\window.cpp:1272: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'
    * opencv version 문제로 판단됨, pip uninstall opencv-python -> pip install opencv-python으로 해결 가능함
- 코랩, 주피터 노트북에서 실행시 변환 과정 시각적으로 확인 가능 (shell 실행으로는 불가능)
- shear 등은 자동 바운딩 박스 변환까지는 안 됨(불가능한 것인지 코드 수정으로 고칠 수 있는 부분인지는 테스트해봐야 함, 그리고 rotation의 각도를 90도 등이 아닌 중간 각도로 할 경우도 테스트 해보아야 함)
- yolo의 mosaic과 비슷한 기능인 RandomGridShuffle 같은 경우, bounding box자동 처리까지는 지원하지 않는듯함(증식 이미지에 대한 수작업 요함)


# 파일 설명
- contants_ori.yaml 파일을 수정하여 contants.yaml로 저장후 사용하면 됨
    - input-ds : contain the input of YOLOv8 and YOLOv5 which are following directories.
        - input-ds / images : image 파일 경로
            Images directory contains the images
        - input-ds / labels : label 파일 경로(yolo의 경우 txt 파일)
            labels directory contains the .txt files. Each .txt file contains the normalized bounding boxes in a following format.

    - out-aug-ds : contain the augmented output contains following directories.
        - out-aug-ds / images : augmentation 적용된 image 출력 결과물 저장 경로 
            Images directory contains the augmented images.
        - out-aug-ds / labels : augmentation 적용된 label 출력 결과물 저장 경로
            labels directory contains the augmented labels.
    - transformed_file_name: use to name augmented output to differentiate from other input dataset.
    - CLASSES: list of input class name according to class number. 

- workflow.py : contain the pipeline to get the desired results.
- get_album_bb.py : it is used to get labels in albumentation format from input yolo format.
- apply_album_aug.py : contain the augmentated operations.
    - album_to_yolo_bb.py : it is used to convert to labels in albumentation format to yolo format
    - save_aug.py : to save the augmented results.
    - validate_results.py : draw the augmented labels on augmented image to visualize the results.

# 사용법
- install requirements using ```pip install -r requirements.txt```
- provide the input and output path in **CONSTANT.yaml** file.
- update the name of transformed_file_name in CONSTANT.yaml otherwise code will overwrite last augmentations.
- Provide the list of classes in CONSTANT.yaml in a sequence as use to assign class number in yolo dataset labelling. 
    - For example, you provided class list is ```['obj1', 'obj2', 'obj3']``` class number used for obj1 in label file should be 0, similarly for 'obj2' class number should be 1 and so on.
- run the pipeline using ```python3 run.py```- 
- model/YoloV8/semiconductor_detection_git.ipynb로 돌려도 import해서 돌려도 됨
- model/YoloV8/semiconductor_detection_colab.ipynb로 colab이나 jupyter notebook에서 코드로만 돌려도 됨 (빠르고, 작업하기 좋음)
