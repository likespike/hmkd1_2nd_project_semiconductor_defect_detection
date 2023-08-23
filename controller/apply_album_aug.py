from controller.album_to_yolo_bb import multi_obj_bb_yolo_conversion
from controller.album_to_yolo_bb import single_obj_bb_yolo_conversion
from controller.save_augs import save_aug_image, save_aug_lab
from controller.validate_results import draw_yolo
import albumentations as A

def apply_aug(image, bboxes, out_lab_pth, out_img_pth, transformed_file_name, classes):
    transform = A.Compose([
        # A.RandomCrop(width=300, height=300),
        # A.RandomBrightnessContrast(p=-1),
        # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0),
        # A.CLAHE(clip_limit=(0, 1), tile_grid_size=(8, 8), always_apply=True),
        # A.Normalize()
        # A.RandomCropFromBorders(always_apply=True, crop_left=0.5, crop_right=0.0, crop_top=0.5, crop_bottom=0.0),
        # A.Crop(x_min=0, y_min=0, x_max=320, y_max=320, always_apply=True),  # 한 번씩 번갈아 가면서 해야 함
        # A.Crop(x_min=320, y_min=0, x_max=640, y_max=320, always_apply=True),
        # A.Crop(x_min=0, y_min=320, x_max=320, y_max=640, always_apply=True),
        # A.Crop(x_min=320, y_min=320, x_max=640, y_max=640, always_apply=True),
        # A.Resize(640, 640),
        A.Rotate(limit=[90,90], interpolation=1, border_mode=1, rotate_method='largest_box', crop_border=False, p=1.0, always_apply=True),
        # A.HorizontalFlip(always_apply=True, p=1.0),
        # A.VerticalFlip(always_apply=True, p=1.0),
        # A.RandomGridShiffle(),
        # A.RandomResizedCrop(),
        # A.SafeRotate(),
        # A.Solarize(),
        # A.ToSepia(),
        # A.Sharpen(),
        # A.InvertImg(),
        # A.ColorJitter(),
        # A.Elastic Transform(),
        # A.ChannelDropout(),
        # A.ChannelShuffle(),
        # A.bboxsafecrop,
        # A.FancyPCA,
        # A.Equalize,
        # A.Emboss,
        # A.HueSaturationValue(),
        # A.ISONoise(),
        # A.Imagecompression(),
        # A.JpegCompression(),
        # A.PiecewiseAffine(),
        # A.Posterize(),
        # A.RGBShift(),
    ], bbox_params=A.BboxParams(format='yolo'))
    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    tot_objs = len(bboxes)
    if tot_objs != 0:
        if tot_objs > 1:
            transformed_bboxes = multi_obj_bb_yolo_conversion(transformed_bboxes, classes)
            save_aug_lab(transformed_bboxes, out_lab_pth, transformed_file_name + ".txt")
        else:
            if len(transformed_bboxes) > 0:  # Check if transformed_bboxes list is not empty
                # transformed_bboxes = [single_obj_bb_yolo_conversion(transformed_bboxes[0]), classes]
                transformed_bboxes = [single_obj_bb_yolo_conversion(transformed_bboxes[0], classes)]    # classes 괄호 안 감싼 오류 수정
                save_aug_lab(transformed_bboxes, out_lab_pth, transformed_file_name + ".txt")
        save_aug_image(transformed_image, out_img_pth, transformed_file_name + ".png")
        draw_yolo(transformed_image, transformed_bboxes)      # 시간 절약 차원에서 일단 주석 처리
    else:
        print("label file is empty")      # 시간 절약 차원에서 일단 주석 처리
        # pass    # 추가
