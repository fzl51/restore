import base64
import os
import re
import time
import uuid
from datetime import datetime
import cv2
import torch
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import get_device
from basicsr.utils.registry import ARCH_REGISTRY
from facelib.utils.face_restoration_helper import FaceRestoreHelper

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}
rootPath = os.path.dirname(os.path.abspath(__file__))
tempPath = os.path.join(rootPath, "temp")

delEndTime = 0


def clean_directory(params={}):
    global delEndTime
    now_time = int(time.time())
    if (delEndTime + 30 * 60) > now_time:
        print('时间未到删除')
        return
    delEndTime = now_time
    current_time = int(time.time())  # 当前时间的时间戳
    path = params.get('path', tempPath)
    if not path:
        return

    try:
        # 获取上传目录中的所有文件
        files = os.listdir(path)
    except OSError as e:
        print(f'无法读取上传目录: {e}')
        return

    # 遍历上传目录中的文件
    for file in files:
        # 使用正则表达式匹配文件名中的时间戳部分（假设文件名中包含时间戳）
        # timestamp_match = file.split('-')
        match = re.search(r'-(\d{10})\.', file)
        if match:
            file_timestamp = int(match.group(1))
            time_difference = current_time - file_timestamp
            # 如果时间戳大于30分钟（30 * 60秒），则删除文件
            if time_difference and time_difference > params.get('t', 30 * 60):
                file_path = os.path.join(path, file)
                try:
                    os.remove(file_path)
                    print(f'已删除过期文件: {file}')
                except OSError as e:
                    continue

    return True


device = get_device()
print('使用算力:',device)
# ------------------ set up CodeFormer restorer -------------------
net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                      connect_list=['32', '64', '128', '256']).to(device)
# ckpt_path = 'weights/CodeFormer/codeformer.pth'
ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'],
                               model_dir='weights/CodeFormer', progress=True, file_name=None)
checkpoint = torch.load(ckpt_path)['params_ema']
net.load_state_dict(checkpoint)
net.eval()


def start(img, w=0.7):
    face_helper = FaceRestoreHelper(
        2,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=device)
    # -------------------- start to processing ---------------------
    # clean all the intermediate results to process the next image
    # face_helper.clean_all()
    # image_np = np.array(image)
    #
    # # 检查图像的通道顺序，如果是 RGB 顺序，需要将其转换为 BGR 顺序
    # # OpenCV 使用 BGR 顺序，而 PIL 使用 RGB 顺序
    # if image_np.shape[-1] == 3:  # 如果是 3 通道图像（RGB）
    #     img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    # elif image_np.shape[-1] == 4:  # 如果是 4 通道图像（RGBA）
    #     img = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGRA)
    # else:
    #     img = image_np  # 灰度图像，不需要转换
    face_helper.read_image(img)
    # get face landmarks for each face
    num_det_faces = face_helper.get_face_landmarks_5(
        only_center_face=False, resize=640, eye_dist_threshold=5)
    print(f'\tdetect {num_det_faces} faces')

    # align and warp each face
    face_helper.align_warp_face()

    # face restoration for each cropped face
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        # prepare data
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                output = net(cropped_face_t, w=w, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except Exception as error:
            print(f'\tFailed inference for CodeFormer: {error}')
            restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

        restored_face = restored_face.astype('uint8')
        face_helper.add_restored_face(restored_face, cropped_face)
    face_helper.get_inverse_affine(None)
    # paste each restored face to the input image
    restored_img = face_helper.paste_faces_to_input_image(upsample_img=None, draw_box=False)
    # save_restore_path = os.path.join('temp', f'demo_ok.png')
    # imwrite(restored_img, save_restore_path)
    face_helper.clean_all()
    face_helper = None
    return restored_img;


def inference(img, w=70, quality=100, isBase64=True):
    now = datetime.now()
    isBase64 = False if isBase64 == 'no' else True
    if not str(w).isdigit():
        w = 70
    w = int(w)
    if w < 10 or w > 100:
        w = 70
    w = w / 100
    if not str(quality).isdigit():
        quality = 100
    quality = int(quality)
    if quality < 10 or quality > 100:
        quality = 100
    print(now.strftime("%Y-%m-%d %H:%M:%S"), w, isBase64)
    start_time = time.time()
    output = start(img, w=w)
    print(f"处理耗时: {time.time() - start_time} 秒")
    start_time2 = time.time()
    success, buffer = cv2.imencode('.webp', output, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if isBase64:
        # 将图像编码为 WebP 格式

        # 将编码后的图像转换为 Base64 编码字符串
        webp_base64 = base64.b64encode(buffer).decode('utf-8')
        print(f"base64耗时: {time.time() - start_time2} 秒")
        # 创建 Base64 编码的 WebP 图像的 Data URL
        return f'data:image/webp;base64,{webp_base64}'

    imageName = f"{uuid.uuid4()}-{int(time.time())}.webp"
    if success:
        with open(os.path.join(tempPath, imageName), 'wb') as f:
            f.write(buffer)
    print(f"保存耗时: {time.time() - start_time2} 秒")
    clean_directory({'t': 30 * 60})
    return f"https://image.yy2169.com/restore/{imageName}"


img_cv = cv2.imread(os.path.join(rootPath, 'inputs/whole_imgs/00.jpg'), cv2.IMREAD_COLOR)
start_time = time.time()
output = start(img_cv, w=0.7)
print(f"处理耗时: {time.time() - start_time} 秒")
start_time2 = time.time()
success, buffer = cv2.imencode('.webp', output, [cv2.IMWRITE_WEBP_QUALITY, 99])
if success:
    with open(os.path.join(rootPath, 'ok.webp'), 'wb') as f:
        f.write(buffer)
print(f"保存耗时: {time.time() - start_time2} 秒")
