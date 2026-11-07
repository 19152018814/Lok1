import cv2
import numpy as np


# 图像读取
def read_image(path, grayscale=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR)
    return img


# 图像配准
def align_images(im1, im2, save_path=None):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches.sort(key=lambda x: x.distance)
    matches = matches[:30]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    aligned = cv2.warpPerspective(im1, M, (im2.shape[1], im2.shape[0]))

    if save_path:
        cv2.imwrite(save_path, aligned)  # 保存配准后的图像

    return aligned


# 图像增强
def enhance_image(img, clahe_clip=2.0, blur_ksize=5, sharpen=True, save_path=None):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)

    if blur_ksize > 0:
        enhanced = cv2.GaussianBlur(enhanced, (blur_ksize, blur_ksize), 0)

    if sharpen:
        kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        enhanced = cv2.filter2D(enhanced, -1, kernel_sharpen)

    if save_path:
        cv2.imwrite(save_path, enhanced)  # 保存增强后的图像

    return enhanced


# 差异计算与增强
def calculate_diff(img_liquid, img_bg, save_path=None):
    diff = cv2.subtract(img_bg,img_liquid)

    if save_path:
        cv2.imwrite(save_path, diff)  # 保存差异图像

    return diff


# 主函数
def process_images(path_liquid, path_bg, config, paths):
    # 读取图片
    img_liquid = read_image(path_liquid, grayscale=config['grayscale'])
    img_bg = read_image(path_bg, grayscale=config['grayscale'])

    # 图像配准
    if config['align']:
        img_liquid = align_images(img_liquid, img_bg, save_path=paths['align'])

    # 图像增强
    if config['enhance']:
        img_liquid = enhance_image(img_liquid,
                                   clahe_clip=config['clahe_clip'],
                                   blur_ksize=config['blur_ksize'],
                                   sharpen=config['sharpen'],
                                   save_path=paths['enhance_liquid'])
        img_bg = enhance_image(img_bg,
                               clahe_clip=config['clahe_clip'],
                               blur_ksize=config['blur_ksize'],
                               sharpen=config['sharpen'],
                               save_path=paths['enhance_bg'])

    # 差异图像计算与增强
    diff_img = calculate_diff(img_liquid, img_bg,
                              save_path=paths['diff'])

    # 数据增强
    final_img = diff_img
    cv2.imwrite(paths['final'], final_img)  # 保存最终结果图像

    return final_img


# 使用示例
if __name__ == "__main__":
    config = {
        'grayscale': False,
        'align': False,
        'enhance': True,
        'clahe_clip': 1.0,
        'blur_ksize': 5,
        'sharpen': True,
    }

    # 自定义每个步骤的保存路径
    paths = {
        'align': 'test/aligned_image.jpg',  # 配准后的图像路径
        'enhance_liquid': 'test/enhanced_liquid_image.jpg',  # 增强后的液滴图像路径
        'enhance_bg': 'test/enhanced_bg_image.jpg',  # 增强后的背景图像路径
        'diff': 'test/difference_image.jpg',  # 差异图像路径
        'final': 'test/final_result.jpg'  # 最终处理结果图像路径
    }

    result = process_images('test/1.jpg', 'test/bg1.jpg', config, paths)
