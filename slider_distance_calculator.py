import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

def calculate_slider_distance(background_image_path, slider_image_path, output_path=None, debug=False):
    """
    计算滑块验证码的滑动距离
    
    参数:
        background_image_path: 背景图片路径
        slider_image_path: 滑块图片路径
        output_path: 输出图片路径，用于调试
        debug: 是否显示调试信息和图片，默认为False
    
    返回:
        滑动距离（像素）
    """
    # 读取图片
    background = cv2.imread(background_image_path)
    slider = cv2.imread(slider_image_path, cv2.IMREAD_UNCHANGED)  # 保留透明通道
    
    if background is None or slider is None:
        raise ValueError("无法读取图片，请检查路径是否正确")
    
    # 打印图片尺寸
    if debug:
        print(f"背景图片尺寸: {background.shape}")
        print(f"滑块图片尺寸: {slider.shape}")
    
    # 处理滑块图片 - 创建掩码
    if len(slider.shape) == 3 and slider.shape[2] == 4:
        # 分离通道
        b, g, r, a = cv2.split(slider)
        # 创建RGB图像
        slider_rgb = cv2.merge([b, g, r])
        # 使用Alpha通道作为掩码
        mask = a
    else:
        slider_rgb = slider
        # 如果没有Alpha通道，创建一个基于图像的掩码
        gray = cv2.cvtColor(slider, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # 改进的方法：使用缺口检测
    # 转换为灰度图
    bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    
    # 对滑块应用高斯模糊以减少噪声
    slider_blurred = cv2.GaussianBlur(slider_rgb, (5, 5), 0)
    slider_gray = cv2.cvtColor(slider_blurred, cv2.COLOR_BGR2GRAY)
    
    # 使用更适合的边缘检测参数
    bg_edges = cv2.Canny(bg_gray, 50, 150)
    slider_edges = cv2.Canny(slider_gray, 50, 150)
    
    # 优化结果
    kernel = np.ones((3, 3), np.uint8)
    bg_edges = cv2.dilate(bg_edges, kernel, iterations=1)
    slider_edges = cv2.dilate(slider_edges, kernel, iterations=1)
    
    # 使用掩码来更准确地匹配滑块轮廓
    slider_masked_edges = cv2.bitwise_and(slider_edges, slider_edges, mask=mask)
    
    # 尝试多种匹配方法
    methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
    best_val = -float('inf')
    best_loc = (0, 0)
    best_method = None
    
    for method in methods:
        # 使用轮廓匹配找到滑块应该放置的位置
        result = cv2.matchTemplate(bg_edges, slider_masked_edges, method, mask=mask)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 根据匹配方法选择最佳位置
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            loc = min_loc
            val = -min_val  # 转换为正值以便比较
        else:
            loc = max_loc
            val = max_val
        
        if val > best_val:
            best_val = val
            best_loc = loc
            best_method = method
    
    # 计算滑动距离
    slider_x = best_loc[0]
    
    # 可视化结果仅在debug=True时显示
    if debug:
        print(f"最佳匹配方法: {best_method}")
        print(f"匹配值: {best_val}")
        print(f"滑块位置: {best_loc}")
        print(f"滑动距离: {slider_x}像素")
        
        # 可视化结果
        top_left = best_loc
        bottom_right = (top_left[0] + slider_rgb.shape[1], top_left[1] + slider_rgb.shape[0])
        
        # 在背景图上绘制矩形标识匹配位置
        result_image = background.copy()
        cv2.rectangle(result_image, top_left, bottom_right, (0, 255, 0), 2)
        
        # 显示结果
        plt.figure(figsize=(15, 10))
        
        plt.subplot(231)
        plt.title('Background Image')
        plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
        
        plt.subplot(232)
        plt.title('Slider Image')
        if len(slider.shape) == 3 and slider.shape[2] == 4:
            # 转换为RGB并显示
            slider_display = cv2.cvtColor(slider, cv2.COLOR_BGRA2RGBA)
            plt.imshow(slider_display)
        else:
            plt.imshow(cv2.cvtColor(slider, cv2.COLOR_BGR2RGB))
        
        plt.subplot(233)
        plt.title('Slider Mask')
        plt.imshow(mask, cmap='gray')
        
        plt.subplot(234)
        plt.title('Background Edges')
        plt.imshow(bg_edges, cmap='gray')
        
        plt.subplot(235)
        plt.title('Slider Edges')
        plt.imshow(slider_masked_edges, cmap='gray')
        
        plt.subplot(236)
        plt.title('Match Result')
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            print(f"结果已保存到: {output_path}")
        
        plt.show()
    
    return slider_x

def main():
    """
    主函数，用于测试
    """
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置图片路径
    background_path = os.path.join(current_dir, 'dd0d410f3c7143069cfbdc598dd07ee6.jpg')  # 背景图片
    slider_path = os.path.join(current_dir, '9ad2541b5a8f4e7d9826f65b50fdfcb9.png')  # 滑块图片
    output_path = os.path.join(current_dir, 'result.png')  # 输出结果图片
    
    # 确保图片文件存在
    if not os.path.exists(background_path) or not os.path.exists(slider_path):
        print("未找到指定的图片文件")
        
        # 询问用户输入图片路径
        background_path = input("请输入背景图片路径: ")
        slider_path = input("请输入滑块图片路径: ")
    
    # 计算滑动距离
    try:
        # 设置debug=False关闭调试输出
        distance = calculate_slider_distance(background_path, slider_path, output_path, debug=False)
        print(f"滑动距离: {distance}像素")
        
        # 如果第一种方法失败，尝试备用方法
        if distance <= 0:
            print("尝试备用方法...")
            distance = calculate_distance_alternate(background_path, slider_path, output_path, debug=False)
            print(f"滑动距离: {distance}像素")
    except Exception as e:
        print(f"错误: {e}")

def calculate_distance_alternate(background_image_path, slider_image_path, output_path=None, debug=False):
    """
    使用备用方法计算滑块验证码的滑动距离
    基于轮廓检测和颜色差异
    """
    # 读取图片
    background = cv2.imread(background_image_path)
    slider = cv2.imread(slider_image_path, cv2.IMREAD_UNCHANGED)
    
    # 检查图片是否成功加载
    if background is None or slider is None:
        raise ValueError("无法读取图片，请检查路径是否正确")
    
    # 获取滑块的Alpha通道作为掩码
    if len(slider.shape) == 4:
        mask = slider[:, :, 3]
    elif len(slider.shape) == 3 and slider.shape[2] == 4:
        _, _, _, mask = cv2.split(slider)
    else:
        # 如果没有Alpha通道，根据图像亮度创建掩码
        gray = cv2.cvtColor(slider, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # 创建滑块轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 获取最大轮廓
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        
        # 创建滑块形状掩码
        slider_mask = np.zeros_like(mask)
        cv2.drawContours(slider_mask, [max_contour], 0, 255, -1)
        
        # 使用掩码获取滑块形状
        slider_roi = cv2.bitwise_and(slider[:, :, :3], slider[:, :, :3], mask=slider_mask)
        
        # 在背景图中搜索相似形状
        # 转为灰度图并增强对比度
        bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.equalizeHist(bg_gray)
        
        # 在背景图上应用不同的阈值查找可能的缺口
        best_score = float('inf')
        best_x = 0
        
        # 滑块宽度和高度
        h, w = mask.shape[:2]
        
        # 检查背景图像中的每个可能位置
        for x in range(0, background.shape[1] - w, 5):  # 步长为5加快搜索速度
            for y in range(0, background.shape[0] - h, 5):
                # 提取背景中的区域
                roi = background[y:y+h, x:x+w]
                
                # 检查提取区域大小是否匹配
                if roi.shape[:2] != (h, w):
                    continue
                
                # 计算区域与滑块的差异
                diff = cv2.absdiff(roi, slider[:, :, :3])
                diff_score = np.sum(diff * slider_mask[:, :, np.newaxis] / 255.0)
                
                if diff_score < best_score:
                    best_score = diff_score
                    best_x = x
        
        # 可视化部分仅在debug=True时执行
        if debug:
            # 可视化最佳匹配位置
            result_image = background.copy()
            cv2.rectangle(result_image, (best_x, 0), (best_x + w, h), (0, 255, 0), 2)
            
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.title('Slider Image')
            plt.imshow(cv2.cvtColor(slider[:, :, :3], cv2.COLOR_BGR2RGB))
            
            plt.subplot(122)
            plt.title(f'Match Result (score: {best_score:.2f})')
            plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            
            plt.tight_layout()
            
            if output_path:
                output_alt_path = output_path.replace('.png', '_alt.png')
                plt.savefig(output_alt_path)
                print(f"备用方法结果已保存到: {output_alt_path}")
            
            plt.show()
        
        return best_x
    
    return 0

# 如果直接运行此脚本，则执行main函数
if __name__ == "__main__":
    main() 