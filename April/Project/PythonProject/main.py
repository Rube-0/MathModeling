import numpy as np
from scipy.interpolate import griddata
from PIL import Image
import math

def cylindrical_anamorphosis(image_path, h=180, r=30, alpha_deg=25, h_prime=20,
                             output_size=(1000, 1000), margin=10):
    """
    生成圆柱反射变形图。
    参数:
        image_path: 输入图像路径
        h: 视点高度 (cm)
        r: 圆柱半径 (cm)
        alpha_deg: 俯角 (度)
        h_prime: 反射像底部离地高度 (cm)
        output_size: 输出图像 (高, 宽)
        margin: 输出物理边界拓展百分比
    """
    # 读取图像，获取像素宽高比
    img = Image.open(image_path).convert('RGB')
    W_img, H_img = img.size
    B = H_img / W_img           # 高/宽 比 (论文中 B = l/w)

    alpha = math.radians(alpha_deg)
    d = h / math.tan(alpha)

    # ---------- 迭代求解 w, l, D ----------
    w = 50.0                     # 初始猜测
    for _ in range(100):
        l = B * w
        # 计算 D
        numerator = (l*(h - h_prime)*math.sin(alpha) +
                     l*(d - r)*math.cos(alpha) +
                     2*d*h_prime - 2*r*h)
        denominator = 2*h + l*math.cos(alpha)
        D = numerator / denominator

        # 相切约束： r / (w/2) = sqrt((d-D)^2 - r^2) / (d - l/2 * sinα)
        rhs = math.sqrt((d - D)**2 - r**2) / (d - (l/2)*math.sin(alpha))
        w_new = 2*r / rhs
        if abs(w - w_new) < 1e-4:
            break
        w = w_new
    else:
        print("Warning: w did not converge")
    l = B * w

    print(f"d = {d:.1f} cm, D = {D:.1f} cm, w = {w:.1f} cm, l = {l:.1f} cm")

    # ---------- 建立输入图像物理坐标网格 ----------
    # 图像中心为 C'，x 向右，y 向上
    x_phys = (np.arange(W_img) - (W_img-1)/2) * (w / W_img)
    y_phys = ((H_img-1)/2 - np.arange(H_img)) * (l / H_img)
    X, Y = np.meshgrid(x_phys, y_phys)  # 形状 (H_img, W_img)

    # ---------- 透视变换 (x,y) -> (x', y') ----------
    denominator = 1 - (Y / h) * math.cos(alpha)
    Y_prime = Y / denominator
    X_prime = X * np.sqrt((h**2 + (d + Y_prime)**2) / (h**2 + d**2 + Y**2))

    # ---------- 镜面反射变换 (x', y'+D) -> (x'', y'') ----------
    # 先计算 ζ
    A = d - D
    xp2 = X_prime**2
    ypd = Y_prime + d
    radicand = r**2 * xp2 + r**2 * ypd**2 - A**2 * xp2
    radicand = np.maximum(radicand, 0)   # 防止数值误差导致负数
    sqrt_term = np.sqrt(radicand)

    cos_zeta = (A * xp2 + ypd * sqrt_term) / (r * (xp2 + ypd**2))
    # 限制在 [-1, 1]
    cos_zeta = np.clip(cos_zeta, -1, 1)
    sin_zeta = (A - r * cos_zeta) * X_prime / (r * ypd + 1e-12)

    # 双角
    cos2z = cos_zeta**2 - sin_zeta**2
    sin2z = 2 * sin_zeta * cos_zeta

    Y_shifted = Y_prime + D
    X_double = 2*r*sin_zeta + X_prime*cos2z + Y_shifted*sin2z
    Y_double = 2*r*cos_zeta - X_prime*sin2z + Y_shifted*cos2z

    # ---------- 输出图像物理范围 ----------
    x_min, x_max = np.min(X_double), np.max(X_double)
    y_min, y_max = np.min(Y_double), np.max(Y_double)
    # 加边距
    dx = (x_max - x_min) * margin / 100
    dy = (y_max - y_min) * margin / 100
    x_min -= dx; x_max += dx
    y_min -= dy; y_max += dy

    out_h, out_w = output_size
    # 输出像素对应的物理坐标（y 向下）
    xi = np.linspace(x_min, x_max, out_w)
    yi = np.linspace(y_max, y_min, out_h)   # 从上到下
    XI, YI = np.meshgrid(xi, yi)

    # ---------- 逆向插值（将正向点插值到规则网格） ----------
    src_points = np.column_stack([X_double.ravel(), Y_double.ravel()])
    colors = np.array(img).reshape(-1, 3) / 255.0
    dst_points = np.column_stack([XI.ravel(), YI.ravel()])

    # 使用 griddata 线性插值
    grid_r = griddata(src_points, colors[:, 0], dst_points, method='linear', fill_value=1.0)
    grid_g = griddata(src_points, colors[:, 1], dst_points, method='linear', fill_value=1.0)
    grid_b = griddata(src_points, colors[:, 2], dst_points, method='linear', fill_value=1.0)

    # 重组图像
    out_img = np.stack([grid_r.reshape(out_h, out_w),
                        grid_g.reshape(out_h, out_w),
                        grid_b.reshape(out_h, out_w)], axis=2)
    out_img = (out_img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out_img)

# 示例调用（使用一张测试图片，或绘制简单几何图案）
if __name__ == "__main__":
    # 创建一个测试图像：白色背景上的黑色文字
    from PIL import ImageDraw, ImageFont
    test_img = Image.new('RGB', (400, 300), 'white')
    draw = ImageDraw.Draw(test_img)
    font = ImageFont.load_default()
    draw.text((100, 100), "Anamorph", fill='black')
    test_img.save("test_input.png")

    result = cylindrical_anamorphosis("test_input.png", h=180, r=30, alpha_deg=25, h_prime=20)
    result.save("anamorphic_output.png")
    print("变形图已保存为 anamorphic_output.png")