import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.collections import PolyCollection
from PIL import Image

# ===================== 全局设置 =====================
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['axes.unicode_minus'] = False

# ===================== 物理参数（与Mathematica完全一致） =====================
h = 180
r = 30
alpha_deg = 25
h_prime = 20
alpha_rad = np.radians(alpha_deg)
d = h / np.tan(alpha_rad)


# ===================== 方程求解 =====================
def equation(c, B):
    l_curr = B * c
    numD = l_curr * (h - h_prime) * np.sin(alpha_rad) + l_curr * (d - r) * np.cos(
        alpha_rad) + 2 * d * h_prime - 2 * r * h
    denD = 2 * h + l_curr * np.cos(alpha_rad)
    Dc = numD / denD
    left = r ** 2 * (-B * c / 2 * np.sin(alpha_rad) + d) ** 2
    right = (c ** 2 / 4) * ((d - Dc) ** 2 - r ** 2)
    return left - right


# ===================== 坐标变换 =====================
def anamorphosis_transform(s, t, D):
    yp = (t / np.sin(alpha_rad)) / (1 - (t / h) * np.cos(alpha_rad))
    xp = s * np.sqrt((h ** 2 + (d + yp) ** 2) / (h ** 2 + d ** 2 + t ** 2))
    z = np.sign(xp)

    sqrt_term = r ** 2 * xp ** 2 + r ** 2 * (yp + d) ** 2 - (d - D) ** 2 * xp ** 2
    sqrt_term = np.clip(sqrt_term, 0, None)

    num_cos = (d - D) * xp ** 2 + (yp + d) * np.sqrt(sqrt_term)
    den_cos = r * (xp ** 2 + (yp + d) ** 2 + 1e-8)
    cos_z = num_cos / den_cos
    cos_z = np.clip(cos_z, -1, 1)
    zeta = z * np.arccos(cos_z)

    s2 = np.sin(2 * zeta)
    c2 = np.cos(2 * zeta)

    x2 = 2 * r * np.sin(zeta) + xp * c2 + (yp + D) * s2
    y2 = -(2 * r * np.cos(zeta) - xp * s2 + (yp + D) * c2)
    return x2, y2


# ===================== 核心修复：纹理渲染 =====================
def render_texture(image_path, grid=400):
    # 1. 加载图片并获取真实宽高比
    with Image.open(image_path).convert('RGB') as img:
        img_width, img_height = img.size
        print(f"原始图片尺寸: {img_width}x{img_height}")
        B = img_height / img_width
        print(f"高宽比 B = {B:.4f}")

        # 按原图比例resize，不再强行拉伸成正方形
        if img_height > img_width:
            new_height = grid
            new_width = int(grid / B)
        else:
            new_width = grid
            new_height = int(grid * B)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img_array = np.array(img) / 255.0
        print(f"调整后图片尺寸: {new_width}x{new_height}")

    # 2. 求解参数
    sol = root_scalar(lambda c: equation(c, B), bracket=(1, h), method='brentq')
    c = sol.root
    w = c - 1
    l = w * B

    numD = l * (h - h_prime) * np.sin(alpha_rad) + l * (d - r) * np.cos(alpha_rad) + 2 * d * h_prime - 2 * r * h
    denD = 2 * h + l * np.cos(alpha_rad)
    D = numD / denD

    print(f"求解参数: w={w:.2f}cm, l={l:.2f}cm, D={D:.2f}cm")

    # 3. 生成与图片比例匹配的网格
    s = np.linspace(-w / 2, w / 2, new_width)
    t = np.linspace(-l / 2, l / 2, new_height)
    S, T = np.meshgrid(s, t)
    X, Y = anamorphosis_transform(S, T, D)

    # 4. 三角剖分
    trig = tri.Triangulation(S.flatten(), T.flatten())
    xf, yf = X.flatten(), Y.flatten()
    tris = trig.triangles

    # 5. 为每个三角形获取正确的颜色（关键修复）
    # 三角形中心在(s,t)坐标系中的位置
    s_c = S.flatten()[tris].mean(axis=1)
    t_c = T.flatten()[tris].mean(axis=1)

    # 映射到图片像素索引（修复索引计算逻辑）
    col = ((s_c - (-w / 2)) / (w) * (new_width - 1)).astype(int)
    row = ((t_c - (-l / 2)) / (l) * (new_height - 1)).astype(int)

    # 限制索引范围
    col = np.clip(col, 0, new_width - 1)
    row = np.clip(row, 0, new_height - 1)

    # 打印索引范围，确认是否正常分布
    print(f"列索引范围: {col.min()} ~ {col.max()}")
    print(f"行索引范围: {row.min()} ~ {row.max()}")

    # 提取颜色
    colors = img_array[row, col]

    # 6. 绘图（直接使用每个三角形的RGB颜色，避免tripcolor默认colormap导致整图发紫）
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    triangles_xy = np.stack(
        [
            np.column_stack((xf[tris[:, 0]], yf[tris[:, 0]])),
            np.column_stack((xf[tris[:, 1]], yf[tris[:, 1]])),
            np.column_stack((xf[tris[:, 2]], yf[tris[:, 2]])),
        ],
        axis=1,
    )
    mesh = PolyCollection(triangles_xy, facecolors=colors, edgecolors='none')
    ax.add_collection(mesh)
    ax.autoscale_view()

    # 绘制圆柱轮廓
    theta = np.linspace(0, 2 * np.pi, 500)
    ax.plot(r * np.cos(theta), r * np.sin(theta), 'w', lw=2)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


# ===================== 运行 =====================
if __name__ == '__main__':
    render_texture("p4.png", grid=500)
