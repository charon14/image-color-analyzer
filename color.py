import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter

IMAGE_DIR = "/Users/marat/Pictures/alex/"
RESIZE = 400

# ---------- 加载 ----------
def load_images(folder):
    imgs, names = [], []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            img = cv2.imread(os.path.join(folder, f))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            s = RESIZE / max(h, w)
            img = cv2.resize(img, (int(w*s), int(h*s)))
            imgs.append(img); names.append(f)
    return imgs, names

# ---------- 1. Lab 空间 a/b 散点 ----------
def plot_lab_ab_distribution(images):
    all_lab = np.vstack([
        cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3)
        for img in images
    ])
    # 采样避免过密
    idx = np.random.choice(len(all_lab), size=min(50000, len(all_lab)), replace=False)
    sample = all_lab[idx]
    L, a, b = sample[:,0], sample[:,1].astype(int)-128, sample[:,2].astype(int)-128
    
    fig, ax = plt.subplots(figsize=(7, 7))
    # 用对应的 RGB 颜色着色
    rgb_colors = cv2.cvtColor(sample.reshape(1,-1,3), cv2.COLOR_LAB2RGB).reshape(-1,3)/255
    ax.scatter(a, b, c=rgb_colors, s=2, alpha=0.4)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel('a* (green ← → red)')
    ax.set_ylabel('b* (blue ← → yellow)')
    ax.set_title('Lab a*b* Distribution — 色彩在感知空间中的分布')
    ax.set_xlim(-128, 128); ax.set_ylim(-128, 128)
    # 计算主轴方向（PCA）
    pts = np.column_stack([a, b])
    pca = PCA(n_components=2).fit(pts)
    vec = pca.components_[0] * 80
    ax.arrow(0, 0, vec[0], vec[1], color='black', width=1, head_width=5)
    ax.arrow(0, 0, -vec[0], -vec[1], color='black', width=1, head_width=5)
    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    ax.set_title(f'Lab a*b* 分布 — 主轴角度: {angle:.1f}° (方差占比 {pca.explained_variance_ratio_[0]:.1%})')
    plt.tight_layout(); plt.show()
    return angle, pca.explained_variance_ratio_[0]

# ---------- 2. 主色色相和谐分析 ----------
def analyze_color_harmony(images, k=6):
    all_hues = []
    for img in images:
        pixels = img.reshape(-1, 3)
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(pixels)
        centers = km.cluster_centers_.astype(np.uint8).reshape(1, -1, 3)
        hsv = cv2.cvtColor(centers, cv2.COLOR_RGB2HSV).reshape(-1, 3)
        # 只考虑饱和度足够的主色
        mask = hsv[:, 1] > 40
        all_hues.extend(hsv[mask, 0].tolist())
    
    hues = np.array(all_hues) * 2  # OpenCV H 范围 0-179，转为 0-360
    
    # 色相对差分布
    diffs = []
    for i in range(len(hues)):
        for j in range(i+1, len(hues)):
            d = abs(hues[i] - hues[j])
            d = min(d, 360-d)
            diffs.append(d)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(hues, bins=36, color='purple')
    axes[0].set_title('主色色相分布 (0-360°)')
    axes[0].set_xlabel('Hue (度)')
    
    axes[1].hist(diffs, bins=36, color='teal')
    for x, label in [(30, '类比'), (120, '三角'), (180, '互补')]:
        axes[1].axvline(x, color='red', ls='--', alpha=0.5)
        axes[1].text(x, axes[1].get_ylim()[1]*0.9, label, ha='center')
    axes[1].set_title('主色之间的色相角度差 — 判断和谐类型')
    axes[1].set_xlabel('角度差 (度)')
    plt.tight_layout(); plt.show()

# ---------- 3. Hue-Saturation 2D 热图 ----------
def plot_hue_saturation_heatmap(images):
    all_hsv = np.vstack([
        cv2.cvtColor(img, cv2.COLOR_RGB2HSV).reshape(-1, 3)
        for img in images
    ])
    h, s = all_hsv[:, 0], all_hsv[:, 1]
    heatmap, xedges, yedges = np.histogram2d(h, s, bins=[90, 50], range=[[0,180],[0,255]])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(heatmap.T, origin='lower', aspect='auto',
                   extent=[0, 360, 0, 255], cmap='hot')
    ax.set_xlabel('Hue (度)')
    ax.set_ylabel('Saturation')
    ax.set_title('色相-饱和度联合分布 — 揭示"哪种颜色用得浓，哪种用得淡"')
    plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.show()

# ---------- 4. 明度分层的色彩倾向 ----------
def analyze_luminance_layers(images):
    all_lab = np.vstack([
        cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3)
        for img in images
    ])
    L = all_lab[:, 0]
    shadow  = all_lab[L < 80]
    mid     = all_lab[(L >= 80) & (L < 170)]
    highlight = all_lab[L >= 170]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, data, title in [(axes[0], shadow, '阴影 (L<80)'),
                             (axes[1], mid, '中间调 (80-170)'),
                             (axes[2], highlight, '高光 (L>170)')]:
        if len(data) == 0: continue
        idx = np.random.choice(len(data), size=min(10000, len(data)), replace=False)
        sample = data[idx]
        a = sample[:, 1].astype(int) - 128
        b = sample[:, 2].astype(int) - 128
        rgb = cv2.cvtColor(sample.reshape(1,-1,3).astype(np.uint8), cv2.COLOR_LAB2RGB).reshape(-1,3)/255
        ax.scatter(a, b, c=rgb, s=3, alpha=0.5)
        ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
        ax.set_xlim(-60, 60); ax.set_ylim(-60, 60)
        mean_a, mean_b = sample[:,1].mean()-128, sample[:,2].mean()-128
        ax.plot(mean_a, mean_b, 'k+', markersize=20, markeredgewidth=3)
        ax.set_title(f'{title}\n平均偏移 a={mean_a:.1f}, b={mean_b:.1f}')
        ax.set_xlabel('a*'); ax.set_ylabel('b*')
    plt.suptitle('明度分层色偏分析 — 看阴影/中间调/高光各自的色彩倾向')
    plt.tight_layout(); plt.show()

# ---------- 5. 局部对比度 ----------
def analyze_contrast(images):
    spatial_contrasts, color_contrasts = [], []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        spatial_contrasts.append(lap.var())
        
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(float)
        dx = np.diff(lab, axis=1)
        dy = np.diff(lab, axis=0)
        delta_e = np.sqrt((dx**2).sum(-1)).mean() + np.sqrt((dy**2).sum(-1)).mean()
        color_contrasts.append(delta_e/2)
    
    print(f"\n=== 对比度分析 ===")
    print(f"空间对比度 (Laplacian var): 均值 {np.mean(spatial_contrasts):.0f}, 标准差 {np.std(spatial_contrasts):.0f}")
    print(f"色彩对比度 (avg ΔE): 均值 {np.mean(color_contrasts):.2f}")
    print("  → 作为参考: 普通快照 ΔE 约 3-5, 高对比摄影 ΔE 通常 >8")

# ---------- 6. 九宫格色彩分区 ----------
def analyze_spatial_color(images):
    grid_colors = np.zeros((3, 3, 3))
    for img in images:
        h, w = img.shape[:2]
        for i in range(3):
            for j in range(3):
                tile = img[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
                grid_colors[i, j] += tile.reshape(-1, 3).mean(axis=0)
    grid_colors /= len(images)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(3):
        for j in range(3):
            ax.add_patch(plt.Rectangle((j, 2-i), 1, 1, color=grid_colors[i,j]/255))
            rgb = grid_colors[i,j].astype(int)
            ax.text(j+0.5, 2-i+0.5, f'{rgb[0]},{rgb[1]},{rgb[2]}',
                    ha='center', va='center', color='white', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    ax.set_xlim(0, 3); ax.set_ylim(0, 3); ax.axis('off')
    ax.set_title('构图区域平均色 (九宫格) — 看色彩的空间倾向')
    plt.tight_layout(); plt.show()

# ---------- 7. 每张图的色彩指纹 + 聚类 ----------
def image_fingerprints(images, names):
    features = []
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h_hist = cv2.calcHist([hsv],[0],None,[18],[0,180]).flatten()
        s_hist = cv2.calcHist([hsv],[1],None,[8],[0,256]).flatten()
        v_hist = cv2.calcHist([hsv],[2],None,[8],[0,256]).flatten()
        feat = np.concatenate([h_hist, s_hist, v_hist])
        feat = feat / feat.sum()
        features.append(feat)
    features = np.array(features)
    
    pca = PCA(n_components=2).fit_transform(features)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pca[:,0], pca[:,1], s=100, alpha=0.7)
    for i, name in enumerate(names):
        ax.annotate(name[:12], (pca[i,0], pca[i,1]), fontsize=8)
    ax.set_title('每张图的色彩指纹 (PCA 降维) — 看组内是否有色彩子风格')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    plt.tight_layout(); plt.show()

# ---------- 8. 情绪坐标 ----------
def mood_coordinates(images, names):
    coords = []
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        sat = hsv[:,:,1].mean()
        val = hsv[:,:,2].mean()
        warmth = lab[:,:,2].mean() - 128  # b 通道,正值偏暖
        coords.append([sat, val, warmth])
    coords = np.array(coords)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(coords[:,2], coords[:,0], c=coords[:,1],
                    s=150, cmap='viridis', alpha=0.8)
    ax.set_xlabel('暖冷倾向 (b* 均值, 正值偏暖)')
    ax.set_ylabel('饱和度均值')
    ax.axvline(0, color='gray', ls='--')
    plt.colorbar(sc, label='明度均值')
    ax.set_title('作品情绪坐标 — 每张图在"暖冷×饱和度×明度"空间的位置')
    plt.tight_layout(); plt.show()

# ---------- 主流程 ----------
if __name__ == "__main__":
    images, names = load_images(IMAGE_DIR)
    print(f"加载 {len(images)} 张图片\n")
    
    plot_lab_ab_distribution(images)
    analyze_color_harmony(images)
    plot_hue_saturation_heatmap(images)
    analyze_luminance_layers(images)
    analyze_contrast(images)
    analyze_spatial_color(images)
    image_fingerprints(images, names)
    mood_coordinates(images, names)