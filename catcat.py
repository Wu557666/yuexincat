import sys
import os
import cv2
import numpy as np
import random
import pygame
from PIL import Image, ImageSequence
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QDialog,
    QMenu, QAction, QListWidget, QListWidgetItem, QPushButton, QFrame
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QUrl
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPainter, QBrush, QPen
import PyQt5

# --- 配置区域 ---
GIF_ASSETS = [
    {"file": "sanduoniecat.gif", "name": "桑多涅猫"},
    {"file": "cat.gif", "name": "月薪-跳"},
    {"file": "cat_pixel_animation.gif", "name": "月薪猫"},
    {"file": "laycat.gif", "name": "月薪-躺"},
]

MUSIC_ASSETS = [
    {"file": "", "name": "🈲 关闭音乐"},
    {"file": "music.mp3", "name": "我真的特别爱你"},
]

CAT_RENDER_WIDTH = 200
AUTO_SWITCH_INTERVAL = 30000

# 缩放配置
SCALE_STEP = 0.1       # 每次缩放的比例
SCALE_MIN = 0.3        # 最小缩放倍数
SCALE_MAX = 10.0        # 最大缩放倍数

# 颜色阈值配置 (用于去除白色背景)
WHITE_DISTANCE_THRESHOLD = 24
WHITE_SPREAD_THRESHOLD = 14

# 平滑配置
BLUR_KERNEL_SIZE = 5
BLUR_SIGMA = 0

CAT_QUOTES = [
    "主人，今天也要开心哦！🐱", "摸摸头，烦恼全消～", "你是世界上最棒的主人！",
    "记得多喝水，注意休息呀。", "喵～想你了！", "工作辛苦啦，歇一会儿吧。",
    "你的笑容是我最大的动力。", "今天天气真好，适合发呆。", "抱抱你，充电完毕！⚡",
    "不要熬夜，早点睡觉觉。", "你是我的小太阳 ☀️", "无论发生什么，我都陪着你。",
    "加油！你可以的！", "今天的你也很可爱呢。", "吃点好吃的犒劳自己吧。",
    "深呼吸，放松一下～", "我爱你，主人！❤️", "喵喵喵？（翻译：求关注）",
    "生活明朗，万物可爱。", "愿你每天都被温柔以待。", "累了就看看我，我在呢。",
    "你是独一无二的宝藏。", "保持微笑，好运连连。", "今天也是元气满满的一天！",
    "别忘了给自己一个拥抱。", "你的努力我都看在眼里。", "世界很大，幸福很小，比如遇见你。",
    "喵呜～给你变个魔术：变出好心情！", "不管多忙，都要照顾好自己。", "你是我的超级英雄！🦸‍♂️",
    "愿所有的美好都如期而至。", "今天也要做一个快乐的小猫奴。", "心情不好吗？让我蹭蹭你。",
    "你是最棒的，不接受反驳！", "记得按时吃饭哦。", "阳光正好，微风不燥。",
    "有你在我身边，真好。", "愿你眼里有光，心中有爱。", "每天都要进步一点点。",
    "小猫祝你今天好运爆棚！🍀", "别皱眉，会有皱纹的～", "你是我的唯一。",
    "今天也要闪闪发光✨", "无论何时，我都在你身后。", "快乐其实很简单，比如此刻。",
    "愿你被这个世界温柔拥抱。", "小猫牌安慰剂，请查收。💊", "你是人间值得。",
    "今天也要爱自己多一点。", "喵～（翻译：我永远爱你）","吴伟权微信：windows557"
]


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def build_foreground_mask(frame_rgb: np.ndarray) -> np.ndarray:
    """构建前景遮罩，去除白色背景并平滑边缘"""
    white_distance = 255 - frame_rgb.max(axis=2)
    channel_spread = frame_rgb.max(axis=2) - frame_rgb.min(axis=2)
    near_white = (
        (white_distance < WHITE_DISTANCE_THRESHOLD) &
        (channel_spread < WHITE_SPREAD_THRESHOLD)
    ).astype(np.uint8)

    h, w = near_white.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    background = near_white.copy() * 255

    cv2.floodFill(background, flood_mask, (0, 0), 128)
    cv2.floodFill(background, flood_mask, (w - 1, 0), 128)
    cv2.floodFill(background, flood_mask, (0, h - 1), 128)
    cv2.floodFill(background, flood_mask, (w - 1, h - 1), 128)

    foreground = np.where(background == 128, 0, 255).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)

    if BLUR_KERNEL_SIZE > 1:
        foreground_float = foreground.astype(np.float32)
        foreground_blurred = cv2.GaussianBlur(foreground_float, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), BLUR_SIGMA)
        foreground = np.clip(foreground_blurred, 0, 255).astype(np.uint8)
    else:
        foreground = cv2.medianBlur(foreground, 3)

    return foreground


def is_transparent_background(rgba_frame: np.ndarray) -> bool:
    h, w = rgba_frame.shape[:2]
    corners = [
        rgba_frame[0, 0, 3],
        rgba_frame[0, w - 1, 3],
        rgba_frame[h - 1, 0, 3],
        rgba_frame[h - 1, w - 1, 3]
    ]
    return all(alpha < 100 for alpha in corners)


def content_bbox(alpha: np.ndarray) -> tuple:
    _, thresh = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(thresh)
    if points is None:
        return 0, 0, alpha.shape[1], alpha.shape[0]
    x, y, w, h = cv2.boundingRect(points)
    pad = 5
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, alpha.shape[1])
    y1 = min(y + h + pad, alpha.shape[0])
    return x0, y0, x1, y1


def union_bbox(boxes: list) -> tuple:
    if not boxes:
        return 0, 0, 10, 10
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return x0, y0, x1, y1


def target_size(src_w: int, src_h: int, render_width: int) -> tuple:
    if src_w == 0:
        return render_width, 10
    scale = render_width / src_w
    height = max(2, round(src_h * scale))
    return render_width, height


def resize_rgba(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)


def process_single_asset(file_path: str, render_width: int):
    if not os.path.exists(file_path):
        print(f"警告: 找不到文件 {file_path}")
        return [], []

    image = Image.open(file_path)
    rgba_frames = []
    boxes = []
    durations = []

    first_frame = next(ImageSequence.Iterator(image))
    first_rgba = np.array(first_frame.convert("RGBA"), dtype=np.uint8)
    has_transparent_bg = is_transparent_background(first_rgba)

    image.seek(0)
    print(f"文件 {os.path.basename(file_path)} 背景类型判定: {'透明背景' if has_transparent_bg else '实心背景'}")

    for frame in ImageSequence.Iterator(image):
        rgba = np.array(frame.convert("RGBA"), dtype=np.uint8)
        if has_transparent_bg:
            alpha_mask = rgba[:, :, 3]
        else:
            alpha_mask = build_foreground_mask(rgba[..., :3])

        rgba[:, :, 3] = alpha_mask
        rgba_frames.append(rgba)
        boxes.append(content_bbox(alpha_mask))

        dur = frame.info.get('duration', 100)
        if dur == 0:
            dur = 100
        durations.append(max(dur, 20))

    if not rgba_frames:
        return [], []

    x0, y0, x1, y1 = union_bbox(boxes)
    crop_w = x1 - x0
    crop_h = y1 - y0
    width, height = target_size(crop_w, crop_h, render_width)

    pixmaps = []
    for i, frame in enumerate(rgba_frames):
        cropped = frame[y0:y1, x0:x1]
        scaled = resize_rgba(cropped, width, height)
        height_img, width_img, channels = scaled.shape
        bytes_per_line = channels * width_img
        qimg = QImage(scaled.data, width_img, height_img, bytes_per_line, QImage.Format_RGBA8888).copy()
        pixmap = QPixmap.fromImage(qimg)
        pixmaps.append(pixmap)

    return pixmaps, durations


def load_all_assets_separated(asset_config_list: list, render_width: int):
    assets_data = []
    for config in asset_config_list:
        file_path = resource_path(config['file'])
        display_name = config.get('name', config['file'])
        frames, durs = process_single_asset(file_path, render_width)
        if frames:
            assets_data.append({'name': display_name, 'frames': frames, 'durations': durs})
    if not assets_data:
        raise RuntimeError("没有成功加载任何图片帧。")
    return assets_data


class BubbleDialog(QDialog):
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.text = text
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.font = QFont("Microsoft YaHei", 10)
        self.padding = 15
        self._calculate_size()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.close)
        self.timer.start(3000)

    def _calculate_size(self):
        fm = self.fontMetrics()
        rect = fm.boundingRect(self.text)
        self.resize(rect.width() + self.padding * 2, rect.height() + self.padding * 2)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QColor(255, 255, 255, 240)))
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawRoundedRect(self.rect(), 10, 10)
        painter.setFont(self.font)
        painter.setPen(QColor(50, 50, 50))
        fm = painter.fontMetrics()
        rect = fm.boundingRect(self.text)
        x = (self.width() - rect.width()) // 2
        y = (self.height() - rect.height()) // 2 + fm.ascent() // 2
        painter.drawText(x, y, self.text)


class AssetSelectorDialog(QDialog):
    asset_selected = pyqtSignal(int)

    def __init__(self, assets_data, current_index, parent=None):
        super().__init__(parent)
        self.assets_data = assets_data
        self.setWindowTitle("选择小猫形象")
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        main_layout = QVBoxLayout(self)

        self.frame = QFrame(self)
        self.frame.setStyleSheet("QFrame { background-color: white; border-radius: 15px; border: 2px solid #ddd; }")
        frame_layout = QVBoxLayout(self.frame)

        title_label = QLabel("🎨 选择形象")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(title_label)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget { background-color: transparent; border: none; font-size: 12px; }
            QListWidget::item { padding: 8px; border-bottom: 1px solid #eee; }
            QListWidget::item:selected { background-color: #e0f7fa; color: #006064; border-radius: 5px; }
        """)
        for i, asset in enumerate(assets_data):
            item = QListWidgetItem(asset['name'])
            if i == current_index:
                item.setSelected(True)
            self.list_widget.addItem(item)

        self.list_widget.itemClicked.connect(
            lambda item: self.asset_selected.emit(self.list_widget.row(item)) or self.close())
        frame_layout.addWidget(self.list_widget)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("QPushButton { background-color: #ff6b6b; color: white; border: none; padding: 8px; border-radius: 5px; }")
        frame_layout.addWidget(close_btn)

        main_layout.addWidget(self.frame)
        self.resize(300, 400)


class ScaleDialog(QDialog):
    """缩放选择对话框"""
    scale_changed = pyqtSignal(float)

    def __init__(self, current_scale, parent=None):
        super().__init__(parent)
        self.setWindowTitle("缩放小猫")
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        self.current_scale = current_scale

        main_layout = QVBoxLayout(self)

        self.frame = QFrame(self)
        self.frame.setStyleSheet("QFrame { background-color: white; border-radius: 15px; border: 2px solid #ddd; }")
        frame_layout = QVBoxLayout(self.frame)

        title_label = QLabel("🔍 调整大小")
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(title_label)

        self.scale_label = QLabel(f"当前缩放: {int(current_scale * 100)}%")
        self.scale_label.setFont(QFont("Microsoft YaHei", 10))
        self.scale_label.setAlignment(Qt.AlignCenter)
        frame_layout.addWidget(self.scale_label)

        # 预设缩放选项
        presets = [
            ("🔬 30%（最小）", 0.3),
            ("🐱 100%（默认）", 1.0),
            ("🐈 400%（大）", 4.0),
            ("🦁 600%（超大）", 6.0),
            ("👑 1000%（巨无霸）", 10.0),
        ]

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget { background-color: transparent; border: none; font-size: 12px; }
            QListWidget::item { padding: 10px; border-bottom: 1px solid #eee; }
            QListWidget::item:selected { background-color: #e0f7fa; color: #006064; border-radius: 5px; }
        """)
        for name, scale_val in presets:
            item = QListWidgetItem(name)
            if abs(scale_val - current_scale) < 0.01:
                item.setSelected(True)
            self.list_widget.addItem(item)

        self.list_widget.itemClicked.connect(self._on_item_clicked)
        frame_layout.addWidget(self.list_widget)

        # 自定义缩放按钮
        btn_layout = QHBoxLayout()

        zoom_in_btn = QPushButton("➕ 放大")
        zoom_in_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border: none; padding: 8px; border-radius: 5px; }")
        zoom_in_btn.clicked.connect(self._zoom_in)
        btn_layout.addWidget(zoom_in_btn)

        zoom_out_btn = QPushButton("➖ 缩小")
        zoom_out_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; border: none; padding: 8px; border-radius: 5px; }")
        zoom_out_btn.clicked.connect(self._zoom_out)
        btn_layout.addWidget(zoom_out_btn)

        reset_btn = QPushButton("🔄 重置")
        reset_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; border: none; padding: 8px; border-radius: 5px; }")
        reset_btn.clicked.connect(lambda: self._apply_scale(1.0))
        btn_layout.addWidget(reset_btn)

        frame_layout.addLayout(btn_layout)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("QPushButton { background-color: #ff6b6b; color: white; border: none; padding: 8px; border-radius: 5px; }")
        frame_layout.addWidget(close_btn)

        main_layout.addWidget(self.frame)
        self.resize(300, 480)

    def _on_item_clicked(self, item):
        row = self.list_widget.row(item)
        presets = [0.3, 1.0, 4.0, 6.0, 10.0]
        if 0 <= row < len(presets):
            self._apply_scale(presets[row])

    def _zoom_in(self):
        new_scale = min(self.current_scale + SCALE_STEP, SCALE_MAX)
        self._apply_scale(new_scale)

    def _zoom_out(self):
        new_scale = max(self.current_scale - SCALE_STEP, SCALE_MIN)
        self._apply_scale(new_scale)

    def _apply_scale(self, scale_val):
        self.current_scale = scale_val
        self.scale_label.setText(f"当前缩放: {int(scale_val * 100)}%")
        self.scale_changed.emit(scale_val)


class CatPetWidget(QWidget):
    def __init__(self, assets_data: list):
        super().__init__()
        self.assets_data = assets_data
        self.current_asset_idx = 0
        self.frames = self.assets_data[0]['frames']
        self.durations = self.assets_data[0]['durations']
        self.current_frame_idx = 0
        self.draggable = False
        self.offset = None
        self.bubble = None
        self.auto_switch_timer = QTimer(self)
        self.auto_switch_timer.timeout.connect(self.auto_switch_asset)
        self.is_auto_switching = False

        # 缩放相关
        self.scale_factor = 1.0

        # 音乐播放器初始化
        pygame.mixer.init()

        self.init_ui()
        self.start_animation()

    def init_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.label = QLabel(self)
        self.label.setStyleSheet("background: transparent;")
        if self.frames:
            self._display_frame(self.frames[0])

    def _display_frame(self, pixmap):
        """根据当前缩放比例显示帧"""
        if self.scale_factor != 1.0:
            new_w = int(pixmap.width() * self.scale_factor)
            new_h = int(pixmap.height() * self.scale_factor)
            scaled = pixmap.scaled(new_w, new_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(scaled)
            self.label.resize(scaled.size())
            self.resize(scaled.size())
        else:
            self.label.setPixmap(pixmap)
            self.label.resize(pixmap.size())
            self.resize(pixmap.size())

    def start_animation(self):
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.next_frame)
        self._schedule_next()

    def _schedule_next(self):
        if self.durations:
            self.anim_timer.start(self.durations[self.current_frame_idx])

    def next_frame(self):
        self.anim_timer.stop()
        if self.frames:
            self._display_frame(self.frames[self.current_frame_idx])
            self.current_frame_idx = (self.current_frame_idx + 1) % len(self.frames)
            self._schedule_next()

    def switch_asset(self, index: int):
        if 0 <= index < len(self.assets_data):
            self.current_asset_idx = index
            asset = self.assets_data[index]
            self.frames = asset['frames']
            self.durations = asset['durations']
            self.current_frame_idx = 0
            if self.frames:
                self._display_frame(self.frames[0])
            self.anim_timer.stop()
            self._schedule_next()
            self.show_bubble(f"切换到了: {asset['name']}")

    def auto_switch_asset(self):
        if len(self.assets_data) > 1:
            new_idx = self.current_asset_idx
            while new_idx == self.current_asset_idx:
                new_idx = random.randint(0, len(self.assets_data) - 1)
            self.switch_asset(new_idx)

    def toggle_auto_switch(self):
        self.is_auto_switching = not self.is_auto_switching
        if self.is_auto_switching:
            self.auto_switch_timer.start(AUTO_SWITCH_INTERVAL)
            self.show_bubble("🎲 自动漫游开启")
        else:
            self.auto_switch_timer.stop()
            self.show_bubble("🛑 自动漫游关闭")

    def apply_scale(self, scale_val: float):
        """应用缩放"""
        self.scale_factor = scale_val
        if self.frames:
            self._display_frame(self.frames[self.current_frame_idx])
        self.show_bubble(f"🔍 缩放: {int(scale_val * 100)}%")

    def play_music(self, index: int):
        if 0 <= index < len(MUSIC_ASSETS):
            self.current_music_idx = index
            music = MUSIC_ASSETS[index]
            if music['file']:
                file_path = resource_path(music['file'])
                if not os.path.exists(file_path):
                    self.show_bubble("❌ 找不到音乐文件！")
                    return
                pygame.mixer.music.load(file_path)
                pygame.mixer.music.set_volume(0.5)
                pygame.mixer.music.play(-1)
                self.show_bubble(f"🎵 正在播放: {music['name']}")
            else:
                pygame.mixer.music.stop()
                self.show_bubble("🔇 背景音乐已关闭")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.draggable = True
            self.offset = event.pos()
        elif event.button() == Qt.RightButton:
            self.show_context_menu(event.globalPos())

    def mouseMoveEvent(self, event):
        if self.draggable and self.offset is not None:
            self.move(self.pos() + event.pos() - self.offset)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.draggable = False
            self.offset = None

    def show_context_menu(self, pos):
        menu = QMenu(self)
        menu.addAction("🗣️ 互动说话", self.interact)
        menu.addAction("🎨 打开形象选择器", self.open_asset_selector)
        menu.addAction("🔍 调整大小", self.open_scale_dialog)

        # 音乐选择子菜单
        music_menu = menu.addMenu("🎵 背景音乐")
        for i, music in enumerate(MUSIC_ASSETS):
            action = music_menu.addAction(music['name'])
            action.triggered.connect(lambda checked, idx=i: self.play_music(idx))

        auto_text = "✅ 关闭自动漫游" if self.is_auto_switching else "❌ 开启自动漫游"
        menu.addAction(auto_text, self.toggle_auto_switch)
        menu.addSeparator()
        menu.addAction("❌ 退出程序", self.quit_app)
        menu.exec_(pos)

    def open_asset_selector(self):
        dialog = AssetSelectorDialog(self.assets_data, self.current_asset_idx, self)
        dialog.asset_selected.connect(self.switch_asset)
        dialog.exec_()

    def open_scale_dialog(self):
        dialog = ScaleDialog(self.scale_factor, self)
        dialog.scale_changed.connect(self.apply_scale)
        dialog.exec_()

    def show_bubble(self, text):
        if self.bubble:
            self.bubble.close()
        self.bubble = BubbleDialog(text, self)
        cat_w, cat_h = self.width(), self.height()
        bubble_w, bubble_h = self.bubble.width(), self.bubble.height()
        bubble_x = self.x() + (cat_w - bubble_w) // 2
        bubble_y = self.y() - bubble_h - 10
        screen_geo = QApplication.primaryScreen().geometry()
        if bubble_y < screen_geo.y():
            bubble_y = self.y() + cat_h + 10
        self.bubble.move(bubble_x, bubble_y)
        self.bubble.show()

    def interact(self):
        self.show_bubble(random.choice(CAT_QUOTES))

    def quit_app(self):
        pygame.mixer.quit()
        QApplication.quit()


if __name__ == "__main__":
    qt_dir = os.path.dirname(PyQt5.__file__)
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.join(qt_dir, 'Qt5', 'plugins', 'platforms')
    app = QApplication(sys.argv)
    try:
        assets_data = load_all_assets_separated(GIF_ASSETS, CAT_RENDER_WIDTH)
        pet = CatPetWidget(assets_data)
        pet.show()
        screen_geo = app.primaryScreen().geometry()
        # 将小猫移动到屏幕右下角
        pet.move(screen_geo.width() - pet.width() - 50,
                 screen_geo.height() - pet.height() - 50)
        sys.exit(app.exec_())
    except Exception as e:
        print(f"程序启动错误: {e}")