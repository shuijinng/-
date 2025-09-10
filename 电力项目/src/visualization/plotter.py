import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.dates import MonthLocator, DateFormatter
import logging

# 设置logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import matplotlib.font_manager as fm
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体作为默认字体以支持中文
plt.rcParams['axes.unicode_minus'] = False
# 其他已有代码保持不变...

def plot_predictions(y_true, y_pred, save_path=None):
    """
    绘制真实值与预测值的对比图，x轴刻度为数据中实际存在的“每月第一个日期”。
    """
    plt.figure(figsize=(16, 6))

    # 确保索引是 datetime 类型
    if not isinstance(y_true.index, pd.DatetimeIndex):
        y_true.index = pd.to_datetime(y_true.index)
    if not isinstance(y_pred.index, pd.DatetimeIndex):
        y_pred.index = pd.to_datetime(y_pred.index)

    # 绘制数据
    plt.plot(y_true.index, y_true, label='真实值', marker='o', linewidth=1.5, markersize=4)
    plt.plot(y_pred.index, y_pred, label='预测值', marker='x', linestyle='--', linewidth=1.5, markersize=4)
    plt.title('电力负荷预测结果对比（按实际月份展示）')
    plt.xlabel('时间（月）')
    plt.ylabel('负荷 (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # --- ✅ 核心修改：只显示数据中实际存在的“每月第一个日期” ---
    # 合并 y_true 和 y_pred 的索引，确保覆盖所有时间点
    all_dates = pd.DatetimeIndex(sorted(set(y_true.index) | set(y_pred.index)))

    # 提取“每个月的第一个日期”（基于实际数据）
    first_of_month_dates = all_dates.to_period('M').drop_duplicates().to_timestamp()

    # 转换回 datetime 并确保在原始索引中存在（或找最近存在的）
    tick_locations = []
    for date in first_of_month_dates:
        # 找到最接近的、实际存在于数据中的日期
        available_dates = all_dates[all_dates.month == date.month]
        if len(available_dates) > 0:
            # 使用该月最早出现的日期
            tick_locations.append(available_dates.min())

    # 去重并排序
    tick_locations = sorted(set(tick_locations))

    # 设置 x 轴刻度和标签
    plt.xticks(tick_locations, [d.strftime("%Y-%m") for d in tick_locations], rotation=45)

    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close()
            logger.info(f"✅ 图表已保存至: {save_path}")
        except Exception as e:
            logger.error(f"❌ 保存图表失败: {e}")
    else:
        plt.show()
def plot_time_series(data: pd.Series, title: str = "时间序列图", save_path=None):
    """绘制时间序列图。"""
    plt.figure(figsize=(12, 6))
    data.plot()
    plt.title(title)
    plt.xlabel("时间")
    plt.ylabel("负荷")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close()
            logger.info(f"✅ 时间序列图已保存至: {save_path}")
        except Exception as e:
            logger.error(f"❌ 保存时间序列图失败: {e}")
    else:
        plt.show()