# 在项目根目录下新建一个 check_data.py，运行一次就删掉
import sys
sys.path.insert(0, ".")
from src.utils.data import load_product_data

product_df, raw_df = load_product_data()
print("产品数量:", len(product_df))
print("评论数量:", len(raw_df))
print(product_df['review_count'].describe())