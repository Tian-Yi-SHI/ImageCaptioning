"""
对比caption文件和图像文件夹，找出缺失的图像
"""
import os
import pandas as pd

def main():
    # 路径
    caption_path = "./storage/dataset/captions/captions.txt"
    image_dir = "./storage/dataset/archive/Images"
    
    # 读取caption文件
    print("读取caption文件...")
    df = pd.read_csv(caption_path)
    caption_images = set(df['image'].unique())
    print(f"Caption文件中的图像数: {len(caption_images)}")
    
    # 读取图像文件夹
    print("\n读取图像文件夹...")
    if not os.path.isdir(image_dir):
        print(f"错误: 图像文件夹不存在: {image_dir}")
        return
    
    actual_images = set(os.listdir(image_dir))
    print(f"图像文件夹中的图像数: {len(actual_images)}")
    
    # 对比
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    
    # Caption中有但图像文件夹中没有的
    missing_in_folder = caption_images - actual_images
    print(f"\nCaption中有但图像文件夹中没有的图像: {len(missing_in_folder)} 张")
    if missing_in_folder:
        print("前20个缺失的图像:")
        for i, img in enumerate(sorted(missing_in_folder)[:20], 1):
            # 统计这个图像有多少个caption
            caption_count = len(df[df['image'] == img])
            print(f"  {i:2d}. {img} (有{caption_count}个caption)")
        if len(missing_in_folder) > 20:
            print(f"  ... 还有 {len(missing_in_folder) - 20} 个")
    
    # 图像文件夹中有但Caption中没有的
    missing_in_caption = actual_images - caption_images
    print(f"\n图像文件夹中有但Caption中没有的图像: {len(missing_in_caption)} 张")
    if missing_in_caption:
        print("前20个缺失的图像:")
        for i, img in enumerate(sorted(missing_in_caption)[:20], 1):
            print(f"  {i:2d}. {img}")
        if len(missing_in_caption) > 20:
            print(f"  ... 还有 {len(missing_in_caption) - 20} 个")
    
    # 匹配的图像
    matched_images = caption_images & actual_images
    print(f"\n匹配的图像（同时存在于Caption和图像文件夹）: {len(matched_images)} 张")
    
    # 统计caption数量
    matched_df = df[df['image'].isin(matched_images)]
    print(f"这些图像的caption总数: {len(matched_df)} 条")
    print(f"平均每张图像的caption数: {len(matched_df) / len(matched_images):.2f} 条")
    
    # 保存缺失图像列表
    if missing_in_folder:
        missing_file = "./storage/missing_images.txt"
        with open(missing_file, 'w', encoding='utf-8') as f:
            f.write("Caption中有但图像文件夹中没有的图像:\n")
            f.write(f"共 {len(missing_in_folder)} 张\n\n")
            for img in sorted(missing_in_folder):
                caption_count = len(df[df['image'] == img])
                f.write(f"{img} (有{caption_count}个caption)\n")
        print(f"\n缺失图像列表已保存到: {missing_file}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()




