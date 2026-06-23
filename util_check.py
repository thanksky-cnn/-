# check_data.py
import os

print("=" * 60)
print("数据路径诊断")
print("=" * 60)

# 当前项目根目录
project_root = r"C:\Users\86152\PycharmProjects\arctic_seaice_prediction lstm SIE"
print(f"\n1. 项目根目录: {project_root}")
print(f"   文件夹是否存在: {os.path.exists(project_root)}")

# 检查 data 文件夹
data_folder = os.path.join(project_root, "data")
print(f"\n2. data 文件夹: {data_folder}")
print(f"   是否存在: {os.path.exists(data_folder)}")

if os.path.exists(data_folder):
    print(f"   data 文件夹内容: {os.listdir(data_folder)}")

# 检查 data/raw 文件夹
raw_folder = os.path.join(project_root, "data", "raw")
print(f"\n3. data/raw 文件夹: {raw_folder}")
print(f"   是否存在: {os.path.exists(raw_folder)}")

if os.path.exists(raw_folder):
    print(f"   raw 文件夹内容: {os.listdir(raw_folder)}")

    # 统计CSV文件
    csv_files = [f for f in os.listdir(raw_folder) if f.endswith('.csv')]
    print(f"\n   找到 {len(csv_files)} 个 CSV 文件:")
    for f in csv_files:
        print(f"      - {f}")
else:
    print("\n   ❌ data/raw 文件夹不存在！")
    print("   请创建这个文件夹")

# 检查你的CSV文件可能在哪里
print("\n" + "=" * 60)
print("4. 搜索当前目录下所有 CSV 文件")
print("=" * 60)

all_csv = []
for root, dirs, files in os.walk(project_root):
    for file in files:
        if file.endswith('.csv') and 'extent' in file.lower():
            all_csv.append(os.path.join(root, file))

if all_csv:
    print(f"找到 {len(all_csv)} 个 CSV 文件:")
    for f in all_csv:
        print(f"   {f}")
else:
    print("❌ 在当前项目中未找到任何 extent CSV 文件")
    print("   请先下载数据文件并放到正确位置")

print("\n" + "=" * 60)
print("建议操作:")
print("1. 如果 data/raw 文件夹不存在，请创建它")
print("2. 把你的12个CSV文件复制到 data/raw 文件夹中")
print("3. 确保文件名格式为: N_01_extent_v4.0.csv 到 N_12_extent_v4.0.csv")
print("=" * 60)