
from load_tools import load_model_and_data
from config import *


from monitor.monitor_method import get_all_layers_stats
from monitor.plot_df import json_to_table, save_table_as_image, plot_heatmap
from monitor.plot_monitor import plot_all_layers_graph

# 讀取模型與資料
checkpoint_filename = 'RGB_SFMCNN_V2_best'
model, train_dataloader, test_dataloader, images, labels = load_model_and_data(checkpoint_filename)


layers = get_layers(model)
layers_infos = config['layers_infos']
# print(layers)
# print(layers_infos)


layer_stats, overall_stats = get_all_layers_stats(model, layers, layers_infos, images)

save_path = f'./detect/{config["dataset"]}_{checkpoint_filename}/RM_monitor'

# 確保保存目錄存在
os.makedirs(save_path, exist_ok=True)

# Convert to table
df = json_to_table(layer_stats, overall_stats )
print(df)

save_table_as_image(df, save_path + "/layers_table.png", title="Layer Metrics Table")
plot_heatmap(df, save_path + "/heat_map.png", title="Layer Metrics Table")

plot_all_layers_graph(model, layers, layers_infos, images, save_path, space_count=10)