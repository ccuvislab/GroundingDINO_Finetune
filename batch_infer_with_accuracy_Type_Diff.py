import os
import json
import argparse
from mmengine.logging import print_log
from mmdet.apis import DetInferencer

# 命令行參數設定
parser = argparse.ArgumentParser(description='Batch inference with controllable wrong color descriptions')
parser.add_argument('--max_wrong_colors', type=int, default=-1, 
                    help='Maximum number of wrong color descriptions to use (-1 for all)')
args = parser.parse_args()

# 設定
#config_file = 'configs/mm_grounding_dino/coco/grounding_dino_swin-t_finetune_16xb4_1x_stanfordcarType.py'
config_file = 'configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_StanfordCarType.py'
weights = 'StanfordCarType_work_dir/epoch_20.pth'
device = 'cuda:0'
image_dir = 'data/StanfordCar/images/cars_test/'
output_dir = 'outputs_StanfordCarType_work_dir/'
input_json_path = 'OV_Benchmark/Difficulty/NEW_test_caption_results_with_trivial_negatives.json'
output_json_path = 'inference_results_StanfordCarType_work_dir.json'

# 如果有指定wrong color數量，在檔名中標示
max_wrong_colors = args.max_wrong_colors
if max_wrong_colors > 0:
    output_dir = f'{output_dir}_max{max_wrong_colors}/'
    output_json_path = f'inference_results_color_max{max_wrong_colors}.json'

# 讀入資料
with open(input_json_path, 'r') as f:
    entries = json.load(f)

# 初始化推論器
inferencer = DetInferencer(
    model=config_file,
    weights=weights,
    device=device,
    palette='random'
)

# 設置chunked_size参數（這可能對大量類別有幫助）
#inferencer.model.test_cfg.chunked_size = -1  # 不要切分類別

# 儲存結果用
results = []

# 開始推論
for i, entry in enumerate(entries):
    try:
        file_name = entry['image_file']
        image_path = os.path.join(image_dir, file_name)
        
        # 檢查檔案是否存在
        if not os.path.exists(image_path):
            print(f"警告: 圖片不存在 {image_path}，跳過處理")
            continue
            
        correct_type = entry['correct_car_type'] # 這邊更改
        wrong_color_descriptions = entry['wrong_car_types'] # 這邊更改
        
        # 根據max_wrong_colors參數限制錯誤顏色的數量
        if max_wrong_colors > 0:
            wrong_color_descriptions = wrong_color_descriptions[:max_wrong_colors]
        
        # 文本格式與命令行相同，使用句點分隔
        prompts = [correct_type]
        for desc in wrong_color_descriptions:
            prompts.append(desc)
        
        text_prompt = '. '.join([p.rstrip('.') for p in prompts]) + '.'
        
        print(f'Processing {file_name}...')
        print(f'Using prompts: {text_prompt}')

        # 推論
        result = inferencer(
            inputs=image_path,
            texts=text_prompt,
            out_dir=output_dir,
            no_save_pred=False,
            no_save_vis=False,
            show=False,
            pred_score_thr=0.075,
            custom_entities=True,
        )

        # 檢查是否有正確類別被偵測出來
        pred_instances = result['predictions'][0]
        detected_labels = pred_instances['labels']
        
        correct_index = 0
        correct_detected = correct_index in detected_labels

        results.append({
            'file_name': file_name,
            'correct_car_type': correct_type,
            'wrong_colors_used': len(wrong_color_descriptions),
            'correct_detected': correct_detected,
            'num_detections': len(detected_labels),
            'detected_labels': detected_labels
        })
        
        # 每處理100張圖片保存一次結果
        if (i + 1) % 100 == 0:
            print(f"已處理 {i + 1}/{len(entries)} 張圖片，正在保存中間結果...")
            with open(f"{output_json_path}.partial", 'w') as f:
                json.dump(results, f, indent=2)
                
    except Exception as e:
        print(f"處理圖片 {file_name} 時發生錯誤: {str(e)}")
        # 記錄錯誤但繼續處理
        results.append({
            'file_name': file_name,
            'error': str(e),
            'processed': False
        })
        
        # 出錯時也保存一次結果
        with open(f"{output_json_path}.error", 'w') as f:
            json.dump(results, f, indent=2)
            
        continue

# 最終保存完整結果
with open(output_json_path, 'w') as f:
    json.dump(results, f, indent=2)

# 顯示 summary
# correct_count = sum(r.get('correct_detected', False) for r in results if 'correct_detected' in r)
# total = sum(1 for r in results if 'correct_detected' in r)
# accuracy = correct_count / total if total > 0 else 0
# print_log(f'Accuracy: {correct_count}/{total} = {accuracy:.2%}')
# print_log(f'使用的錯誤顏色數量上限: {max_wrong_colors if max_wrong_colors > 0 else "全部"}')
