
# 🏎️ Fine-Grained Open-Vocabulary Detection with Stanford Car Dataset

本專案旨在使用 Grounding DINO 模型，結合 Stanford Car 資料集與 FG-OVVD Benchmark，進行車輛屬性細分類別的 open-vocabulary 物件偵測任務。

---

## 📦 1. 安裝 MMDetection 與 Grounding DINO

請參考官方指引完成安裝：

👉 [MMDetection + Grounding DINO 安裝指南](https://github.com/open-mmlab/mmdetection/tree/main/configs/grounding_dino#installation)

---

## 📁 2. Custom Dataset 準備步驟

請依照下列教學準備自定義資料集格式與註解：

👉 [Custom Dataset for Grounding DINO](https://github.com/open-mmlab/mmdetection/tree/main/configs/grounding_dino#custom-dataset)

---

## 🚗 3. Stanford Car Dataset 準備

1. 將下載後的資料集整理如下：

```
data/StanfordCar/
├── annotations/
│   ├── train14label.json
│   ├── val14label.json
│   ├── test14label.json
│   └── stanford_car_label_map.json
├── images/
│   ├── cars_trainval/
│   └── cars_test/
```

請依照上圖所示格式放置圖片資料與標註檔。

---

## ⚙️ 4. Config 檔案設置

將訓練設定檔 `grounding_dino_swin-t_finetune_8xb2_20e_StanfordCarType.py` 放置於：

```
configs/grounding_dino/
```

---

## 🚀 5. 開始訓練模型

在完成訓練後，於你指定的 `--work-dir` 資料夾中（例如：`StanfordCarType_work_dir/`）會產生訓練完成的模型權重檔 `.pth`。
請記得在 `batch_infer_with_accuracy_Type_Diff.py` 中設定該 `.pth` 檔案的完整路徑作為推論用模型的 checkpoint。

使用以下指令進行訓練：

```bash
./tools/dist_train.sh configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_StanfordCarType.py 1 --work-dir StanfordCarType_work_dir
```

---

## 🧪 6. FG-OVVD Benchmark 準備

將 [FG-OVVD Benchmark](https://github.com/ccuvislab/FG-OVVD) clone 並放入專案目錄下的 `OV_BENCHMARK/`：

```
OV_BENCHMARK/
├── Difficulty/
│   ├── NEW_test_caption_results_15words.json
│   ├── NEW_test_caption_results_15words_easy_negatives_template.json
│   ├── ...
│   └── combined_color_model_wrong_models_modified.json
...
```

---

## 🧾 7. Inference 程式準備與執行

1. 將推論腳本 `batch_infer_with_accuracy_Type.py` 放入：

```
demo/
```

2. 設定推論腳本中 `input_path` 與 `output_path`。

3. 執行推論：

```bash
python demo/batch_infer_with_accuracy_Type_Diff.py
```

4. 輸出將出現在以下路徑中：

```
outputs_StanfordCarType_work_dir/
├── preds/
└── vis/
```

---

## 📊 8. Top-K 準確率計算

1. 使用 `AccuracyScoreMulti.py` 進行準確率統計，設定 `pred_dir` 為上步輸出的 `/preds` 路徑。

2. 執行計算：

```bash
python AccuracyScoreMulti.py
```

3. 輸出將顯示不同 `Thresholds` 下的 Top-K 正確率（Top-1 / Top-5 等）。

---

## 📝 備註補充

**注意事項：**

- 若遇錯誤請確認 annotations 與 caption 對應正確。
在 `demo/batch_infer_with_accuracy_Type_Diff.py` 中的第 60、61 行，請確保索引鍵名稱對應正確：

```python
correct_type = entry['caption']  # 此處需修改
wrong_color_descriptions = entry['trivial_negatives']  # 此處需修改
```

若使用的是 `annotations_with_wrong_vehicle_categories_test_modified.json` 檔案，其欄位如下所示：

```json
{
  "file_name": "00001.jpg",
  "correct_car_type": "a photo of a Sedan",
  "wrong_car_types": [
    "a photo of a Hatchback",
    "a photo of a ClubCab",
    ...
  ]
}
```

則應修改為：

```python
correct_type = entry['correct_car_type']
wrong_color_descriptions = entry['wrong_car_types']
```

請依據所選用的 FG-OVVD caption json 結構對應正確欄位名稱，以確保推論正確運作。
