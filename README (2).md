
# 🏎️ Fine-Grained Open-Vocabulary Detection with Stanford Car Dataset

This project aims to perform fine-grained open-vocabulary object detection on vehicle attributes using the Grounding DINO model, Stanford Car dataset, and FG-OVVD Benchmark.

---

## 📦 1. Install MMDetection and Grounding DINO

Please follow the official instructions to complete the installation:

👉 [MMDetection + Grounding DINO Installation Guide](https://github.com/open-mmlab/mmdetection/tree/main/configs/grounding_dino#installation)

---

## 📁 2. Custom Dataset Preparation

Follow the guide below to prepare your dataset and annotation format:

👉 [Custom Dataset for Grounding DINO](https://github.com/open-mmlab/mmdetection/tree/main/configs/grounding_dino#custom-dataset)

---

## 🚗 3. Stanford Car Dataset Setup

1. Organize the downloaded dataset in the following structure:

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

Ensure the images and annotations follow the structure above.

---

## ⚙️ 4. Configuration File Setup

Place the training config file `grounding_dino_swin-t_finetune_8xb2_20e_StanfordCarType.py` in:

```
configs/grounding_dino/
```

---

## 🚀 5. Start Training

After training completes, a `.pth` weight file will be saved in the folder specified by `--work-dir` (e.g., `StanfordCarType_work_dir/`).  
Be sure to specify the full path to this `.pth` file in `batch_infer_with_accuracy_Type_Diff.py` as the model checkpoint for inference.

Run the following command to start training:

```bash
./tools/dist_train.sh configs/grounding_dino/grounding_dino_swin-t_finetune_8xb2_20e_StanfordCarType.py 1 --work-dir StanfordCarType_work_dir
```

---

## 🧪 6. FG-OVVD Benchmark Setup

Clone the [FG-OVVD Benchmark](https://github.com/ccuvislab/FG-OVVD) and place it under your project as:

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

## 🧾 7. Inference Script Setup and Execution

1. Place the inference script `batch_infer_with_accuracy_Type.py` in:

```
demo/
```

2. Set the `input_path` and `output_path` inside the script.

3. Run inference:

```bash
python demo/batch_infer_with_accuracy_Type_Diff.py
```

4. Output results will be saved in:

```
outputs_StanfordCarType_work_dir/
├── preds/
└── vis/
```

---

## 📊 8. Top-K Accuracy Evaluation

1. Use `AccuracyScoreMulti.py` to calculate accuracy. Set `pred_dir` to the `/preds` path generated above.

2. Run:

```bash
python AccuracyScoreMulti.py
```

3. The output will show Top-K accuracy at different thresholds (e.g., Top-1 / Top-5).

- To calculate Top-5 accuracy, modify line 17 in `AccuracyScoreMulti.py` to:

```python
top_n_values = [1, 5]
```

---

## 📝 Additional Notes

**Important:**

- If errors occur, ensure annotations match the correct captions.
- In `demo/batch_infer_with_accuracy_Type_Diff.py`, make sure the keys at lines 60 and 61 are mapped properly:

```python
correct_type = entry['caption']  # Modify this
wrong_color_descriptions = entry['trivial_negatives']  # Modify this
```

If you are using the file `annotations_with_wrong_vehicle_categories_test_modified.json`, the fields are structured as:

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

Then the script should be updated as:

```python
correct_type = entry['correct_car_type']
wrong_color_descriptions = entry['wrong_car_types']
```

Always make sure the field names in your FG-OVVD caption JSON file match the code to ensure correct inference.

