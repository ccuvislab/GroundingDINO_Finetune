_base_ = 'grounding_dino_swin-t_finetune_16xb2_1x_coco.py'

# 配置數據根目錄
data_root = 'data/StanfordCar/'

# 創建開放式訓練流程
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=_base_.lang_model_name,
        num_sample_negative=20,
        label_map_file=data_root + 'annotations/stanford_car_label_map.json',
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]

# 讀取 label_map.txt 並提取類別名稱
class_name = (
    'SUV','Sedan','Coupe','Hatchback','Convertible','Wagon','CrewCab','ExtendedCab','RegularCab','SuperCab','Van','Minivan','QuadCab','ClubCab'
)
num_classes = len(class_name)  # 實際車輛類別數量（不包括背景類）
metainfo = dict(
    classes=class_name,  # 不包括背景類的類別列表
    palette=[(int(i * 255 / num_classes), int((1 - i / num_classes) * 255), int(127 + i * 128 / num_classes)) 
             for i in range(0, num_classes)]  # 從 1 開始生成調色板
)

# 數據加載器設定 - 使用ODVGDataset
train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='ODVGDataset',
        need_text=False,
        data_root=data_root,
        ann_file='annotations/train14label.json',
        label_map_file='annotations/stanford_car_label_map.json',
        data_prefix=dict(img='images/cars_trainval/'),
        return_classes=True,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline
    ),
    batch_size=4,
    num_workers=4,
    pin_memory=True
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/val14label.json',
        data_prefix=dict(img='images/cars_trainval/')
    ),
    batch_size=1,
    num_workers=4,
    pin_memory=True
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/test_updated_annotations_14_categories.json',
        data_prefix=dict(img='images/cars_test/')
    ),
    batch_size=1,
    num_workers=4,
    pin_memory=True
)

#評估器設定
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val14label.json',
    metric=['bbox'],
    classwise=True,
    format_only=False)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test_updated_annotations_14_categories.json',
    metric=['bbox'],
    classwise=True,
    format_only=False)

# 訓練設定
max_epoch = 20
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook', draw=True)
)
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

# 學習率調度器
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[15],
        gamma=0.1
    )
]

# 優化器設定
optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.0001),  # 給予極小學習率而非完全凍結
        }
    )
)

auto_scale_lr = dict(base_batch_size=32) 

# 修改 visualizer 設定
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='TensorboardVisBackend',
            save_dir='{{work_dir}}/tf_logs'
        )
    ],
    name='visualizer'
)

