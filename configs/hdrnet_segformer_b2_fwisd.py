_base_ = './hdrnet_segformer_b2_common.py'

data_root = 'data/FWISD'
dataset_type = 'FWISDDataset'
num_classes = 11

# IMPORTANT: replace class names and mappings in mmseg_custom/datasets/disaster_datasets.py
# and below if your FWISD annotations use a different official id order.
model = dict(
    decode_head=dict(
        num_classes=num_classes,
        num_object_classes=8,
        num_damage_classes=2,
        damage_ordinal=False,
        # placeholder object ids: 0 bg, 1 water, 2 road, 3 vegetation, 4 vehicle,
        # 5 building/structure, 6 waterfront infrastructure, 7 debris/other
        object_class_mapping=[0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7],
        # placeholder damage ids: damaged structure/infrastructure/debris as 1
        damage_class_mapping=[0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
        damage_semantic_ids=[6, 8, 9]))

train_dataloader = dict(dataset=dict(
    _delete_=True,
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='images/train', seg_map_path='annotations/train'),
    pipeline={{_base_.train_pipeline}}))

val_dataloader = dict(dataset=dict(
    _delete_=True,
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='images/test', seg_map_path='annotations/test'),
    pipeline={{_base_.test_pipeline}}))

test_dataloader = val_dataloader
work_dir = './work_dirs/hdrnet_segformer_b2_fwisd'
