_base_ = './hdrnet_segformer_b2_common.py'

data_root = 'data/FWISD'
dataset_type = 'FWISDDataset'
num_classes = 11

# Coarse disaster classes used by DRCA:
# 0 water/flood, 1 building/infrastructure, 2 road, 3 vegetation, 4 vehicle/other
class_cooccurrence_matrix = [
    [1.0, 0.9, 0.7, 0.4, 0.3],
    [0.9, 1.0, 0.6, 0.4, 0.4],
    [0.7, 0.6, 1.0, 0.4, 0.4],
    [0.4, 0.4, 0.4, 1.0, 0.3],
    [0.3, 0.4, 0.4, 0.3, 1.0],
]

# IMPORTANT: replace class names and mappings in mmseg_custom/datasets/disaster_datasets.py
# and below if your FWISD annotations use a different official id order.
model = dict(
    decode_head=dict(
        num_classes=num_classes,
        num_coarse_classes=5,
        num_object_classes=8,
        num_damage_classes=2,
        damage_ordinal=False,
        class_cooccurrence_matrix=class_cooccurrence_matrix,
        # placeholder coarse ids: water/flood, building/infra, road, vegetation, vehicle/other
        coarse_class_mapping=[4, 0, 2, 3, 4, 1, 1, 1, 1, 4, 4],
        # placeholder object ids: 0 bg, 1 water, 2 road, 3 vegetation, 4 vehicle,
        # 5 building/structure, 6 waterfront infrastructure, 7 debris/other
        object_class_mapping=[0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7],
        # placeholder damage ids: damaged structure/infrastructure/debris as 1
        damage_class_mapping=[0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
        damage_semantic_ids=[6, 8, 9],
        loss_weights=dict(seg=1.0, coarse=0.1, obj=0.4, damage=0.6, hier=0.3, boundary=0.4)))

train_dataloader = dict(dataset=dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='images/train', seg_map_path='annotations/train'),
    pipeline={{_base_.train_pipeline}}))

val_dataloader = dict(dataset=dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(img_path='images/test', seg_map_path='annotations/test'),
    pipeline={{_base_.test_pipeline}}))

test_dataloader = val_dataloader
work_dir = './work_dirs/hdrnet_segformer_b2_fwisd'
