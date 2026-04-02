import sys
import logging
import numpy as np
import pandas as pd
from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)

from Fluid_MM import Fliud_MMRec


def load_item_features(item_path, title_npy_path, max_item_id):
    """
    우리 데이터에서 피처 룩업 테이블 구성
    반환:
        image_feat: (max_item_id+1, image_dim) numpy float32  — 0번 인덱스 = padding
        text_feat : (max_item_id+1, text_dim)  numpy float32  — 0번 인덱스 = padding
    """
    item_df   = pd.read_parquet(item_path)
    title_raw = np.load(title_npy_path)          # (num_items, text_dim)

    # item_id 1-based 변환 (RecBole이 내부적으로 재인덱싱하므로
    # 여기서는 원본 0-based item_id 기준으로 구성 후 +1)
    item_df = item_df.copy().reset_index(drop=True)
    item_df['item_id'] = item_df['item_id'] + 1  # 1-based

    # ── 텍스트 피처 ────────────────────────────────────────────────────────
    text_dim  = title_raw.shape[1]
    text_feat = np.zeros((max_item_id + 1, text_dim), dtype=np.float32)
    for raw_idx, row in item_df.iterrows():
        iid = int(row['item_id'])
        if raw_idx < len(title_raw) and iid <= max_item_id:
            text_feat[iid] = title_raw[raw_idx].astype(np.float32)

    # ── 이미지/비디오 피처 ─────────────────────────────────────────────────
    sample_img = item_df['video_feature'].iloc[0]
    image_dim  = len(sample_img)
    image_feat = np.zeros((max_item_id + 1, image_dim), dtype=np.float32)
    for _, row in item_df.iterrows():
        iid = int(row['item_id'])
        if iid <= max_item_id:
            image_feat[iid] = np.array(row['video_feature'], dtype=np.float32)

    print(f"[load_item_features] text_dim={text_dim}, image_dim={image_dim}, "
          f"max_item_id={max_item_id}")
    return image_feat, text_feat


def main():
    parameter_dict = {
        'dataset': 'micro-lens-100k-mm',
        'config_file_list': ['config.yaml']
    }

    config = Config(
        model=Fliud_MMRec,
        dataset=parameter_dict['dataset'],
        config_file_list=parameter_dict['config_file_list']
    )

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # ── 데이터셋 생성 ────────────────────────────────────────────────────
    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    # ── 피처 룩업 테이블 로딩 ────────────────────────────────────────────
    # RecBole이 item을 1-based로 재인덱싱하므로 n_items - 1 이 max_item_id
    max_item_id = dataset.num(dataset.iid_field) - 1
    image_feat, text_feat = load_item_features(
        item_path     = config['item_path'],
        title_npy_path= config['title_npy_path'],
        max_item_id   = max_item_id,
    )

    # ── 모델 초기화 ──────────────────────────────────────────────────────
    init_seed(config['seed'], config['reproducibility'])
    model = Fliud_MMRec(
        config,
        train_data.dataset,
        image_feat=image_feat,
        text_feat=text_feat,
    ).to(config['device'])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config['device'], logger, transform)
    logger.info(set_color('FLOPs', 'blue') + f': {flops}')

    # ── 학습 & 평가 (RecBole Trainer 그대로) ────────────────────────────
    trainer = Trainer(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=config['show_progress']
    )

    test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        show_progress=config['show_progress']
    )

    environment_tb = get_environment(config)
    logger.info(
        'The running environment of this training is as follows:\n'
        + environment_tb.draw()
    )
    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    main()
