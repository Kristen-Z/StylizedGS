DATA_TYPE=$1
SCENE=$2
STYLE=$3

ckpt_gs=output/ckpt_gs/${DATA_TYPE}/${SCENE}
ckpt_stylegs=output/ckpt_stylegs/${DATA_TYPE}/${SCENE}_${STYLE}
data_dir=datasets/${DATA_TYPE}/${SCENE}
style_img=datasets/styles/${STYLE}.jpg


if [[ ! -f "${ckpt_gs}/point_cloud/iteration_30000/point_cloud.ply" ]]; then
    python train.py -s ${data_dir} \
                  -m ${ckpt_gs}
fi

python train_style.py -s ${data_dir} \
                -m ${ckpt_stylegs} \
                --point_cloud ${ckpt_gs}/point_cloud/iteration_30000/point_cloud.ply \
                --style ${style_img} \
                --histgram_match \
                # --preserve_color

python render.py -m ${ckpt_stylegs} \
                        --point_cloud ${ckpt_stylegs}/point_cloud/iteration_31500/point_cloud.ply \
                        --video --eval --fps 30
