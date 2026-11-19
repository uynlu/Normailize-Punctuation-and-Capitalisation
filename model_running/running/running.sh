python model_running\\running\\running_model.py
    --pretrained-flag True
    --data-folder data\\official\\temp\\splitting
    --label-type toxicity
    --pretrained-name vinai/phobert-base
    --cache-dir model_checkpoints\\pretrained_models\\phobert
    --max-len 128
    --batch-size 16
    --num-labels 2
    --checkpoint-path model_checkpoints\\checkpoints\\toxicity\\phobert
    --freeze-model False
    --num-epochs 100 

python python model_running\\running\\running_model_evaluation.py
    # --pretrained-flag False
    --data-folder data\\official\\temp\\splitting
    --label-type toxicity
    --pretrained-name vinai/phobert-base
    --cache-dir model_checkpoints\\pretrained_models\\phobert
    --max-len 128
    --batch-size 16
    --num-labels 2
    --checkpoint-path model_checkpoints\\checkpoints\\toxicity\\phobert
    --freeze-model False
    --num-epochs 100 