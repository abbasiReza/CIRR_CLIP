

# Composed Image Retrieval with CLIP

## Overview
<p align="center">
  <img src="method.png" alt="Description of the image">
</p>

This project utilizes CLIP's Image Encoder and Text Encoder as backbones to train a model using contrastive learning for composed image retrieval. The project leverages 10,000 images for training and 1,500 images for testing from the NLVR dataset.

## Project Structure

| Folder/File       | Description                                                                                |
|-------------------|--------------------------------------------------------------------------------------------|
| `models/`         | Contains main model and different fusion models used to combine text and image embeddings. |
| `train.py`        | Script for training the model.                                                             |
| `eval.py`         | Script for evaluating the model.                                                           |
| `requirements.txt`| File containing the list of dependencies.                                                  |

## Requirements
```bash
pip install -r requirements.txt
```

## Training

To train the model, run the following command:

```bash
python train.py --data_path train_cirr.csv --batch_size 128 --num_epochs 20 --run_name simple --model_name ViT-L-14 --pretrained datacomp_xl_s13b_b90k --lr_text 3e-5 --lr_image 3e-4 --lr_fusion 3e-4 --save_path weights/
```

### Arguments:

- `--data_path`: Path to the training dataset.
- `--batch_size`: Batch size for training.
- `--num_epochs`: Number of epochs for training.
- `--run_name`: Name for the training run.
- `--model_name`: Name of the CLIP model variant.
- `--pretrained`: Pretrained weights to use.
- `--lr_text`: Learning rate for the text encoder.
- `--lr_image`: Learning rate for the image encoder.
- `--lr_fusion`: Learning rate for the fusion model.
- `--save_path`: Path to save the trained model weights.

## Evaluation

To evaluate the model, run the following command:

```bash
python eval.py --data_path train_cirr.csv --weights_path weights/contrastive_model_final_simple.pt --batch_size 128 --model_name ViT-L-14 --pretrained datacomp_xl_s13b_b90k --top_k 20
```

### Arguments:

- `--data_path`: Path to the evaluation dataset.
- `--weights_path`: Path to the trained model weights.
- `--batch_size`: Batch size for evaluation.
- `--model_name`: Name of the CLIP model variant.
- `--pretrained`: Pretrained weights to use.
- `--top_k`: Number of top results to retrieve.


