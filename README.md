## 1. Download pre-trained swin transformer model (Swin-T)
* [Get pre-trained model in this link] (https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/"

## 2. Prepare data

- EndoVis 2017 [1] consisting of 8 X 225-frame sequences is used as train set and 2 X 300-frame sequences is used as test set. Instrument labels are Bipolar Forceps Prograsp Forceps Large Needle Driver Vessel Sealer Grasping Retractor Monopolar Curved Scissors Ultrasound Probe. 

    Download the EndoVis 2017 Dataset from [here](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/).

    Here we describe the steps for using the Endoscopic Vision 2017 [1] for instrument-type segmentation. As a preprocessing step we cropped black unindormative border from all frames with a file prepare_data.py that creates folder data/cropped_train with masks and images of the smaller size that are used for training. Then, to split the dataset for 4-fold cross-validation one can use the file: prepare_train_val.

- The Kvasir-Instrument dataset [2] includes 590 frames consisting of various GI endoscopy tools used during both endoscopic surveillance and therapeutic or surgical procedures. Moreover, information about the dataset uses, their application, annotation protocol can be found from their webpage.

    Download the Kvasir-Instrument Dataset from [here](https://datasets.simula.no/kvasir-instrument/).

- The format of dataset we used are provided by TransUnet's authors [3]. Please go to ["./datasets/README.md"](datasets/README.md) for details, or please send an Email to jienengchen01 AT gmail.com to request the preprocessed data (following the TransUnet's License). If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it [4].

## 3. Environment

- Please prepare an environment with python=3.8.

- Run the following code to install the Requirements.

    `pip install -r requirements.txt`

## 4. Train/Test

- Run the train script on synapse dataset. The batch size we used is 8. If you do not have enough GPU memory, the bacth size can be reduced to 4 or 2 to save memory.

- Train

```bash
    python train-ins.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your DATA_DIR --max_epochs 100 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.0001 --batch_size 8
```

- Test 

```bash
    python test_ins.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 100 --base_lr 0.0001 --img_size 224 --batch_size 8
```

--root_path     [Train data path]

--max_epochs   [train epoch]

--output_dir    [output path]

## References
[1] Allan, Max, et al. "2017 robotic instrument segmentation challenge." arxiv preprint arxiv:1902.06426 (2019).

[2] Jha, Debesh, et al. "Kvasir-instrument: Diagnostic and therapeutic tool segmentation dataset in gastrointestinal endoscopy." MultiMedia Modeling: 27th International Conference, MMM 2021, Prague, Czech Republic, June 22–24, 2021, Proceedings, Part II 27. Springer International Publishing, 2021.

[3] Chen, Jieneng, et al. "Transunet: Transformers make strong encoders for medical image segmentation." arxiv preprint arxiv:2102.04306 (2021).

[4] Cao, Hu, et al. "Swin-unet: Unet-like pure transformer for medical image segmentation." European conference on computer vision. Cham: Springer Nature Switzerland, 2022.


## Citation
```bibtex
@article{allan20192017,
  title={2017 robotic instrument segmentation challenge},
  author={Allan, Max and Shvets, Alex and Kurmann, Thomas and Zhang, Zichen and Duggal, Rahul and Su, Yun-Hsuan and Rieke, Nicola and Laina, Iro and Kalavakonda, Niveditha and Bodenstedt, Sebastian and others},
  journal={arXiv preprint arXiv:1902.06426},
  year={2019}
}
@inproceedings{jha2021kvasir,
  title={Kvasir-instrument: Diagnostic and therapeutic tool segmentation dataset in gastrointestinal endoscopy},
  author={Jha, Debesh and Ali, Sharib and Emanuelsen, Krister and Hicks, Steven A and Thambawita, Vajira and Garcia-Ceja, Enrique and Riegler, Michael A and de Lange, Thomas and Schmidt, Peter T and Johansen, H{\aa}vard D and others},
  booktitle={MultiMedia Modeling: 27th International Conference, MMM 2021, Prague, Czech Republic, June 22--24, 2021, Proceedings, Part II 27},
  pages={218--229},
  year={2021},
  organization={Springer}
}
@article{chen2021transunet,
  title={Transunet: Transformers make strong encoders for medical image segmentation},
  author={Chen, Jieneng and Lu, Yongyi and Yu, Qihang and Luo, Xiangde and Adeli, Ehsan and Wang, Yan and Lu, Le and Yuille, Alan L and Zhou, Yuyin},
  journal={arXiv preprint arXiv:2102.04306},
  year={2021}
}
@misc{cao2021swinunet,
      title={Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation}, 
      author={Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
      year={2021},
      eprint={2105.05537},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
