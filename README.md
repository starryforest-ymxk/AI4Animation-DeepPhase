# DeepPhase-DanceGeneration

This repository is forked from [sebastianstarke/AI4Animation](https://github.com/sebastianstarke/AI4Animation).

In this project, we focus on Dance Generation with DeepPhase. I added a custom dance pipeline in unity and also imported KTH Dance Dataset.

For music features extraction, I referred to the method of the Transflower paper and wrote the code. See more details in folder `/music features extraction`.

### Clone this repository

``` shell
git clone https://github.com/starryforest-ymxk/DeepPhase-DanceGeneration.git
```

### Pull the imported KTH Dance Dataset (If needed)

```shell
cd DeepPhase-DanceGeneration
git lfs pull
```

---

## Music Features Extract

Open the folder `/music features extraction` and install requirements with:

```shell
pip install -r requirements.txt
```

Then, you can download your dataset into `/music features extraction/data`. If the audio in your dataset is recorded in numpy format, you need to use `numpy2wav.py` to convert them to .wav format. You can specify the input and output folders in `numpy2wav.py` and then run this script.

Place your prepared audio files (.wav) into `/music_features_extraction/data`, and run `audio_features_extract.py` to extract audio features. The code will generate a subfolder named features, which contains the corresponding audio features for all the audio files stored in .txt format. You can also modify the input folder location or the sampling rate for audio feature extraction in `audio_features_extract.py`.

## Import Audio Features into Unity

In Unity, select `AI4Animation > Importer > Audio Spectrum Importer`. In the Source Path field, enter the folder containing the audio files, choose the output path, click 'Load Source Directory,' and then click 'Process' to wait for importing.

## Dance Generation Workflow

The process of generating dance motions is similar to the process of generating quadruped motions. You can find more information in [https://www.youtube.com/watch?v=3ASGrxNDd0k](https://github.com/sebastianstarke/AI4Animation)

Unlike the above process, when using our dance pipeline, you need to pay attention to the following points:

- When using the **Process Assets Mode** for the first time, you need to check the options for *Update Contact Sensors* (this is a time-consuming step) and *Update Audio Assets*. Additionally, you should specify the relevant audio path, which should contain AudioSpectrum files with the same names as all corresponding motion clips (these AudioSpectrum files are generated after importing the audio as described above).
- When using **Export Controller Mode**, it is recommended to use the pre-saved Music Series settings. For the specific meaning of Music Series, please refer to the related code.

You can also modify the dance pipeline to suit your own needs.

---

## Reference

- [sebastianstarke/AI4Animation: Bringing Characters to Life with Computer Brains in Unity (github.com)](https://github.com/sebastianstarke/AI4Animation)

- [Transflower | MetaGen](https://metagen.ai/transflower.html)

- [thyzju17/KTHDanceDataset (github.com)](https://github.com/thyzju17/KTHDanceDataset)
