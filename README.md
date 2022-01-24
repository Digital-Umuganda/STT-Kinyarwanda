# Kinyarwanda STT

## Introduction

This README is a quick-start guide to training or ﬁnetuning an STT model using the [Coqui toolkit](https://github.com/coqui-ai/STT) on the [Kinyarwanda speech data](https://commonvoice.mozilla.org/en/datasets).

## Dockerﬁle Setup (Recommended)

To avoid problems setting up Coqui STT on your environment and compatibility issues, we recommend pulling or building the Coqui STT dockerﬁle from the `stt-train:latest` image:

```bash
$ git clone --recurse-submodules https://github.com/coqui-ai/STT
$ cd STT
$ docker build -f Dockerfile.train . -t stt-train:latest
$ docker run -it stt-train:latest
```

## Data Preprocessing

### Data Formatting

After downloading and extracting the dataset, we found the following contents:

- `.tsv` ﬁles, containing metadata such as text transcripts
- `.mp3` audio ﬁles, located in the clips directory

Coqui STT cannot directly work with Common Voice data, so we need the Coqui importer script [bin/import_cv2.py](https://github.com/coqui-ai/STT/blob/main/bin/import_cv2.py) to format the data correctly:

```bash
$ bin/import_cv2.py --validate_label_locale /path/to/validate_locale_rw.py /path/to/extracted/common-voice/archive
```

The importer script above would create `.csv` ﬁles from the `.tsv` ﬁles, and `.wav` ﬁles from the `.mp3` ﬁles.

The `--validate_label_locale` flag is optional but needed for data cleaning. The details on the input to the flag can be found in the [data cleaning](#data-cleaning) section below.

### Data Cleaning

1. As a way to clean the data, we need to validate the text. It checks a sentence to see if it can be converted and if possible normalizes the encoding, removes special characters, etc. For this we use the [commonvoice-utils](https://github.com/ftyers/commonvoice-utils) tool to clean the text for Kinyarwanda (rw).

The script below is passed as an argument to the `--validate_label_locale` flag in the importer command above

```python
#validate_locale_rw.py
from cvutils import Validator

def validate_label(label):
	v = Validator("rw") #rw - locale for Kinyarwanda. You should change accordingly.
	return v.validate(label)
```

1. We also need to ensure that each audio/input is longer than the transcript/output. This step is important so as not to run into training errors. As a result, we remove data from (train, dev, and test CSVs) that don’t meet this criterion.

```bash
$ python3 /path/to/remove_outliers.py /path/to/train.csv --clips_dir /path/to/clips
```

## Training

Since we would be training (and validating) our model within the docker environment we created initially, we need to ﬁrst create and run a container:

```bash
$ docker run -it --name sttcontainer -v ~/data:/code/data/host_data --gpus all stt-train:latest
```

The above command does the following:

- creates a container named `sttcontainer`
- bind mounts the `/data` directory on the host environment to the `/code/data/host_data` on the docker environment.
- gives the docker environment access to all the host GPUs

(Assuming we are within the docker environment:)

```bash
$ python -m coqui_stt_training.train \
	--checkpoint_dir data/host_data/jan-8-2021-best-kinya-deepspeech \
	--alphabet_config_path data/host_data/kinyarwanda_alphabet.txt \
	--n_hidden 2048 \
	--train_cudnn true \
	--train_files data/host_data/misc/lg-rw-oct2021/rw/clips/train.csv \
	--dev_files data/host_data/misc/lg-rw-oct2021/rw/clips/dev.csv \
	--test_files data/host_data/misc/lg-rw-oct2021/rw/clips/test.csv \
	--epochs 20 \
	--train_batch_size 128 \
	--dev_batch_size 128 \
	--test_batch_size 128 \
	--summary_dir data/host_data/tensorboard \
    --reduce_lr_on_plateau true
```

## Testing

By default, the trained model (after training) is tested on the test data at the end of the speciﬁed epoch, however, you can use a previously saved model on some test data:

```bash
$ python -m coqui_stt_training.train \
	--checkpoint_dir data/host_data/jan-8-2021-best-kinya-deepspeech \
	--alphabet_config_path data/host_data/kinyarwanda_alphabet.txt \
	--test_files data/host_data/misc/lg-rw-oct2021/rw/clips/test.csv \
	--test_batch_size 128
```

