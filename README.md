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

The file (script) below is passed as an argument to the `--validate_label_locale` flag in the importer command above

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
- gives the docker environment access to `all` the host GPU

The following assumes we are within the docker environment. If not, run `docker exec -it sttcontainer bash` to be in the environment:

```bash
# directory to save the training (loss) results
$ mkdir data/host_data/tensorboard
```

```bash
$ python -m coqui_stt_training.train \
	--load_checkpoint_dir data/host_data/jan-8-2021-best-kinya-deepspeech \
	--save_checkpoint_dir data/host_data/best-kinya-checkpoint \
	--alphabet_config_path data/host_data/kinyarwanda_alphabet.txt \
	--n_hidden 2048 \
	--train_cudnn true \
	--train_files data/host_data/misc/lg-rw-oct2021/rw/clips/train.csv \
	--dev_files data/host_data/misc/lg-rw-oct2021/rw/clips/dev.csv \
	--epochs 20 \
	--train_batch_size 128 \
	--dev_batch_size 128 \
	--summary_dir data/host_data/tensorboard
```

The below flags were explored to (experimentally) get a better model. You may wish to consider them (with intuition)

```
--learning_rate 0.00001 \
--reduce_lr_on_plateau true
--plateau_epochs 5 \
--dropout_rate 0.5
```

## Testing

By default, if a test file (with a test batch size) is specified in the training script (above), the trained model (after training) is tested on the test data at the end of the specified epoch. However, if you choose to omit the test file and test differently, you can use a previously saved model on some test data:

```bash
$ python -m coqui_stt_training.evaluate \
    --show_progressbar true \
    --train_cudnn true \
    --test_batch_size 128 \
    --test_files data/host_data/misc/lg-rw-oct2021/rw/clips/test.csv \
    --checkpoint_dir data/host_data/best-kinya-checkpoint
```

The above script will test only the acoustic model on the test data. This produces the WER for the acoustic model alone.

### Testing with a Language Model

If you have trained and generated a Language Model (scorer) previously, you can use it to produce a combined (overall) WER:

```bash
$ python -m coqui_stt_training.evaluate \
    --show_progressbar true \
    --train_cudnn true \
    --test_batch_size 128 \
    --test_output_file data/host_data/test_output \
    --test_files data/host_data/misc/lg-rw-oct2021/rw/clips/test.csv \
    --checkpoint_dir data/host_data/best-kinya-checkpoint \
    --scorer data/host_data/kinyarwanda.scorer
```

### Testing with a Language Model (with optimized alpha and beta)

If you have optimized your generated language model and have generated an optimized `--default_alpha` and `--default_beta` previously, you can use them to produce a better combined (overall) WER:

```bash
$ python -m coqui_stt_training.evaluate \
    --show_progressbar true \
    --train_cudnn true \
    --test_batch_size 128 \
    --test_output_file data/host_data/test_output \
    --test_files data/host_data/misc/lg-rw-oct2021/rw/clips/test.csv \
    --checkpoint_dir data/host_data/best-kinya-checkpoint \
    --scorer data/host_data/kinyarwanda_optm.scorer \
    --lm_alpha 0.6840899155626436 \
    --lm_beta 1.2497230003074578
```



## Language Model

You will usually want to deploy a language model in production. A good language model will improve transcription accuracy by correcting predictable spelling and grammatical mistakes. If you can predict what kind of speech your STT will encounter, you can make great gains in terms of accuracy with a custom language model.

*This section assumes that you are using a Docker image and container for training, as outlined in the [environment](#dockerﬁle-setup-(recommended)) section. If you are not using the Docker image, then some of the scripts such as `generate_lm.py` will not be available in your environment.*

*This section assumes that you have already trained an (acoustic) model and have a set of **checkpoints** for that model.*

### Generate binary and vocab files

The following assumes we are within the docker environment. If not, run `docker exec -it sttcontainer bash` to be in the environment:

```bash
$ python3 data/lm/generate_lm.py \
    --input_txt data/host_data/common_voice_kinyarwanda_kinnews_corpus.txt \
    --output_dir data/host_data/kinya_lm \
    --top_k 500000 \
    --kenlm_bins kenlm/build/bin \
    --arpa_order 5 \
    --max_arpa_memory "85%" \
    --arpa_prune "0|0|1" \
    --binary_a_bits 255 \
    --binary_q_bits 8 \
    --binary_type trie
```

The above script will save the new language model as two files in the specified output directory: `lm.binary` and `vocab-500000.txt`. The value 500000 comes from the specified value in the `--top_k` flag.

### Generate scorer

To generate our language model for use, we have to satisfy some environmental requirements. To achieve this, we do:

```bash
$ docker exec -it sttcontainer bash
$ export PATH=${STT_DIR_PATH}:$PATH
$ export $LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${STT_DIR_PATH}:${KENLM_BIN_PATH}:${STT_DIR_PATH}/data/lm

# E.g
# Since we are in using the docker environment,
# STT_DIR = /code
# KENLM_BIN = /code/kenlm/build/bin
$ export PATH=/code:$PATH
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/code:/code/kenlm/build/bin:/code/data/lm
```

**Note: The above steps need to be done every time we exit the docker environment.**

After this is done, we proceed to generate the scorer:

```bash
$ data/lm/generate_scorer_package \
    --checkpoint data/host_data/best-kinya-checkpoint \
    --lm data/host_data/kinya_lm/lm.binary \
    --vocab data/host_data/kinya_lm/vocab-500000.txt \
    --package data/host_data/kinyarwanda.scorer \
    --default_alpha 0.931289039105002 \
    --default_beta 1.1834137581510284
```

The above script will create a scorer called “kinyarwanda.scorer” in the `/data/host_data/` directory

The `--checkpoint` flag should point to the acoustic model checkpoint with which you will use the generated scorer.

The `--default_alpha` and `--default_beta` parameters shown above are optimized parameters and were found with the `lm_optimizer.py` Python script (on some data) and were used as a starting point. However, if you want to generate an optimized alpha and beta value specific to your data, do:

### Find the optimal values of Alpha and Beta (Optional)

The following assumes we are within the docker environment. If not, run `docker exec -it sttcontainer bash` to be in the environment:

```bash
$ python3 lm_optimizer.py \
    --show_progressbar true \
    --train_cudnn true \
    --test_batch_size 128 \
    --alphabet_config_path data/host_data/kinyarwanda_alphabet.txt \
    --scorer_path data/host_data/kinyarwanda.scorer \
    --test_files data/host_data/misc/lg-rw-oct2021/rw/clips/test.csv \
    --checkpoint_dir data/host_data/best-kinya-checkpoint \
    --n_hidden 2048 \
    --n_trials 300
```

`--n_hidden` should be the same as specified when training your (acoustic) model.

`--n_trials` specifies how many trials `lm_optimizer.py` should run to find the optimal values of `--default_alpha` and `--default_beta`. You may wish to reduce `--n_trials`.

### Generate (an optimized) scorer

If you have generated an optimized `alpha` and `beta` value specific to your data, you can pass them as values to `--default_alpha` and `--default_beta`. 

For example, on the Kinyarwanda data, the following values were found to be the best alpha and beta. Hence were used to generate an optimized scorer.

```bash
$ data/lm/generate_scorer_package \
  --checkpoint data/host_data/best-kinya-checkpoint \
  --lm data/host_data/kinya_lm/lm.binary \
  --vocab data/host_data/kinya_lm/vocab-500000.txt \
  --package data/host_data/kinyarwanda_optm.scorer \
  --default_alpha 0.6840899155626436 \
  --default_beta 1.2497230003074578
```