# Herbarium

Toolkit I built for the [Herbarium 2022, FGVC9](https://www.kaggle.com/c/herbarium-2022-fgvc9) competition hosted by Kaggle.

The basic concept is to build a pipeline of operations, with each node being a generator. This allows iteration over 
the pipeline for each batch in the dataset. To modify the processing, swap nodes in the pipeline. 

There is no inversion of control, no callbacks, no event handlers.

It has a basic (and not entirely robust) configuration system that allows writing templated configuration files
(jinja2) for running training/prediction/etc trials. The configuration for the trial, as well as tensorboard data 
and any checkpoints are written to a snapshot directory so you have a full record of what has been run.

## Getting Started

### Installation

From the root of the repository:

MacOS:

    python3 -m venv pyenv
    source pyenv/bin/activate
    pip install -r requirements-macos.txt
    pip install -e .

Linux:

    python3 -m venv pyenv
    source pyenv/bin/activate
    pip install -r requirements-linux.txt
    pip install -e .

### Data Preparation

The configurations expect the data to be located at `~/Projects/datasets/fgvc9-herbarium-2022` in a subdirectory 
called `train_images_500`. These were created using a tool `prepare-data` to resize the images to 640x640 and then
crop centre at 500x500. The command used was:

    prepare-data ~/Projects/datasets/fgvc9-herbarium-2022/train_images \
                    ~/Projects/datasets/fgvc9-herbarium-2022/train_images_500 \
                    640x640 500x500

### Run an Example

    cd trials/000-mobilenet-v3-small
    ./run.sh

## Trials

In the `trials` directory are subdirectories for different model types. Each contains configurations for training 
and prediction, and shell scripts which show how to run the systems. 

The most up to date is `001-mobilenet-v3-large`; the others need some cleanup.

To understand the code:

* start with the run scripts to see examples of the commands and calling them
* take a look at the config files to see how it's all put together
* take a look at the pipeline nodes from the config file to see how they work
* take a look at the `train` etc. commands to see how they pull it all together

## Commands

The commands all have built-in help - run with the `--help` flag to see the full help. The `run.sh`
scripts have examples of how to use a lot of these.


| Command      | Description                                                |
|--------------|------------------------------------------------------------|
| prepare-data | resize and crop the data                                   |
| dsviewer     | a dataset viewer written in PySide6                        |
| dsinfo       | basic information about the dataset                        |
| minfo        | basic information about models                             |
| find-lr      | tool to find a reasonable learning rate                    |
| train        | training and validation                                    |
| validate     | validating a model with weights loaded from a checkpoint   |
| predict      | predicting and writing out the csv the competition needed  |
| explain      | explaining the predictions using LIME                      |
| swaify       | create a SWA model from a bunch of checkpoints             |
| grid-search  | takes a variables file and a configuration file and repeatedly calls a command with the variables as parameters |

