| **Authors**  | **Project** |
|:------------:|:-----------:|
|   N. Curti   |    Cardio   |

<a href="https://github.com/UniboDIFABiophysics">
<div class="image">
<img src="https://cdn.rawgit.com/physycom/templates/697b327d/logo_unibo.png" width="90" height="90">
</div>
</a>

# Cardio
### (Cardiological data processing)

The project is developed at the University of Bologna and all rights are reserved.

1. [Prerequisites](#prerequisites)
2. [Description](#description)
3. [Authors](#authors)
4. [License](#license)
5. [Acknowledgments](#acknowledgments)

## Prerequisites

The project is written in python language and it uses the most common scientific packages (numpy, pandas, matplotlib, sklearn, ...), so the installation of [Anaconda](https://www.anaconda.com/) is recommended.

## Description

The project, still in progress, proposes to analyze cardiological data through a machine learning approach. The analysis work-flow is the following:

- [pre_process.py](https://github.com/Nico-Curti/cardio/blob/master/py/pre_process.py): The first step is the pre-processing of raw data. In this step we perform the demodulation of the signal and its smoothing, focusing on the red channel only.

- [create_db.py](https://github.com/Nico-Curti/cardio/blob/master/py/create_db.py): The first step is called inside the DB creation, in which we process each signal (joined with its information labels) and we extract many features. the code implements the most common features of cardiological data, like the BPM, SDNN, SDSD, PNN20, PNN50 until more complex (and dynamical) features like the mutual information and the embedding dimension (see. Takens theorem).

- Work in progress...

## Authors

* **Nico Curti** [git](https://github.com/Nico-Curti), [unibo](https://www.unibo.it/sitoweb/nico.curti2)

See also the list of [contributors](https://github.com/Nico-Curti/cardio/contributors) [![GitHub contributors](https://img.shields.io/github/contributors/Naereen/StrapDown.js.svg)](https://GitHub.com/Nico-Curti/cardio/graphs/contributors/) who participated in this project.

## License

NO LICENSEs are available. All rights are reserved.

## Acknowledgments

Thanks goes to all contributors of this project:

| [<img src="https://avatars0.githubusercontent.com/u/1419337?s=400&v=4" width="100px;"/><br /><sub><b>Enrico Giampieri</b></sub>](https://github.com/EnricoGiampieri)<br /> | [<img src="https://avatars3.githubusercontent.com/u/44815392?s=400&v=4" width="100px;"/><br /><sub><b>Lorenzo Dall'Olio</b></sub>](https://github.com/Lyro96)<br /> |
| :---: | :---: |
