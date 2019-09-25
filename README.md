| **Authors**  | **Project** |
|:------------:|:-----------:|
| [**N. Curti**](https://github.com/Nico-Curti) <br/> [**L. Dall'Olio**](https://github.com/Lorenzo-DallOlio) <br/> [**V. Recaldini**](https://github.com/valentinorecaldini)  |    Cardio   |

<a href="https://github.com/UniboDIFABiophysics">
<div class="image">
<img src="https://cdn.rawgit.com/physycom/templates/697b327d/logo_unibo.png" width="90" height="90">
</div>
</a>

# Cardio
### (Cardiological data processing)

The project is developed at the University of Bologna and all rights are reserved.

- [Cardio](#cardio)
    - [(Cardiological data processing)](#cardiological-data-processing)
  - [Prerequisites](#prerequisites)
  - [Description](#description)
  - [Authors](#authors)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Prerequisites

The project is written in python language and it uses the most common scientific packages (numpy, pandas, matplotlib, sklearn, ...), so the installation of [Anaconda](https://www.anaconda.com/) is recommended.

## Description

The project, still in progress, proposes to analyze cardiological data through a machine learning approach.
The analysis work-flow is the following:

- (1) [pre_process.py](https://github.com/Nico-Curti/cardio/blob/master/modules/pre_process.py):

  The first step is the pre-processing of raw data. In this step we perform the demodulation and the smoothing of the signal, focusing only on the red channel of an RGB signal (since G and B are mostly absorbed by skin). From now on we will call as signal the variation of R channel respect time. The smoothing is made possible through the use of a central simple moving average whose window length is equal to approximately 1 second (so its length expressed in number of points is numerically equivalent to the sampling frequency of the acquired signal divided by 1Hz). This procedure acts like a high-pass filter. Then the Hilbert transform of the signal is computed. This last procedure allows to obtain the analytic signal (a complex evaluated function whose real and imaginary part are one the Hilbert tranform of the other). From the analytic signal we can extract the instantaneous amplitude (also called envelope) and the instantaneous phase. Dividing the analytic signal element-wise by its envelope we obtain the demodulated signal which is sent as output. Please note that, due to the usage of a movinga average window, the output signal will be shorter than the input signal, with difference equal to the window length.

  
  - INPUT:
  
    Raw data --> data must be stored in such a way that data.R is the sequence of R values and data.Time is the ordered sequence of acquisition time values. For obvious reasons the signal must be longer than 1 second.
  
  - OUTPUT:

    high-pass filtered and demodulated signal, shortened by approximately 1 second, splitted in its signal values (stored into the variable "sign") and its ordered time values (stored into the variable "time").

- (2) [create_db.py](https://github.com/Nico-Curti/cardio/blob/master/modules/create_db.py): 
  
  The pre-process step is called inside the DB creation, in which we process each signal (joined with its information labels) and we extract many features in order to store them in a DB. the code implements the most common features of cardiological data, like the BPM (beats per minute), SDNN (standard deviation normal to normal), SDSD (standard deviation of successive difference), PNN20 (percentage normal to normal < 20ms), PNN50 (percentage normal to normal < 50ms) or TPR (turning point ratio) and some similar features. More complex (and dynamical) features like the mutual information and the embedding dimension (see. Takens theorem) are also computed.

  - INPUT:
  
    data directory and info directory (optionally different from data directory, but we used the same) where data and info are stored. Inside these directories filenames must be set to:
    -  yadda_number_data.txt for data
    -  yadda_number_info.txt for info

    where yadda can be any text not containing underscores ("_"), it is substatially ignored (so it can be different from data and info about the same patient), we highly recommend to use the same yadda for every data and every info; since data and info are first alphabetically sorted, writing different yadda can lead to wrong coupling of data and info. number must be a sequence of digits, the only request is to use the same number only twice, one for data and one for info, both of the same patient.

    Data needs to be a CSV and needs to contain a column called "Time" and a column called "R".
    
    Info needs to be a CSV in which the first row will be skipped (since it is the header), and the following rows have the structure

        FIELDNAME, VALUE
    
    Each FIELDNAME must be unique, and the necessary FIELDNAMEs are: Device, Sex, Age, Length, Weight, City, Country, Lifestyle, Smoking, Afib, Rhythm, Class.

    Each text is a valid VALUE, but some [replacements](https://github.com/Nico-Curti/cardio/blob/master/modules/create_db.py#L122) will be employed.

    Optional FIELDNAMEs can be added but will be discarded.

    For further information about the generation of the data and info files please see [here](https://github.com/Nico-Curti/cardio/blob/master/test/test_create_db.py).


  - OUTPUT:

    A .json file in the current working directory as *cardio.json*, containing a dictionary of dictionaries. The outer dictionary has the patient filename as key and, as value, an inner dictionary whose structure is shown [here](https://github.com/Nico-Curti/cardio/blob/master/modules/create_db.py#L250).

- [feature_extraction.py](https://github.com/Nico-Curti/cardio/blob/master/modules/clean_db.py):
  
  Composed of:
  

  - [clean_db.py](https://github.com/Nico-Curti/cardio/blob/master/modules/clean_db.py): 
    
    Python module before analyzing the data, the database must be cleaned; clean_db offer a way to load and clean it by removing useless columns or patients with incomplete or incorrect data wherever specified. It contains only functions; it cannot be executed as \_\_main\_\_.

    - INPUT:
    
      Raw data, we need a ... configuration to run pre_process properly.
    
    - OUTPUT:

      yadday... .

  - [sdppg_features.py](https://github.com/Nico-Curti/cardio/blob/master/modules/sdppg_features.py): 
    
    Python module to allow feature extraction from SDPPG, which is the second derivative of a PPG signal. A few examples of intersting features are the waves "a", "b", "c", "d", and "e" and the AGI index. It contains only functions; it cannot be executed as  \_\_main\_\_.

    - INPUT:
    
      Raw data, we need a ... configuration to run pre_process properly.
    
    - OUTPUT:

      yadday... .


  - [double_gaussian_features.py](https://github.com/Nico-Curti/cardio/blob/master/modules/double_gaussian_features.py): 
    
    Python module to allow feature extraction from dicrotic notch by performing a double gaussian fit over each beat of a signal; it cannot be executed as \_\_main\_\_.

    - INPUT:
    
      Raw data, we need a ... configuration to run pre_process properly.
    
    - OUTPUT:

      yadday... .

- [data_analysis](https://github.com/Nico-Curti/cardio/blob/master/modules/clean_db.py):
  
  When.

- Work in progress...

further information about called functions available [here](https://github.com/Nico-Curti/cardio/blob/master/doc.md)

## Authors

* **Nico Curti** [git](https://github.com/Nico-Curti), [unibo](https://www.unibo.it/sitoweb/nico.curti2)
* **Lorenzo Dall'Olio** [git](https://github.com/Lorenzo-DallOlio)
* **Valentino Recaldini** [git](https://github.com/valentinorecaldini)

See also the list of [contributors](https://github.com/Nico-Curti/cardio/contributors) [![GitHub contributors](https://img.shields.io/github/contributors/Naereen/StrapDown.js.svg)](https://GitHub.com/Nico-Curti/cardio/graphs/contributors/) who participated in this project.

## License

NO LICENSEs are available. All rights are reserved.

## Acknowledgments

Thanks goes to all contributors of this project:

[<img src="https://avatars0.githubusercontent.com/u/1419337?s=400&v=4" width="100px;"/><br /><sub><b>Enrico Giampieri</b></sub>](https://github.com/EnricoGiampieri) <br />
