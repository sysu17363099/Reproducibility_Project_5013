# Reproducibility_Project_5013

This project aims to ensure the reproducibility of experimental results through careful documentation and standardized procedures. Below is a guide on how to set up and run the project.




## Setup Instructions

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Set the `root_path` and `dataset_path` directly in the code as needed.
4. Install required packages.
5. Run the experiments.


## Parameter Description

The project includes test files that have been pre-configured with the necessary parameters for execution. To adapt the project to your specific environment, you will need to modify two critical paths.

1. **Root Path (`root_path`)**: This should point to the directory that contains the entire project. It is essential for the project to locate its files correctly.

2. **Dataset Path (`dataset_path`)**: This path specifies the location of the processed dataset that the project will utilize. Ensure that this path is correctly set to avoid data loading errors.


## Project Structure
- **/datapreprocessing**: Contains scripts for preprocessing data.
- **/evaluations**: Test files for different algorithms.
- **/filters**: Specific implementation code for various filter algorithms.
- **/framework**: The core framework of the project, including the main execution script and utility functions.
- **README.md**: This file, which provides an overview and instructions for the project.

##  More Datasets
Datasets can be downloarded from
* https://www.kaggle.com/datasets/vivovinco/19912021-nba-stats?select=players.csv
* https://www.kaggle.com/datasets/nishiodens/japan-real-estate-transaction-prices
* https://www.kaggle.com/datasets/ekibee/car-sales-information?select=region25_en.csv
* https://www.kaggle.com/datasets/rsrishav/youtube-trending-video-dataset?select=US_youtube_trending_data.csv, US region, version 800
* https://www.kaggle.com/datasets/fronkongames/steam-games-dataset


##  Describe which codes are newly developed
Based on the paper and open-source code, we have reproduced the implementation and testing of various filter algorithms. Apart from the preprocessing of the dataset, which we adopted directly (and of course, we also spent time debugging it, as the dataset has been updated), we have reimplemented other parts to gain more hands-on practice.
