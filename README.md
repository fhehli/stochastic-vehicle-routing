#  Stochastic Vehicle Routing
## Visually debugging the solution of the model
This branch provides a visual debugger (`visual_debugger.ipynb`) to compare the solution of the model with the ground truth.

### How to use the visual debugger
Run the cells in the notebook following the steps below:
1. Set the arguments in the config file. The default dataset path is `data/test.pkl` and the default config file is `configs/test.yaml`.
2. Create the dataset using `dataset_creator.py`, for example `python3 src/dataset_creator.py --city --n_samples 5 --n_tasks 4 --out_file data/test.pkl`. The `--city` flag indicates that the city instance related to a datapoint is stored in the dataset as well. If you use this flag, you have to set `city: true` in the data args. Without the `--city` flag, you will not see the visualized solution.
3. Run the trainer code in the notebook, and open the tensorboard using `tensorboard --logdir=runs` to view the solution. 
