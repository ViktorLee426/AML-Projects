import os
import numpy as np

from utils import load_zipped_pickle, save_zipped_pickle

experiment_names = ['128_AE-all-unet++', '128_AE-all-unet++_imagenet', '128_A-ED-smp_unet_jacc', '128_AE-all-smp_unet_reg', '128_AE_added_win5-all_1-unet++_c5-4', 
                    '128_AE-all-manet-1', '128_AE-all-linknet-2']
experiments = []
combined = []
for name in experiment_names:
    preds = load_zipped_pickle(f"./results/{name}/pred_test.pkl")
    assert len(preds) == 20
    experiments.append(preds)

for i in range(20):
    combined_dict = {}
    combined_dict['name'] = experiments[0][i]['name']
    prediction = np.zeros_like(experiments[0][i]['prediction'].astype('float64'))
    for preds in experiments:
        assert preds[i]['name'] == combined_dict['name']
        prediction += preds[i]['prediction'].astype('float64')

    prediction = prediction >= 2.99

    combined_dict['prediction'] = prediction
    combined.append(combined_dict)

os.mkdir('results/combined')
save_zipped_pickle(combined, 'results/combined/pred_test.pkl')

