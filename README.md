# Experiment results we saved
`classifier_comparison` | Linear, one-hidden-layer, and two-hidden-layer comparisons for 10, 20, and 40 countries that we used to validate our initial training performance
`data_growth_20261108` | Training summaries

`checksums.json` records the size and SHA-256 of every copied file and the original `source_hashes.json` identifies the implementation used for the final model. Source files in the snapshot are historical records
--> run the maintained scripts from the repository root

Files such as `split_summary.json` and `development_selection.json` record the state before final evaluation we did for slides, including `final_test_evaluated: false`. The later `holdout_status.json` records the completed evaluation.

The archived weighted training histories averaged batch losses using batch sizes --> `train.py` now uses the sum of class weights when combining weighted batch losses. Loss histories were stored to help monitor training results. The final 40-country holdout contains 2,435 images: 32.65% accuracy, macro F1 0.258, and 55.93% top-three accuracy.

To run the 40 country model -- the one we used in the demo -- use:
& ../.venv/Scripts/python.exe -B web_demo/server.py

AI useage:
Used AI to help create demo, and helped debug many instances of code. 