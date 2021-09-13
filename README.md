Deep Neural Network Prediction of Genome-Wide Transcriptome Signatures – beyond the Black-box
================
These are the scripts used to generate the results in the "Deep Neural Network Prediction of Genome-Wide Transcriptome Signatures – beyond the Black-box" project. All analyses were performed on the tetralith cluster of the National Supercomputer Centre (NSC) in Linköping, Sweden. 

The neural networks were built in the TF_to_target.py file, and an example model - TFs_to_targets_250_250.h5 is included in the set. Moreover, to do the light-up analyses, the write_model_sigmas.py script must be executed. This generates large csv files that the "validate_w_XX.py"-scripts can access. Please note that these files are large and require a substantial amount of RAM. The diseases can be analysed in the test_diseases.zip folder, using the 'main.py' script in the folder.

Dependencies
------------
The following versions were used in the project.

- [NumPy = 1.19.5](https://www.numpy.org/)

- [Pandas = 0.25.0](https://pandas.pydata.org/)

- [Scipy = 1.5.2](https://www.scipy.org/)

- [Keras = 2.4.3](https://keras.io/)

