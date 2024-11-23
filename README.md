# mushroom_classifier

-   author: Benjamin Frizzell, Hankun Xiao, Yichi Zhang, Mingyang Zhang

This project is part of a data analysis demonstration for DSCI 522 (Data Science Workflows), a course in the Master of Data Science program at the University of British Columbia.

## About

Here's the corrected version of the text with typos fixed:

In this project, a Support Vector Classifier was built and tuned to identify mushroom edibility. A mushroom is classified as edible or poisonous based on attributes such as color, habitat, class, and others. The final classifier performed quite well on unseen test data, achieving a final overall accuracy of 0.99 and an F2 (beta = 2) score of 0.99. Furthermore, we used a confusion matrix to evaluate the accuracy of classifying mushrooms as poisonous or edible. The model made 12,174 correct predictions out of 12,214 test observations. However, there were 17 false negatives (predicting a poisonous mushroom as edible) and 23 false positives (predicting an edible mushroom as poisonous). The model’s performance shows promise for practical implementation, prioritizing safety by minimizing false negatives that could result in consuming poisonous mushrooms. While false positives may lead to unnecessarily discarding safe mushrooms, they pose no safety risk. Further development is needed to improve the model’s utility, focusing on enhancing performance and analyzing cases of incorrect predictions.

The dataset used in this project is the Secondary Mushroom Dataset created by Wagner, D., Heider, D., and Hattab, G., from the UCI Machine Learning Repository. This dataset contains 61,069 hypothetical mushrooms with caps based on 173 species (353 mushrooms per species). Each mushroom is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended (the latter class was combined with the poisonous class).

## Report

The final report can be found [here]().

## Usage

First time running the project, run the following command from the root of this repository:

``` bash
conda-lock install --name mushroom_classifier conda-lock.yml
```

To run the analysis, run the following command from the root of this repository:

``` bash
jupyter lab
```

Open `notebooks/Load_Data_and_EDA.ipynb` in Jupyter Lab and under Switch/Select Kernel choose "Python [conda env:environment]".

Next, under the "Kernel" menu click "Restart Kernel and Run All Cells...".

## Dependencies

-   `conda`
-   `conda-lock`
-   `jupyterlab`
-   `nb_conda_kernels`
-   Python and packages listed in [`environment.yml`](environment.yml)

## License

The mushroom_classifier is licensed under:

-   **Codebase**: MIT License. See [LICENSE.md](LICENSE.md).

-   **Reports and Visualizations**: Creative Commons Attribution-NonCommercial-NoDerivatives (CC BY-NC-ND 4.0).

## References

Dheeru, D., & Karra Taniskidou, E. (2017). Secondary Mushroom Dataset. UCI Machine Learning Repository. Retrieved from <https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset>

Scikit-learn developers. (n.d.). QuantileTransformer. Scikit-learn. Retrieved November 21, 2024, from <https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.QuantileTransformer.html>

Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3), 90–95.

McKinney, W. (2010). Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference, 51–56.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., … Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., … Oliphant, T. E. (2020). Array programming with NumPy. Nature, 585(7825), 357–362.

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., … van der Walt, S. J. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17, 261–272.
